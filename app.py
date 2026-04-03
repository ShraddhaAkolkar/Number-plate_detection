import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pytesseract
from PIL import Image
import tempfile
import json
from datetime import datetime
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Page config
st.set_page_config(
    page_title="Indian License Plate Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI - NEW COLOR SCHEME (Teal/Cyan)
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        min-height: 100vh;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95);
        box-shadow: -2px 0 10px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebarContent"] {
        padding: 2rem 1.5rem;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        margin: 1.5rem;
        padding: 2.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    h1, h2, h3 {
        color: #00838f;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #00838f, #00bcd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        color: #78909c;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .detection-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #00838f;
        box-shadow: 0 5px 15px rgba(0, 131, 143, 0.2);
    }
    
    .plate-text {
        font-family: 'Courier New', monospace;
        font-size: 2rem;
        font-weight: bold;
        color: #00838f;
        text-align: center;
        letter-spacing: 0.15em;
        padding: 1.2rem;
        background: #fff;
        border-radius: 12px;
        border: 3px solid #00bcd4;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 188, 212, 0.3);
    }
    
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00838f, #00bcd4);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        margin: 0.5rem 0.5rem 0.5rem 0;
        box-shadow: 0 4px 10px rgba(0, 131, 143, 0.3);
    }
    
    .info-box {
        background: #e0f7fa;
        border-left: 4px solid #00838f;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-text {
        color: #00897b;
        font-weight: 600;
    }
    
    .warning-text {
        color: #fb8c00;
        font-weight: 600;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00838f, #00bcd4) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(0, 131, 143, 0.4) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        background: transparent !important;
        border-bottom: 3px solid transparent !important;
        color: #00838f !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom-color: #00838f !important;
        color: #00838f !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #00838f, #00bcd4);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 131, 143, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "detection_results" not in st.session_state:
    st.session_state.detection_results = None

@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def filter_duplicate_detections(detections, iou_threshold=0.5):
    """Remove duplicate detections based on IoU"""
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    
    filtered_detections = []
    while detections:
        # Keep the detection with highest confidence
        best = detections.pop(0)
        filtered_detections.append(best)
        
        # Remove detections that overlap significantly with the best one
        detections = [
            det for det in detections 
            if calculate_iou(best["bbox"], det["bbox"]) < iou_threshold
        ]
    
    return filtered_detections

def extract_plate_text(plate_image):
    """Extract text from number plate using Tesseract"""
    try:
        # Preprocess the image for better OCR
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Better config for license plates
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # Extract text
        text = pytesseract.image_to_string(cleaned, config=custom_config)
        text = text.strip().replace(' ', '')
        
        return text if text else "OCR Failed"
        
    except Exception as e:
        st.warning(f"OCR processing note: {e}")
        return "Install Tesseract"

def detect_plates(image, model, confidence_threshold=0.5, iou_threshold=0.45):
    """Detect license plates in image with NMS"""
    if model is None:
        return None
    
    try:
        # Run inference with NMS parameters
        results = model(
            image, 
            conf=confidence_threshold,
            iou=iou_threshold,  # IoU threshold for NMS
            max_det=10,  # Maximum detections
            agnostic_nms=False
        )
        return results
    except Exception as e:
        st.error(f"Detection error: {e}")
        return None

def process_detections(image, results):
    """Process detection results and extract plate text"""
    detections = []
    annotated_image = image.copy()
    
    if results and len(results) > 0:
        result = results[0]
        
        if result.boxes is not None:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Extract plate region
                plate_roi = image[y1:y2, x1:x2]
                
                # Extract text if plate is visible
                plate_text = ""
                if plate_roi.size > 0:
                    plate_text = extract_plate_text(plate_roi)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add label
                label = f"Plate {len(detections) + 1}: {plate_text}"
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "plate_text": plate_text,
                    "plate_roi": plate_roi
                })
            
            # Filter duplicates AFTER collecting all detections
            detections = filter_duplicate_detections(detections, iou_threshold=0.5)
    
    return detections, annotated_image

# Main UI
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🚗 Indian License Plate Detection")
        st.markdown('<p class="subtitle">Detect and recognize Indian vehicle license plates using YOLOv8</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Model upload
    model_file = st.file_uploader(
        "📤 Upload your best.pt model file",
        type=["pt"],
        help="Upload your trained YOLO model (best.pt)"
    )
    
    if model_file:
        # Save uploaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            tmp_file.write(model_file.read())
            model_path = tmp_file.name
        
        st.session_state.model = load_model(model_path)
        if st.session_state.model:
            st.success("✅ Model loaded successfully!")
    else:
        st.info("👆 Upload a trained YOLO model to get started")
    
    st.divider()
    
    # Detection settings
    st.markdown("### 🎯 Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    iou_threshold = st.slider(
        "NMS IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="IoU threshold for Non-Maximum Suppression (lower = more aggressive filtering)"
    )
    
    st.divider()
    
    # Information
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app detects Indian license plates and extracts the plate numbers using:
    - **YOLOv8** for plate detection
    - **Tesseract OCR** for text extraction
    - **NMS** to remove duplicate detections
    
    **Requirements:**
    - Trained YOLO model (best.pt)
    - Pytesseract & Tesseract-OCR installed
    """)

# Main content
if st.session_state.model is None:
    st.warning("⚠️ Please upload a trained YOLO model in the sidebar to proceed.")
else:
    tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "📊 Results"])
    
    # Tab 1: Image Detection
    with tab1:
        st.markdown("### Upload Image for Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            image_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="image_upload"
            )
        
        with col2:
            image_source = st.radio(
                "Or use sample image from camera",
                ["Upload File", "Camera"],
                key="image_source",
                label_visibility="collapsed"
            )
        
        image_input = None
        
        if image_file:
            image_input = Image.open(image_file)
        elif image_source == "Camera":
            image_input = st.camera_input("Take a picture")
        
        if image_input is not None:
            # Convert to numpy array
            image_np = np.array(image_input)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Detect plates
            with st.spinner("🔍 Detecting license plates..."):
                results = detect_plates(image_bgr, st.session_state.model, confidence_threshold, iou_threshold)
                detections, annotated = process_detections(image_bgr, results)
            
            st.session_state.detection_results = detections
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Image")
                st.image(image_np, use_column_width=True)
            
            with col2:
                st.markdown("#### Detections")
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_column_width=True)
            
            # Display detections
            st.divider()
            
            if detections:
                st.markdown(f"### ✅ Found {len(detections)} License Plate(s)")
                
                for idx, detection in enumerate(detections, 1):
                    with st.container():
                        st.markdown(f'<div class="detection-card">', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Plate #{idx}**")
                            
                            # Display plate ROI
                            plate_rgb = cv2.cvtColor(detection["plate_roi"], cv2.COLOR_BGR2RGB)
                            st.image(plate_rgb, width=200)
                            
                            # Display extracted text
                            if detection["plate_text"] and detection["plate_text"] != "OCR Failed":
                                st.markdown(f'<div class="plate-text">{detection["plate_text"]}</div>', unsafe_allow_html=True)
                            else:
                                st.warning("⚠️ Could not extract text from this plate")
                        
                        with col2:
                            st.markdown("**Statistics**")
                            confidence = detection["confidence"]
                            st.markdown(
                                f'<div class="confidence-badge">Confidence: {confidence:.2%}</div>',
                                unsafe_allow_html=True
                            )
                            
                            x1, y1, x2, y2 = detection["bbox"]
                            width = x2 - x1
                            height = y2 - y1
                            
                            st.metric("Width (px)", width)
                            st.metric("Height (px)", height)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.divider()
            else:
                st.info("🔍 No license plates detected in the image. Try adjusting the confidence threshold.")
    
    # Tab 2: Video Detection
    with tab2:
        st.markdown("### Video Detection")
        
        video_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_upload"
        )
        
        if video_file:
            # Save video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(video_file.read())
                video_path = tmp_video.name
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("Processing video... This may take a moment.")
                process_video = st.button("🎬 Process Video")
            
            if process_video:
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                progress_bar = st.progress(0)
                all_detections = []
                
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every 5th frame for speed
                    if frame_idx % 5 == 0:
                        results = detect_plates(frame, st.session_state.model, confidence_threshold, iou_threshold)
                        detections, _ = process_detections(frame, results)
                        
                        if detections:
                            all_detections.append({
                                "frame": frame_idx,
                                "detections": detections
                            })
                    
                    frame_idx += 1
                    progress_bar.progress(min(frame_idx / frame_count, 1.0))
                
                cap.release()
                
                st.success(f"✅ Video processing complete!")
                st.markdown(f"### Results: Found plates in {len(all_detections)} frames")
                
                if all_detections:
                    for frame_result in all_detections[:5]:  # Show first 5 results
                        st.markdown(f"**Frame {frame_result['frame']}** - {len(frame_result['detections'])} plate(s) detected")
                        for detection in frame_result["detections"]:
                            st.write(f"  • {detection['plate_text']} (Confidence: {detection['confidence']:.2%})")
    
    # Tab 3: Results & Analytics
    with tab3:
        st.markdown("### Detection Results & Analytics")
        
        if st.session_state.detection_results:
            results = st.session_state.detection_results
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{len(results)}</h3><p>Total Plates</p></div>', unsafe_allow_html=True)
            
            with col2:
                avg_confidence = np.mean([r["confidence"] for r in results])
                st.markdown(f'<div class="metric-card"><h3>{avg_confidence:.1%}</h3><p>Avg Confidence</p></div>', unsafe_allow_html=True)
            
            with col3:
                high_confidence = len([r for r in results if r["confidence"] > 0.8])
                st.markdown(f'<div class="metric-card"><h3>{high_confidence}</h3><p>High Confidence</p></div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Export results
            st.markdown("### 📥 Export Results")
            
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "total_detections": len(results),
                "detections": [
                    {
                        "plate_number": r["plate_text"],
                        "confidence": float(r["confidence"]),
                        "coordinates": r["bbox"]
                    }
                    for r in results
                ]
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                "📥 Download Results as JSON",
                json_str,
                "detection_results.json",
                "application/json"
            )
        else:
            st.info("👆 Detect license plates in an image to see results here")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #78909c; padding: 2rem 0;">
    <p>🚗 Indian License Plate Detection v2.0 | Powered by YOLOv8 & Tesseract OCR</p>
    <p style="font-size: 0.9rem;">Built for accurate detection and recognition of Indian vehicle license plates</p>
</div>
""", unsafe_allow_html=True)