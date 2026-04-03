from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
from typing import List
import tempfile
import os
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize FastAPI
app = FastAPI(title="License Plate Detection API - Debug Version")

# Load model
MODEL_PATH = "best.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def extract_plate_text(plate_image):
    """Extract text from plate with debugging"""
    try:
        if plate_image is None or plate_image.size == 0:
            print("⚠️ Plate image is empty")
            return None
        
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Try OCR
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(cleaned, config=custom_config)
        text = text.strip().replace(' ', '').upper()
        text = text.replace('I', '1').replace('O', '0').replace('S', '5')
        
        print(f"🔤 OCR Raw text: '{text}' (length: {len(text)})")
        
        if len(text) >= 6:
            print(f"✅ Valid plate text: {text}")
            return text
        else:
            print(f"⚠️ Text too short (< 6 chars): {text}")
            return None
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return None

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """Debug version - shows detailed info"""
    try:
        print("\n" + "="*50)
        print("📥 Received image for detection")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("❌ Failed to decode image")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        print(f"✅ Image loaded: {image.shape}")
        
        if model is None:
            print("❌ Model not loaded!")
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Run detection
        print("🔍 Running YOLO detection...")
        results = model(image, conf=0.5, iou=0.3, max_det=10, verbose=False)
        
        print(f"📊 Detection results: {len(results)}")
        
        if results and len(results) > 0 and results[0].boxes is not None:
            num_boxes = len(results[0].boxes)
            print(f"🎯 Found {num_boxes} bounding boxes")
            
            detections = []
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                print(f"\n📦 Box #{i+1}: ({x1},{y1}) to ({x2},{y2}), conf={conf:.2f}")
                
                plate_roi = image[y1:y2, x1:x2]
                print(f"🖼️ Plate ROI shape: {plate_roi.shape if plate_roi.size > 0 else 'EMPTY'}")
                
                plate_text = extract_plate_text(plate_roi)
                
                if plate_text:
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "plate_text": plate_text
                    })
                    print(f"✅ Added plate: {plate_text}")
                else:
                    print(f"❌ No valid text extracted")
            
            # Filter duplicates
            if detections:
                detections.sort(key=lambda x: x["confidence"], reverse=True)
                filtered = []
                while detections:
                    best = detections.pop(0)
                    filtered.append(best)
                    detections = [
                        d for d in detections 
                        if calculate_iou(best["bbox"], d["bbox"]) < 0.5
                    ]
                
                plate_numbers = [d["plate_text"] for d in filtered]
                print(f"\n✅ Final plates: {plate_numbers}")
                print("="*50 + "\n")
                return JSONResponse(content={"plates": plate_numbers})
        
        print("⚠️ No plates detected")
        print("="*50 + "\n")
        return JSONResponse(content={"plates": []})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    unique_plates: bool = Query(True)
):
    try:
        print("\n" + "="*50)
        print("📥 Received video for detection")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            video_path = tmp.name
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video")
        
        all_plates = []
        seen_plates = set()
        frame_count = 0
        process_every_n_frames = 15
        
        print(f"🎬 Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % process_every_n_frames == 0:
                print(f"\n📹 Processing frame {frame_count}")
                
                if model:
                    results = model(frame, conf=0.5, iou=0.3, max_det=10, verbose=False)
                    
                    if results and len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            plate_roi = frame[y1:y2, x1:x2]
                            plate_text = extract_plate_text(plate_roi)
                            
                            if plate_text:
                                if unique_plates:
                                    if plate_text not in seen_plates:
                                        all_plates.append(plate_text)
                                        seen_plates.add(plate_text)
                                        print(f"✅ New plate found: {plate_text}")
                                else:
                                    all_plates.append(plate_text)
                                    print(f"✅ Plate found: {plate_text}")
            
            frame_count += 1
        
        cap.release()
        os.unlink(video_path)
        
        print(f"\n🎯 Total plates found: {all_plates}")
        print("="*50 + "\n")
        return JSONResponse(content={"plates": all_plates})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/rtsp")
async def detect_rtsp(
    rtsp_url: str = Query(...),
    duration: int = Query(30, ge=5, le=300),
    unique_plates: bool = Query(True)
):
    try:
        print("\n" + "="*50)
        print(f"📥 Connecting to RTSP: {rtsp_url}")
        
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print("❌ Cannot open RTSP stream")
            raise HTTPException(status_code=400, detail="Cannot open RTSP stream")
        
        print("✅ RTSP connected")
        
        all_plates = []
        seen_plates = set()
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(fps * duration)
        process_every_n_frames = max(1, int(fps / 2))
        
        import time
        start_time = time.time()
        
        print(f"⏱️ Processing for {duration} seconds...")
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % process_every_n_frames == 0:
                if model:
                    results = model(frame, conf=0.5, iou=0.3, max_det=10, verbose=False)
                    
                    if results and len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            plate_roi = frame[y1:y2, x1:x2]
                            plate_text = extract_plate_text(plate_roi)
                            
                            if plate_text:
                                if unique_plates and plate_text not in seen_plates:
                                    all_plates.append(plate_text)
                                    seen_plates.add(plate_text)
                                    print(f"✅ Plate: {plate_text}")
                                elif not unique_plates:
                                    all_plates.append(plate_text)
            
            frame_count += 1
            
            if time.time() - start_time > duration:
                break
        
        cap.release()
        
        print(f"\n🎯 Total plates: {all_plates}")
        print("="*50 + "\n")
        return JSONResponse(content={"plates": all_plates})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Starting License Plate Detection API")
    print(f"📦 Model: {MODEL_PATH}")
    print(f"🔧 Tesseract version: {pytesseract.get_tesseract_version()}")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)