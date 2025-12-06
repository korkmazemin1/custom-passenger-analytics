import cv2
import torch
import time
import numpy as np
import os
import sys
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cosine

# ====================================================
# PROJECT CONFIGURATION
# ====================================================

# ----------------------------------------------------
# ⚠️ PATH FIX (ABSOLUTE PATHS)
# ----------------------------------------------------
# We use absolute paths to prevent "File Not Found" errors
# regardless of where you run the terminal from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REID_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'osnet_ain_ms_d_c.pth')
YOLO_MODEL_NAME = os.path.join(BASE_DIR, 'weights', 'yolo11x.pt')

VIDEO_SOURCE = 0 

# ----------------------------------------------------
# ⚙️ MEMORY & REGISTRATION SETTINGS
# ----------------------------------------------------
# Similarity threshold to recognize a returning person
SIMILARITY_THRESHOLD = 0.65 

# UPDATED: We now wait for 50 frames (~2 seconds) before saving to memory.
# This ensures we get a clear, stable shot of the person.
FRAMES_TO_REGISTER = 50

# Databases
IDENTITY_GALLERY = {}
REGISTRATION_PROGRESS = {}

# ----------------------------------------------------
# ⚠️ DEVICE CONFIGURATION
# ----------------------------------------------------
if torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"✅ SYSTEM CHECK: NVIDIA GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print("⚠️ SYSTEM CHECK: Running on CPU (Performance might be low).")

# Color Palette
COLOR_SCANNING = (0, 255, 255)   # Yellow
COLOR_SAVED = (0, 255, 0)        # Green
COLOR_REID = (0, 165, 255)       # Orange
TEXT_COLOR = (255, 255, 255)

def get_similarity(v1, v2):
    """Calculates Cosine Similarity score."""
    return 1 - cosine(v1, v2)

def main():
    print(f"🚀 Initializing Smart Memory Pipeline...")
    
    # 1. FILE CHECKS
    # Check if the .pth file actually exists at the absolute path
    if not os.path.exists(REID_MODEL_PATH):
        print(f"\n❌ CRITICAL ERROR: Model file missing at: {REID_MODEL_PATH}")
        print("   Please make sure 'osnet_ain_ms_d_c.pth' is inside the 'weights' folder.")
        return
    else:
        print(f"   >> Found ReID weights at: {REID_MODEL_PATH}")

    # 2. LOAD MODELS
    print(f"📦 Loading YOLO ({YOLO_MODEL_NAME})...")
    try:
        # Check if local yolo file exists, otherwise let Ultralytics download it
        if os.path.exists(YOLO_MODEL_NAME):
            detector = YOLO(YOLO_MODEL_NAME)
        else:
            print("   >> Local YOLO weights not found, downloading 'yolo11x.pt'...")
            detector = YOLO('yolo11x.pt') # Fallback to auto-download

        if DEVICE == 'cuda': detector.to('cuda')
    except Exception as e:
        print(f"❌ YOLO Error: {e}")
        return
    
    print(f"🧠 Loading OSNet Memory Engine...")
    try:
        extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0', 
            model_path=REID_MODEL_PATH, 
            device=DEVICE
        )
    except Exception as e:
        print(f"❌ OSNet Error: {e}")
        return

    # 3. START STREAM (BUG FIX: cv2.CAP_DSHOW)
    # Using CAP_DSHOW fixes the MSMF error (-1072873822) on Windows
    print("🎥 Opening Camera with DirectShow...")
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ CRITICAL ERROR: Could not open video source. Camera might be busy.")
        return

    print("\n🟢 SYSTEM ONLINE. Stand still for registration!")
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Video stream ended or camera disconnected.")
            break

        current_time = time.time()
        use_half = (DEVICE == 'cuda')
        
        # ------------------------------------------------
        # A. TRACKING (YOLO + ByteTrack)
        # ------------------------------------------------
        results = detector.track(
            frame, 
            classes=0, 
            verbose=False, 
            conf=0.5, 
            persist=True,
            tracker="bytetrack.yaml",
            device=DEVICE, 
            half=use_half 
        )
        
        boxes = results[0].boxes

        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            coordinates = boxes.xyxy.int().cpu().tolist()

            for yolo_id, (x1, y1, x2, y2) in zip(track_ids, coordinates):
                
                # Boundary Check
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # ------------------------------------------------
                # B. EXTRACT FEATURE VECTOR
                # ------------------------------------------------
                person_crop = frame[y1:y2, x1:x2]
                
                # Quality filter: Ignore small/blurry crops
                if person_crop.size > 0 and person_crop.shape[0] > 50 and person_crop.shape[1] > 50:
                    features = extractor(person_crop)
                    current_vector = features.cpu().detach().numpy()[0]
                    
                    # ------------------------------------------------
                    # C. SMART REGISTRATION LOGIC
                    # ------------------------------------------------
                    
                    # 1. Check if this person is already known (Re-ID)
                    match_found = False
                    best_match_id = -1
                    best_score = 0
                    
                    for stored_id, stored_vector in IDENTITY_GALLERY.items():
                        score = get_similarity(current_vector, stored_vector)
                        if score > best_score:
                            best_score = score
                            best_match_id = stored_id

                    # ------------------------------------------------
                    # D. STATE DETERMINATION & VISUALIZATION
                    # ------------------------------------------------
                    
                    # CASE 1: KNOWN PERSON (Re-ID Match)
                    if best_score > SIMILARITY_THRESHOLD:
                        display_color = COLOR_REID
                        status_text = f"WELCOME BACK ID #{best_match_id}"
                        progress_ratio = 1.0 # Full bar
                        
                        # Reset registration counter for the YOLO ID since we know who it is
                        if yolo_id in REGISTRATION_PROGRESS:
                            del REGISTRATION_PROGRESS[yolo_id]

                    # CASE 2: UNKNOWN / NEW PERSON -> START SCANNING
                    else:
                        # Increment frame counter for this temporary YOLO ID
                        current_count = REGISTRATION_PROGRESS.get(yolo_id, 0) + 1
                        REGISTRATION_PROGRESS[yolo_id] = current_count
                        
                        # Calculate progress (0.0 to 1.0)
                        progress_ratio = min(current_count / FRAMES_TO_REGISTER, 1.0)

                        if current_count < FRAMES_TO_REGISTER:
                            # STATE: SCANNING
                            display_color = COLOR_SCANNING
                            pct = int(progress_ratio * 100)
                            status_text = f"ANALYZING... {pct}% (HOLD STILL)"
                        else:
                            # STATE: REGISTRATION COMPLETE
                            # Save to memory now!
                            if yolo_id not in IDENTITY_GALLERY:
                                IDENTITY_GALLERY[yolo_id] = current_vector
                            
                            display_color = COLOR_SAVED
                            status_text = f"ID #{yolo_id} SAVED. YOU MAY EXIT."

                    # ------------------------------------------------
                    # E. DRAW UI (HUD)
                    # ------------------------------------------------
                    # 1. Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)
                    
                    # 2. Draw Progress Bar (Above head)
                    bar_width = x2 - x1
                    bar_height = 10
                    
                    # Background bar
                    cv2.rectangle(frame, (x1, y1 - 15), (x2, y1 - 15 + bar_height), (50, 50, 50), -1)
                    
                    # Active bar
                    active_width = int(bar_width * progress_ratio)
                    cv2.rectangle(frame, (x1, y1 - 15), (x1 + active_width, y1 - 15 + bar_height), display_color, -1)

                    # 3. Draw Status Text
                    (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.putText(frame, status_text, (x1, y1 - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)

        # ------------------------------------------------
        # F. SYSTEM METRICS PANEL
        # ------------------------------------------------
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        cv2.rectangle(frame, (10, 10), (320, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 130), (255, 255, 255), 1)

        cv2.putText(frame, f"FPS: {fps:.1f}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"System: Smart Registration", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Req. Frames: {FRAMES_TO_REGISTER}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SCANNING, 1)
        cv2.putText(frame, f"Database Size: {len(IDENTITY_GALLERY)}", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_REID, 2)

        cv2.imshow('Smart Re-ID System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()