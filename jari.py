import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Konfigurasi ---
model_path = "hand_landmarker.task"

# Garis penghubung antar titik jari
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Jempol
    (0, 5), (5, 6), (6, 7), (7, 8),        # Telunjuk
    (0, 9), (9, 10), (10, 11), (11, 12),   # Tengah
    (0, 13), (13, 14), (14, 15), (15, 16), # Manis
    (0, 17), (17, 18), (18, 19), (19, 20), # Kelingking
    (5, 9), (9, 13), (13, 17)              # Telapak tangan
]

def to_pixel(x_norm, y_norm, w, h):
    x = min(max(x_norm, 0.0), 1.0)
    y = min(max(y_norm, 0.0), 1.0)
    return int(x * w), int(y * h)

def draw_landmarks(image_bgr, hand_landmarks_list):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        pts = [to_pixel(lm.x, lm.y, w, h) for lm in hand_landmarks]
        
        # Gambar garis (hijau)
        for a, b in HAND_CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
            
        # Gambar titik sendi (merah)
        for (x, y) in pts:
            cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
    return annotated

# --- Setup MediaPipe Tasks ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

print("🚀 Memulai Deteksi Jari Real-Time...")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Kamera tidak terdeteksi!")
            break
        
        # Balik gambar biar kaya ngaca
        frame = cv2.flip(frame, 1) 
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Proses deteksi
        result = landmarker.detect(mp_image)
        
        # Gambar overlay kalau jari terdeteksi
        if result.hand_landmarks:
            frame = draw_landmarks(frame, result.hand_landmarks)
            
        # Tampilkan hasil ke layar
        cv2.imshow('Deteksi Jari Real-Time', frame)
        
        # Tekan ESC buat keluar
        if cv2.waitKey(5) & 0xFF == 27: 
            break
            
cap.release()
cv2.destroyAllWindows()
