A real-time computer vision project that detects and counts fingers using a webcam feed. Leverages OpenCV's contour analysis and convexity defect algorithms (or MediaPipe's hand landmark model) to accurately identify finger positions and gestures with minimal latency and no external hardware required.
**Download Hand Landmarker Model:**
   Run this command in your terminal/PowerShell to automatically download the AI model:
```powershell
   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```
