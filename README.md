# Dance_Guru
# AI-Powered Kuchipudi Virtual Tutor

 **real-time AI-powered virtual tutor** for **Kuchipudi dance**, allowing students to compare their movements with their **Guru’s pre-recorded video**. The system provides **live pose estimation, movement similarity scoring, and real-time feedback** using **MediaPipe, OpenCV, and Dynamic Time Warping (DTW)**.

---

## 📌 Features
- 🎥 **Real-time Pose Tracking** using **MediaPipe**
- 🔍 **Live Comparison** with a Guru’s pre-recorded video
- 📊 **Dynamic Time Warping (DTW) Similarity Scoring**
- 🎯 **Gamification**: Score is displayed live on the screen
- 📡 **Seamless Webcam Integration**
- 📈 **Deployable on Lightning AI, Google Colab, or Local System**

---

## 📦 Installation
Ensure you have the necessary dependencies installed.

```bash
pip install lightning mediapipe opencv-python numpy fastdtw scipy pydub ffmpeg-python
```

For **Google Colab**, install additional dependencies:
```bash
!sudo apt update && sudo apt install v4l-utils
```

---

## 🚀 How It Works
1. **Guru’s Pose Extraction**: Pre-recorded video keypoints are extracted and saved.
2. **Live Student Tracking**: Captures real-time pose keypoints from webcam.
3. **Similarity Analysis**: Uses **DTW** to compare student vs. Guru poses.
4. **Scoring System**: Displays a **real-time similarity score** (0-100%).
5. **Live Feedback**: Students adjust their movements based on the score.



## 🏗️ Setup & Execution
### 1️⃣ **Prepare Guru's Pre-recorded Video**
Ensure you have a **Guru’s video file (MP4)** ready.

### 2️⃣ **Extract Guru’s Pose Keypoints**
Run the script to extract Guru’s pose keypoints and save them as `.npy`:
```python
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load Guru Video
guru_video = cv2.VideoCapture("guru_video.mp4")
guru_keypoints = []

while guru_video.isOpened():
    ret, frame = guru_video.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        guru_keypoints.append([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])

guru_video.release()
numpy.save("guru_poses.npy", np.array(guru_keypoints))
```

### 3️⃣ **Run the Live Comparison System**
Execute the following script to **compare real-time student movements with the Guru**:
```python
import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

guru_video = cv2.VideoCapture("guru_video.mp4")
guru_poses = np.load("guru_poses.npy")

student_cam = cv2.VideoCapture(0)
while student_cam.isOpened() and guru_video.isOpened():
    ret_guru, guru_frame = guru_video.read()
    if not ret_guru:
        guru_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    ret_student, student_frame = student_cam.read()
    if not ret_student:
        break
    
    rgb_student_frame = cv2.cvtColor(student_frame, cv2.COLOR_BGR2RGB)
    student_results = pose.process(rgb_student_frame)
    
    if student_results.pose_landmarks:
        student_keypoints = [(lm.x, lm.y) for lm in student_results.pose_landmarks.landmark]
        distance, _ = fastdtw(student_keypoints, guru_poses, dist=euclidean)
        similarity_score = max(0, 100 - distance)

        mp_drawing.draw_landmarks(student_frame, student_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(student_frame, f"Score: {int(similarity_score)}%", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    guru_frame = cv2.resize(guru_frame, (500, 500))
    student_frame = cv2.resize(student_frame, (500, 500))
    combined_frame = np.hstack((guru_frame, student_frame))
    
    cv2.imshow("Guru vs Student", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

student_cam.release()
guru_video.release()
cv2.destroyAllWindows()
```

---

## 🛠️ Debugging Issues
### **1️⃣ Webcam Not Opening**
If you see `can't open camera by index`, run:
```python
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        cap.release()
    else:
        print(f"❌ No camera at index {i}")
```
If another index (e.g., 1) works, change:
```python
student_cam = cv2.VideoCapture(1)
```

### **2️⃣ TensorFlow Lite Warnings**
Suppress them by adding:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

---

## 🚀 Deployment
### **🔹 Run on Lightning AI**
1. Install Lightning AI:
```bash
pip install lightning
```
2. Create a `lightning.yaml` config file:
```yaml
app: train.py
resources:
  - type: GPU
```
3. Deploy using:
```bash
lightning run app
```

### **🔹 Run on Google Colab**
- Upload `guru_video.mp4` and `guru_poses.npy`
- Enable webcam access (`cv2.VideoCapture(0)`) in a Colab cell

---


## To-Do :🎭 **Transform Online Kuchipudi Training with AI!** 🚀
- **Integrate with Zoom** to allow real-time feedback in live classes.
- **Develop an AI-powered virtual teacher** that suggests personalized movement corrections and improvements.
- **Enable live online class functionality** to facilitate remote learning with AI assistance.

✅ **Voice Feedback** – AI-based audio coaching  
✅ **Leaderboard & Gamification** – Track top students  
✅ **Support for Multiple Dance Forms** (Bharatanatyam, Odissi, etc.)  


## Contributing
Feel free to contribute by improving the feedback mechanism, optimizing performance, or adding new features.

## License
[MIT License](LICENSE)

