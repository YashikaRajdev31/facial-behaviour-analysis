# Facial Behaviour Analysis System

A computer vision-based application that detects and analyses observable facial movements using Python, OpenCV, and MediaPipe.

## Features

- **Facial Landmark Detection:** Detects facial landmarks using MediaPipe Face Landmarker.
- **Eye Blink Detection:** Estimates eye blinks using the Eye Aspect Ratio (EAR).
- **Head Nod Detection:** Identifies vertical head movements using landmark positions, temporal smoothing, and a calibration-based state machine.
- **Real-Time Webcam Processing:** Supports webcam input.
- **Video Analysis:** Processes prerecorded video files.
- **Live Visualisation:** Displays facial landmarks, blink count, head-nod count, and detection status.

## Technologies Used

- Python 3.11
- OpenCV
- MediaPipe
- NumPy

## Project Structure

```text
facial-analysis-project/
├── src/
│   ├── __init__.py
│   └── main.py
├── models/
│   └── face_landmarker.task
├── videos/
│   └── test_video.mp4
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

Note: The model file and test videos are excluded from Git tracking.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd facial-analysis-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe model

Download the Face Landmarker model from the official MediaPipe model URL:

https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Place the downloaded file inside the `models` folder.

### 5. Run the application

```bash
python src/main.py
```

## Input Configuration

The application supports two input modes:

- Webcam input for live facial movement analysis.
- Prerecorded video input for offline analysis.

Select the required mode using the `USE_WEBCAM` configuration in `main.py`.

For video input, place the video inside the `videos` directory and configure the video path accordingly.

## How It Works

1. OpenCV captures frames from a webcam or video.
2. MediaPipe detects facial landmarks.
3. Eye Aspect Ratio is calculated to estimate eye closure and count blinks.
4. Relative nose and eye landmark positions are used to estimate vertical head movement.
5. A calibration-based detection system with temporal smoothing and cooldown logic helps identify head nods.
6. OpenCV displays the detection results.

## Limitations

- Head-nod detection is based on estimated vertical facial movement and may be affected by head rotation, camera angle, lighting, and facial landmark accuracy.
- Eye blink detection depends on reliable eye landmark detection.
- Detection results are estimates of observable movements, not clinical measurements.
- The application does not infer emotions, honesty, attention, personality, or intentions.

## Future Improvements

- Improve robustness across different camera angles and lighting conditions.
- Add analytical summaries and visual reports.
- Extend the system with additional observable facial movement metrics.

## Author

Developed as a computer vision project using Python and MediaPipe.