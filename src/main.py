import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# project paths and configurations

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
VIDEO_PATH = BASE_DIR / "videos" / "test_video.mp4"

# Eye landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Head tracking landmarks
NOSE_LANDMARK = 1
EYE_REFERENCE_LANDMARKS = [33, 133, 362, 263]

# Blink detection settings
EAR_THRESHOLD = 0.18
MIN_CLOSED_FRAMES = 2
MAX_CLOSED_FRAMES = 12

closed_frames = 0
eyes_closed = False
blink_count = 0

# Head nod detection settings
CALIBRATION_FRAMES = 30

HEAD_DOWN_THRESHOLD = 0.035
HEAD_RETURN_THRESHOLD = 0.015

MIN_MOVEMENT_FRAMES = 3
COOLDOWN_SECONDS = 0.6

head_signal_history = []
calibration_values = []

head_baseline = None
head_state = "calibrating"

down_frames = 0
return_frames = 0
cooldown_frames_remaining = 0

nod_count = 0

# helper functions
def calculate_distance(point1, point2):

    return np.linalg.norm(
        np.array(point1) - np.array(point2)
    )

def calculate_ear(eye_landmarks):

    vertical_1 = calculate_distance(
        eye_landmarks[1],
        eye_landmarks[5]
    )

    vertical_2 = calculate_distance(
        eye_landmarks[2],
        eye_landmarks[4]
    )

    horizontal = calculate_distance(
        eye_landmarks[0],
        eye_landmarks[3]
    )

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * max(horizontal, 1e-6))

    return ear


def get_eye_landmarks(face_landmarks, eye_indices, width, height):

    eye_points = []

    for index in eye_indices:

        landmark = face_landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        eye_points.append((x, y))

    return eye_points


def calculate_head_signal(face_landmarks):

    nose_y = face_landmarks[NOSE_LANDMARK].y

    eye_y_values = [
        face_landmarks[index].y
        for index in EYE_REFERENCE_LANDMARKS
    ]

    eye_center_y = np.mean(eye_y_values)

    face_y_values = [
        landmark.y for landmark in face_landmarks
    ]

    face_height = max(face_y_values) - min(face_y_values)

    if face_height < 1e-6:
        return None

    # Nose position relative to the eye line,
    # normalized by face height.
    head_signal = (
        nose_y - eye_center_y
    ) / face_height

    return head_signal

# checking project files 

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

# mediapipe task options
BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),

    running_mode=RunningMode.VIDEO,

    num_faces=1
)

# opening video capture
# cap = cv2.VideoCapture(str(VIDEO_PATH))   (if pre recorded video is used)

USE_WEBCAM = True

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the input video."
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

original_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

original_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

COOLDOWN_FRAMES = int(
    fps * COOLDOWN_SECONDS
)

print("Video opened successfully!")

print(
    f"Resolution: {original_width} x {original_height}"
)

print(f"FPS: {fps}")


# creating display window
window_name = "Facial Behaviour Analysis"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    960,
    540
)

# face behaviour detection loop
frame_number = 0

try:

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:

                print("Video processing completed.")
                break

            height, width = frame.shape[:2]

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Generate video timestamp
            timestamp_ms = int(
                frame_number * 1000 / fps
            )

            # Detect facial landmarks
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # process detected frame 
            if result.face_landmarks:

                for face_landmarks in result.face_landmarks:

                    # blink detection 
                    right_eye_points = get_eye_landmarks(
                        face_landmarks,
                        RIGHT_EYE,
                        width,
                        height
                    )

                    left_eye_points = get_eye_landmarks(
                        face_landmarks,
                        LEFT_EYE,
                        width,
                        height
                    )

                    right_ear = calculate_ear(
                        right_eye_points
                    )

                    left_ear = calculate_ear(
                        left_eye_points
                    )

                    avg_ear = (
                        right_ear + left_ear
                    ) / 2.0

                    # Detect eye closure
                    if avg_ear < EAR_THRESHOLD:

                        closed_frames += 1
                        eyes_closed = True

                    else:

                        # Count only when eyes reopen
                        if eyes_closed:

                            if (
                                MIN_CLOSED_FRAMES
                                <= closed_frames
                                <= MAX_CLOSED_FRAMES
                            ):

                                blink_count += 1

                        closed_frames = 0
                        eyes_closed = False

                    # head nod detection
                    head_signal = calculate_head_signal(
                        face_landmarks
                    )

                    if head_signal is not None:

                        head_signal_history.append(
                            head_signal
                        )

                        # Keep only the latest 5 signals
                        if len(head_signal_history) > 5:

                            head_signal_history.pop(0)

                        # Smooth the movement signal
                        smoothed_signal = np.mean(
                            head_signal_history
                        )

                        # inintial head position calibration
                        if head_baseline is None:

                            calibration_values.append(
                                smoothed_signal
                            )

                            if (
                                len(calibration_values)
                                >= CALIBRATION_FRAMES
                            ):

                                head_baseline = float(
                                    np.median(
                                        calibration_values
                                    )
                                )

                                head_state = "neutral"

                                print(
                                    "Head calibration completed."
                                )

                    #    head nod detection logic
                        else:

                            if head_state == "neutral":

                                # Detect downward movement
                                if (
                                    smoothed_signal
                                    > head_baseline
                                    + HEAD_DOWN_THRESHOLD
                                ):

                                    down_frames += 1

                                else:

                                    down_frames = 0

                                # Confirm downward movement
                                if (
                                    down_frames
                                    >= MIN_MOVEMENT_FRAMES
                                ):

                                    head_state = "down"

                                    return_frames = 0
                                    down_frames = 0

                            elif head_state == "down":

                                # Detect return to neutral
                                if (
                                    smoothed_signal
                                    < head_baseline
                                    + HEAD_RETURN_THRESHOLD
                                ):

                                    return_frames += 1

                                else:

                                    return_frames = 0

                                # Count one complete nod
                                if (
                                    return_frames
                                    >= MIN_MOVEMENT_FRAMES
                                ):

                                    nod_count += 1

                                    head_state = "cooldown"

                                    cooldown_frames_remaining = (
                                        COOLDOWN_FRAMES
                                    )

                                    return_frames = 0

                                    print(
                                        f"Head nod detected! "
                                        f"Total nods: {nod_count}"
                                    )

                            elif head_state == "cooldown":

                                cooldown_frames_remaining -= 1

                                if (
                                    cooldown_frames_remaining <= 0
                                ):

                                    head_state = "neutral"

                    # draw facial landmarks
                    for landmark in face_landmarks:

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            1,
                            (0, 255, 0),
                            -1
                        )

                    # Highlight nose position
                    nose = face_landmarks[NOSE_LANDMARK]

                    nose_x = int(nose.x * width)
                    nose_y = int(nose.y * height)

                    cv2.circle(
                        frame,
                        (nose_x, nose_y),
                        8,
                        (0, 0, 255),
                        -1
                    )

                    #  display facial information
                    if head_baseline is None:
                        head_status = "Calibrating..."
                    else:
                        head_status = head_state.capitalize()

                    # Display labels
                    display_texts = [
                        "Face Detected",
                        f"Blinks: {blink_count}",
                        f"Head Nods: {nod_count}",
                        f"Head: {head_status}"
                    ]

                    # Common text styling
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9
                    font_color = (255, 255, 255)
                    font_thickness = 2

                    start_x = 30
                    start_y = 40
                    line_spacing = 40

                    # Calculate background size
                    max_text_width = max(
                        cv2.getTextSize(
                            text,
                            font,
                            font_scale,
                            font_thickness
                        )[0][0]
                        for text in display_texts
                    )

                    background_height = len(display_texts) * line_spacing + 15

                    # Dark background for readability
                    overlay = frame.copy()

                    cv2.rectangle(
                        overlay,
                        (15, 10),
                        (max_text_width + 50, background_height),
                        (30, 30, 30),
                        -1
                    )

                    # Blend background with original frame
                    cv2.addWeighted(
                        overlay,
                        0.65,
                        frame,
                        0.35,
                        0,
                        frame
                    )

                    # Draw all labels consistently
                    for index, text in enumerate(display_texts):

                        y_position = start_y + index * line_spacing

                        cv2.putText(
                            frame,
                            text,
                            (start_x, y_position),
                            font,
                            font_scale,
                            font_color,
                            font_thickness,
                            cv2.LINE_AA
                        )

            #  if no face is detected 
            else:

                # Avoid carrying an incomplete blink
                # across a period without face detection
                closed_frames = 0
                eyes_closed = False

                cv2.putText(
                    frame,
                    "No Face Detected",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

        #     resizing the frame for display
            screen_width = 960
            screen_height = 540

            scale = min(
                screen_width / width,
                screen_height / height
            )

            display_width = int(
                width * scale
            )

            display_height = int(
                height * scale
            )

            display_frame = cv2.resize(
                frame,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA
            )

            cv2.imshow(
                window_name,
                display_frame
            )

            frame_number += 1

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Processing stopped by user.")
                break

finally:

    #  release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released successfully.")
