import cv2
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
from playsound import playsound
import tempfile

# Load the YOLO model
model = YOLO("yolov8s-pose.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Define the body part indices
body_index = {"left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}

# Initialize step count, previouqs positions, and thresholds
step_count = 0
prev_left_ankle_y = None
prev_right_ankle_y = None
step_threshold = 12
min_wait_frames = 8
wait_frames = 0

# Generate the 'Step' audio file
tts = gTTS(text="Step", lang="en")
temp_file = tempfile.NamedTemporaryFile(delete=False)
tts.save(temp_file.name)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame, verbose=False, conf=0.5)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Round the results to the nearest decimal
        rounded_results = np.round(results[0].keypoints.numpy(), 1)

        # Get the keypoints for the body parts
        try:
            left_knee = rounded_results[0][body_index["left_knee"]]
            right_knee = rounded_results[0][body_index["right_knee"]]
            left_ankle = rounded_results[0][body_index["left_ankle"]]
            right_ankle = rounded_results[0][body_index["right_ankle"]]

            if (
                (left_knee[2] > 0.5)
                and (right_knee[2] > 0.5)
                and (left_ankle[2] > 0.5)
                and (right_ankle[2] > 0.5)
            ):
                if (
                    prev_left_ankle_y is not None
                    and prev_right_ankle_y is not None
                    and wait_frames == 0
                ):
                    left_diff = abs(left_ankle[1] - prev_left_ankle_y)
                    right_diff = abs(right_ankle[1] - prev_right_ankle_y)

                    if max(left_diff, right_diff) > step_threshold:
                        step_count += 1
                        print(f"Step taken: {step_count}")
                        wait_frames = min_wait_frames

                prev_left_ankle_y = left_ankle[1]
                prev_right_ankle_y = right_ankle[1]

                if wait_frames > 0:
                    wait_frames -= 1

        except:
            print("No human detected.")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows
