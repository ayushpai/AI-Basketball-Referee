# Import necessary libraries
import cv2
import numpy as np
import time
from ultralytics import YOLO
from singlestoredb import database

# Initialize connection to SingleStore DB
singlestore_db = database.SingleStoreDatabase(
    "mysql+pymysql://username:password@localhost:3306/databasename"
)

# Define DoubleDribbleDetector class
class DoubleDribbleDetector:
    def __init__(self):
        # Initialize YOLO models for pose and ball detection
        self.pose_model = YOLO("yolov8s-pose.pt")
        self.ball_model = YOLO("basketballModel.pt")
        # Initialize video capture from default camera
        self.cap = cv2.VideoCapture(0)
        # Define indices for left and right wrists in pose model
        self.body_index = {"left_wrist": 10, "right_wrist": 9}
        # Define state variables for the detector
        self.hold_start_time = None
        self.is_holding = False
        self.was_holding = False
        self.hold_duration = 0.85
        self.hold_threshold = 300
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None
        self.dribble_count = 0
        self.dribble_threshold = 18
        self.double_dribble_time = None
        self.frame_width = int(self.cap.get(3))

    # Main method for the detector
    def run(self):
        # While the video stream is open
        while self.cap.isOpened():
            # Read a frame from the stream
            success, frame = self.cap.read()
            if success:
                # If the frame was read successfully, process it
                pose_annotated_frame, ball_detected = self.process_frame(frame)
                # Check for double dribble after processing the frame
                self.check_double_dribble()
                # If double dribble was detected recently (<= 3 seconds ago),
                # add a red tint to the frame and display a warning text
                if (
                    self.double_dribble_time
                    and time.time() - self.double_dribble_time <= 3
                ):
                    red_tint = np.full_like(
                        pose_annotated_frame, (0, 0, 255), dtype=np.uint8
                    )
                    pose_annotated_frame = cv2.addWeighted(
                        pose_annotated_frame, 0.7, red_tint, 0.3, 0
                    )
                    cv2.putText(
                        pose_annotated_frame,
                        "Double dribble!",
                        (
                            self.frame_width - 600,
                            150,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 255, 255),
                        4,
                        cv2.LINE_AA,
                    )
                # Show the processed frame in a new window
                cv2.imshow("Basketball Referee AI", pose_annotated_frame)
                # If 'q' is pressed, stop the loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        # After the loop, release the video capture and destroy all windows
        self.cap.release()
        cv2.destroyAllWindows()

    # Method to process a frame
    def process_frame(self, frame):
        # Perform pose detection on the frame
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        # Draw pose detection results on the frame
        pose_annotated_frame = pose_results[0].plot()
        # Round pose detection results
        rounded_results = np.round(pose_results[0].keypoints.numpy(), 1)
        # Try to find left and right wrists in the pose detection results
        try:
            left_wrist = rounded_results[0][self.body_index["left_wrist"]]
            right_wrist = rounded_results[0][self.body_index["right_wrist"]]
        except:
            # If no human detected, print a warning and return
            print("No human detected.")
            return pose_annotated_frame, False

        # Perform ball detection on the frame
        ball_results_list = self.ball_model(frame, verbose=False, conf=0.65)
        ball_detected = False

        # For each detected ball
        for ball_results in ball_results_list:
            for bbox in ball_results.boxes.xyxy:
                # Compute center of the ball
                x1, y1, x2, y2 = bbox[:4]
                ball_x_center = (x1 + x2) / 2
                ball_y_center = (y1 + y2) / 2
                # Update dribble count and previous center of the ball
                self.update_dribble_count(ball_x_center, ball_y_center)
                self.prev_x_center = ball_x_center
                self.prev_y_center = ball_y_center
                # Mark ball as detected
                ball_detected = True

                # Calculate distances from the ball to the wrists
                left_distance = np.hypot(
                    ball_x_center - left_wrist[0], ball_y_center - left_wrist[1]
                )
                right_distance = np.hypot(
                    ball_x_center - right_wrist[0], ball_y_center - right_wrist[1]
                )
                # Check if the player is holding the ball
                self.check_holding(left_distance, right_distance)

                # Draw bounding box for the ball on the frame
                cv2.rectangle(
                    pose_annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )

                # If the player is holding the ball, add a blue tint to the frame
                if self.is_holding:
                    blue_tint = np.full_like(
                        pose_annotated_frame, (255, 0, 0), dtype=np.uint8
                    )
                    pose_annotated_frame = cv2.addWeighted(
                        pose_annotated_frame, 0.7, blue_tint, 0.3, 0
                    )

        # If no ball was detected, reset holding state
        if not ball_detected:
            self.hold_start_time = None
            self.is_holding = False

        # Return the frame and whether a ball was detected
        return pose_annotated_frame, ball_detected

    # Method to check if the player is holding the ball
    def check_holding(self, left_distance, right_distance):
        # If the ball is close to either wrist
        if min(left_distance, right_distance) < self.hold_threshold:
            # If this is the first frame where the ball is close,
            # start the timer
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            # If the ball has been close for longer than hold_duration,
            # mark the player as holding the ball
            elif time.time() - self.hold_start_time > self.hold_duration:
                self.is_holding = True
                self.was_holding = True
                # Reset dribble count
                self.dribble_count = 0
        else:
            # If the ball is not close to either wrist,
            # reset the timer and holding state
            self.hold_start_time = None
            self.is_holding = False

    # Method to update dribble count
    def update_dribble_count(self, x_center, y_center):
        # If there were previous frames with a ball
        if self.prev_y_center is not None:
            # Compute change in y-coordinate of the ball
            delta_y = y_center - self.prev_y_center
            # If the ball was moving up in the previous frame
            # and is moving down in the current frame (a dribble),
            # increment dribble count and update database
            if (
                self.prev_delta_y is not None
                and delta_y < 0
                and self.prev_delta_y > self.dribble_threshold
            ):
                self.dribble_count += 1
                # update SingleStore database
                singlestore_db.run(f"UPDATE dribble_counter SET count = {self.dribble_count} WHERE id = 1;")
            # Update previous change in y-coordinate
            self.prev_delta_y = delta_y

    # Method to check for double dribble
    def check_double_dribble(self):
        # If the player was holding the ball and then dribbled,
        # set double dribble timestamp, reset was_holding flag
        # and dribble count, and print a warning
        if self.was_holding and self.dribble_count > 0:
            self.double_dribble_time = time.time()
            self.was_holding = False
            self.dribble_count = 0
            print("Double dribble!")
            # reset dribble count in SingleStore database
            singlestore_db.run("UPDATE dribble_counter SET count = 0 WHERE id = 1;")


if __name__ == "__main__":
    # If this script is the main module, create a detector and run it
    detector = DoubleDribbleDetector()
    detector.run()
