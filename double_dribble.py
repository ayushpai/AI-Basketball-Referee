import cv2
import numpy as np
import time
from ultralytics import YOLO


# The DoubleDribbleDetector uses computer vision to detect when a player in a basketball game
# has committed a double dribble. This is achieved through tracking the position of the basketball
# and the player's wrists, detecting when the ball is held and then determining if the player
# starts another dribble.
class DoubleDribbleDetector:
    def __init__(self):
        # Load YOLO (You Only Look Once), a popular object detection model.
        # One model is trained for pose estimation, and the other for detecting the basketball.
        self.pose_model = YOLO("yolov8s-pose.pt")
        self.ball_model = YOLO("basketballModel.pt")

        # Initialize the video capture object to capture video from the default camera.
        self.cap = cv2.VideoCapture(0)

        # The indices of the left and right wrists in the pose model's output keypoints array.
        self.body_index = {"left_wrist": 10, "right_wrist": 9}

        # Various variables used to keep track of state during the detection process.
        self.hold_start_time = None
        self.is_holding = False
        self.was_holding = False

        # Hold duration and distance threshold for considering a ball as held.
        self.hold_duration = 0.85
        self.hold_threshold = 300

        # Used for tracking the previous frame's center of the ball.
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None

        # Used for keeping track of dribble count and dribble threshold for
        # considering a ball as dribbled.
        self.dribble_count = 0
        self.dribble_threshold = 18

        # Timestamp when a double dribble is detected.
        self.double_dribble_time = None

        # Get the frame width for positioning text in the frame.
        self.frame_width = int(self.cap.get(3))

    # The main loop where video frames are read and processed.
    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                # Process the frame to detect the pose and the basketball.
                pose_annotated_frame, ball_detected = self.process_frame(frame)

                # Check if a double dribble has occurred.
                self.check_double_dribble()

                # If a double dribble was detected recently, tint the frame red.
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

                    # Also display a "Double dribble!" message on the frame.
                    cv2.putText(
                        pose_annotated_frame,
                        "Double dribble!",
                        (
                            self.frame_width - 600,
                            150,
                        ),  # You might need to adjust these values
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 255, 255),
                        4,
                        cv2.LINE_AA,
                    )

                # Display the frame.
                cv2.imshow("Basketball Referee AI", pose_annotated_frame)

                # Break the loop if 'q' is pressed.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        # Release resources.
        self.cap.release()
        cv2.destroyAllWindows()

    # Process a frame to detect the pose and the basketball.
    def process_frame(self, frame):
        # Pass the frame through the pose model.
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)

        # Create a copy of the frame with pose annotations.
        pose_annotated_frame = pose_results[0].plot()

        # Get the detected keypoints and round their coordinates for simplicity.
        rounded_results = np.round(pose_results[0].keypoints.numpy(), 1)

        try:
            # Extract the coordinates of the wrists.
            left_wrist = rounded_results[0][self.body_index["left_wrist"]]
            right_wrist = rounded_results[0][self.body_index["right_wrist"]]
        except:
            # If no human was detected, return the annotated frame as it is.
            print("No human detected.")
            return pose_annotated_frame, False

        # Pass the frame through the ball model.
        ball_results_list = self.ball_model(frame, verbose=False, conf=0.65)

        ball_detected = False

        # Iterate through the detected bounding boxes.
        for ball_results in ball_results_list:
            for bbox in ball_results.boxes.xyxy:
                # Get the coordinates of the bounding box.
                x1, y1, x2, y2 = bbox[:4]

                # Compute the center of the bounding box.
                ball_x_center = (x1 + x2) / 2
                ball_y_center = (y1 + y2) / 2

                # Update the dribble count based on the motion of the ball.
                self.update_dribble_count(ball_x_center, ball_y_center)

                # Store the center of the ball for the next frame's calculations.
                self.prev_x_center = ball_x_center
                self.prev_y_center = ball_y_center

                # A ball was detected in this frame.
                ball_detected = True

                # Compute the distance from each wrist to the ball.
                left_distance = np.hypot(
                    ball_x_center - left_wrist[0], ball_y_center - left_wrist[1]
                )
                right_distance = np.hypot(
                    ball_x_center - right_wrist[0], ball_y_center - right_wrist[1]
                )

                # Check if the ball is being held.
                self.check_holding(left_distance, right_distance)

                # Annotate the frame with the bounding box of the ball.
                cv2.rectangle(
                    pose_annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )

                # Annotate the frame with various information for debugging.
                cv2.putText(
                    pose_annotated_frame,
                    f"Ball: ({ball_x_center:.2f}, {ball_y_center:.2f})",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    pose_annotated_frame,
                    f"Left Wrist: ({left_wrist[0]:.2f}, {left_wrist[1]:.2f})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    pose_annotated_frame,
                    f"Right Wrist: ({right_wrist[0]:.2f}, {right_wrist[1]:.2f})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    pose_annotated_frame,
                    f"Differentials: ({min(left_distance, right_distance):.2f})",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    pose_annotated_frame,
                    f"Holding: {'Yes' if self.is_holding else 'No'}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    pose_annotated_frame,
                    f"Dribble count: {self.dribble_count}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                # If the player is holding the ball, tint the frame blue for emphasis.
                if self.is_holding:
                    blue_tint = np.full_like(
                        pose_annotated_frame, (255, 0, 0), dtype=np.uint8
                    )
                    pose_annotated_frame = cv2.addWeighted(
                        pose_annotated_frame, 0.7, blue_tint, 0.3, 0
                    )

        # If no ball was detected in this frame, reset the holding state.
        if not ball_detected:
            self.hold_start_time = None
            self.is_holding = False

        return pose_annotated_frame, ball_detected

    # Check if the ball is being held based on its proximity to the wrists.
    def check_holding(self, left_distance, right_distance):
        # If the ball is close to either wrist...
        if min(left_distance, right_distance) < self.hold_threshold:
            # If this is the first frame in which the ball is being held, record the current time.
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            # If the ball has been held for longer than the allowed duration, mark it as being held.
            elif time.time() - self.hold_start_time > self.hold_duration:
                self.is_holding = True
                self.was_holding = True
                self.dribble_count = 0
        else:
            # If the ball is not close to either wrist, reset the holding state.
            self.hold_start_time = None
            self.is_holding = False

    # Update the dribble count based on the motion of the ball.
    def update_dribble_count(self, x_center, y_center):
        # If there are previous ball coordinates to compare with...
        if self.prev_y_center is not None:
            # Compute the change in the ball's y-coordinate.
            delta_y = y_center - self.prev_y_center

            # If the ball's motion indicates a dribble...
            if (
                self.prev_delta_y is not None
                and delta_y < 0
                and self.prev_delta_y > self.dribble_threshold
            ):
                # Increment the dribble count.
                self.dribble_count += 1

            # Store the current delta_y for the next frame's calculations.
            self.prev_delta_y = delta_y

    # Check if a double dribble has occurred.
    def check_double_dribble(self):
        # If the player was holding the ball and has started another dribble...
        if self.was_holding and self.dribble_count > 0:
            # Record the current time as the time of the double dribble.
            self.double_dribble_time = time.time()

            # Reset the state variables.
            self.was_holding = False
            self.dribble_count = 0

            print("Double dribble!")


# Create a DoubleDribbleDetector instance and start it.
if __name__ == "__main__":
    detector = DoubleDribbleDetector()
    detector.run()
