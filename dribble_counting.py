import cv2
from ultralytics import YOLO
import numpy as np


class DribbleCounter:
    def __init__(self):
        # Load the YOLO model for ball detection
        self.model = YOLO("/Users/ayushpai/Desktop/yolov8_custom/best17.pt")

        # Open the webcam
        self.cap = cv2.VideoCapture(1)

        # Initialize variables to store the previous position of the basketball
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None

        # Initialize the dribble counter
        self.dribble_count = 0

        # Threshold for the y-coordinate change to be considered as a dribble
        self.dribble_threshold = 18

    def run(self):
        # Process frames from the webcam until the user quits
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                results_list = self.model(frame, verbose=False, conf=0.65)

                for results in results_list:
                    for bbox in results.boxes.xyxy:
                        x1, y1, x2, y2 = bbox[:4]

                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2

                        print(f"Ball coordinates: (x={x_center:.2f}, y={y_center:.2f})")

                        self.update_dribble_count(x_center, y_center)

                        self.prev_x_center = x_center
                        self.prev_y_center = y_center

                    annotated_frame = results.plot()

                    # Draw the dribble count on the frame

                    cv2.imshow("YOLOv8 Inference", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        # Release the webcam and destroy the windows
        self.cap.release()
        cv2.destroyAllWindows()

    def update_dribble_count(self, x_center, y_center):
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center

            if (
                self.prev_delta_y is not None
                and self.prev_delta_y > self.dribble_threshold
                and delta_y < -self.dribble_threshold
            ):
                self.dribble_count += 1

            self.prev_delta_y = delta_y


if __name__ == "__main__":
    dribble_counter = DribbleCounter()
    dribble_counter.run()
