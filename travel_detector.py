"""
    Author: Ayush Pai
    Date Last Edited: Oct 29, 2022

    Travel Detection of a Basketball Player
    https://youtu.be/3UeoKxw8UYs
"""

import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


def main():
    lower_bound = np.array([0, 255, 0])  # Change these HSV values to your own color
    upper_bound = np.array([35, 255, 255])  # Change these HSV values to your own color

    video_feed = cv2.VideoCapture(0)  # initialize camera

    lows = []
    dribbles = 0

    # initialize firebase parameters to create get realtime step count data
    cred = credentials.Certificate(r"C:\Users\ayush\PycharmProjects\BasketballRef\service.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    doc_ref = db.collection(u'live').document(u'tracker')

    step_list = []
    dribble_list = []
    steps_since_last_dribble = 0

    travel_watcher = False

    dribble_list.append(0)
    dribble_list.append(0)

    while True:
        frame = video_feed.read()[1]  # initialize video feed
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # initialize color processor in HSV
        kernel = np.ones((10, 10), np.uint8)  # create a 10x10 pixel image kernel
        mask = cv2.inRange(hsv, lower_bound, upper_bound)  # create a openCV mask with the defined bounds and hsv
        post_processed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # post processed mask (using kernel)
        mask_overlay = cv2.bitwise_and(frame, frame, mask=post_processed_mask)  # overlay the mask onto live image

        """The post processed mask will be used for scraping the coordinates. The mask is a 2D array with each array
        containing a row of pixels on the mask. By using np.where(), we can find the coordinate at which the pixel
        is white with the value 255. Since the post processed mask is a 2D array, np.where() will return 2 arrays.
        The first array is the row location (which we will use as Y-Coordinates) and the second array is column
        location (which we will use as X-coordinates)."""

        x_coordinates = np.where(post_processed_mask == 255)[1]  # array of x-coordinates of the ball
        y_coordinates = np.where(post_processed_mask == 255)[0]  # array of y-coordinates of the ball

        if len(np.where(post_processed_mask == 255)[0]) > 5:  # if the mask detects something
            bottom_y = np.max(y_coordinates)  # get the bottom y value (max value of array)
            top_y = np.min(y_coordinates)  # get the top y value (min value of array)
            left_x = np.min(x_coordinates)  # get the left x value (min value of array)
            right_x = np.max(x_coordinates)  # get the right x value (max value of array)

            # calculate the average radius of the ball (by averaging the x and y radius)
            avg_radius = (((right_x - left_x) / 2) + ((bottom_y - top_y) / 2)) / 2

            # calculate the center x and y positions of the ball
            center_x = (left_x + right_x) / 2
            center_y = (top_y + bottom_y) / 2

            # create a circle around the ball on the video feed
            cv2.circle(frame, (int(center_x), int(center_y)), int(avg_radius), (0, 255, 0), thickness=2, lineType=8,
                       shift=0)

            # implement linear regression equation y = |2.31967x−23.8525| to account for the z-value of the ball
            z_threshold = abs((2.31967 * avg_radius) - 23.8525)  # note: you may need to adjust this formula
            # implement linear regression equation y = |345.902−2.21311x| to account for stray min y values
            min_y_threshold = abs(345.902 - (2.21311 * avg_radius))
            # add the lowest y value to the list of low points
            lows.append(bottom_y)

            # if the current lowest point is greater than the old lowest point by the z amount AND
            # if the current lowest point is greater than the minimum Y threshold value
            if lows[len(lows) - 1] > lows[len(lows) - 2] + z_threshold and lows[len(lows) - 1] > min_y_threshold:
                lows.clear()
                dribbles = dribbles + 1  # account for a new dribble!
                print(dribbles)

        # add the current number of steps to the array of steps
        steps = doc_ref.get(field_paths={'steps'}).to_dict().get('steps')
        step_list.append(steps)
        # add the current number of dribbles to the array of dribbles
        dribble_list.append(dribbles)

        # Check if there is an difference between steps of last 2 cycles to turn the travel watcher on
        if (step_list[len(step_list) - 1] > step_list[len(step_list) - 2]
                and dribble_list[len(dribble_list) - 1] == dribble_list[len(dribble_list) - 2]
                and not travel_watcher):
            travel_watcher = True
        else:
            travel_watcher = False

        # If travel watcher is on and the current step count is greater than previous step count and if the dribble
        # count has not changed, then the steps since the last dribble is incremented by 1
        if (travel_watcher and step_list[len(step_list) - 1] > step_list[len(step_list) - 2] + 1
                and dribble_list[len(dribble_list) - 1] == dribble_list[len(dribble_list) - 2]):
            steps_since_last_dribble = steps_since_last_dribble + 1

        # If travel watcher is on and the current step count is greater than previous step count by 2and if the dribble
        # count has not changed, then the steps since the last dribble is 2
        elif (travel_watcher and step_list[len(step_list) - 1] > step_list[len(step_list) - 2] + 2
              and dribble_list[len(dribble_list) - 1] == dribble_list[len(dribble_list) - 2]):
            steps_since_last_dribble = steps_since_last_dribble + 2 

        # print("Steps Since Last Dribble: ", steps_since_last_dribble, ", Steps: ", steps, ", Dribbles: ", dribbles,
        # ", Travel Watcher: ", travel_watcher)

        if steps_since_last_dribble > 2:  # Basic basketball rule (doesn't account for gather step sorry James Harden)
            print("Travel Detected")

        # display the various feeds
        cv2.imshow('Live Image', frame)
        cv2.imshow('Mask Overlay', mask_overlay)
        cv2.imshow('Mask', mask)
        cv2.imshow('Post Processed Kernel Mask', post_processed_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # exit
    video_feed.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
