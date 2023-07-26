# AI Basketball Referee v2.0

## Demo Video
[![Youtube Video Demo](https://img.youtube.com/vi/VZgXUBi_wkM/0.jpg)](https://www.youtube.com/watch?v=VZgXUBi_wkM)


## Purpose
Sports are evolving day by day and the technology that supports these sports are evolving at an exponential rate. Many sports have implemented computer vision in order to improve referee calls and the overall fairness of the game. Tennis uses cameras to detect if a ball is out, Track and Field use cameras to detect who won a race, and many more. One sport however that has failed to do so on a significant scale is basketball. On top of that, basketball is one of the sports that is notorious for championship-changing, egregious referee calls. Implementing computer vision to watch over basketball games can not only make the game a much more fair experience for players and fans but also be a way of collecting data to use for greater machine learning models and statistics.

## How it works
The AI Basketball Referee is a computer vision-based system that uses a custom YOLO (You Only Look Once) machine learning model trained on 3000 annotated images to detect basketballs in real-time. Additionally, it utilizes YOLO pose estimation to detect keypoints on the body of the players. By combining these two techniques, the AI Basketball Referee is capable of accurately identifying travels and double dribbles in basketball games.

## Basketball Detection
The first step in the AI Basketball Referee's process is basketball detection. The YOLO machine learning model is trained to recognize basketballs within the video frames. It has been trained on a diverse dataset of 3000 annotated images containing various basketball poses, lighting conditions, and backgrounds. During runtime, the model analyzes each frame in real-time and predicts bounding boxes around the detected basketballs.

## Pose Estimation
To enable the detection of travels and double dribbles, the AI Basketball Referee also employs YOLO pose estimation. This technique allows the system to identify and track keypoints on the body of the players. Key body joints such as the ankles, knees, hips, elbows, and wrists are crucial for determining player movements accurately.

## Travel Detection
Once the basketballs and player keypoints are detected, the AI Basketball Referee applies a set of predefined rules to determine if a travel violation has occurred. By analyzing the position and movement of the player's keypoints over consecutive frames, the system can detect instances where a player has taken steps without dribbling the ball or has moved more than the allowed distance without dribbling or passing.

## Double Dribble Detection
Similarly, the AI Basketball Referee leverages the detected basketballs and player keypoints to identify double dribbles. By tracking the position and movement of the player's keypoints and analyzing the interactions with the basketball, the system can detect situations where a player dribbles the ball, stops, and then starts dribbling again without another player touching or possessing the ball in the meantime.

## Real-time Feedback
The AI Basketball Referee provides real-time feedback on travel and double dribble violations during basketball games. It highlights the detected violations on the video feed, making it easy for referees or users to identify and assess the accuracy of the system's decisions. Additionally, the system can generate logs or alerts to record detected violations for further analysis or review.

## Customizability and Expansion
The AI Basketball Referee has been designed to be customizable and expandable. Users can fine-tune the system's parameters, such as the detection threshold for basketballs and the sensitivity of travel and double dribble detection, to suit their specific requirements. Furthermore, additional rules and detection capabilities can be incorporated into the system to address other basketball violations or game situations.

Overall, the AI Basketball Referee combines state-of-the-art computer vision techniques, including YOLO object detection and pose estimation, to accurately detect travels and double dribbles in real-time basketball games. It provides a valuable tool for referees, coaches, and players to analyze gameplay, improve player performance, and enhance the overall fairness of basketball matches.

## Setup
1. Clone project
2. Open project in VSCode
3. Create a new conda environment: `conda create -n exercise-tracking python=3.11`
4. Activate conda environment: `conda activate exercise-tracking`
5. Install ultralytics package: `pip install ultralytics`
6. Run any of the Python scripts you would like to try out. `double_dribble.py` and `travel_detection.py` are the ones that provide realtime referee calls.
7. Change the input of the video to either your webcam (`cv2.VideoCapture(0)`) or a video file with the relative path (`cv2.VideoCapture('video.mp4')`).

## basketballModel.pt
This file is the core to the basketball detection model. Unfortunately, the file is too big and has exceeded GitHub storage limits. Please download the file here:
https://drive.google.com/file/d/1e6HLRuhh1IEmxOFaxHQMxfRqhzD92t3B/view?usp=sharing

## In The News:
- https://news.gatech.edu/news/2023/07/25/tech-student-brings-artificial-intelligence-basketball-officiating
- https://www.hackster.io/news/ai-basketball-referee-detects-traveling-ed1ed45f8ccd
- https://aifinityhub.com/2023/06/03/hoops-and-algorithms-ais-role-in-nbas-refereeing/
- https://www.fry-ai.com/p/ai-basketball-referee-days-yelling-human-officials-soon
- SingleStore Webinar https://www.singlestore.com/resources/webinar-how-to-build-an-openai-basketball-referee-system-with-computer-vision-2023-07/
- Overtime (7M+) https://www.instagram.com/reel/CtMd6LgAAMo/?igshid=ZmZiYTY5ZDNhOA%3D%3D
- Barsee AI https://twitter.com/WGMImedia/status/1664205786644455424
