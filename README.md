# Computer Vison-Based Basketball Referee

## Purpose
Sports are evolving day by day and the technology that supports these sports are evolving at an exponential rate. Many sports have implemented computer vision in order to improve referee calls and the overall fairness of the game. Tennis uses cameras to detect if a ball is out, Track and Field use cameras to detect who won a race, and many more. One sport however that has failed to do so on a significant scale is basketball. On top of that, basketball is one of the sports that is notorious for championship-changing, egregious referee calls. Implementing computer vision to watch over basketball games can not only make the game a much more fair experience for players and fans but also be a way of collecting data to use for greater machine learning models and statistics.

**Demo Video:** https://youtu.be/3UeoKxw8UYs

## An Evolving Project
This project is the first step towards that goal. I developed a computer vision-based basketball referee to detect if a player travels in a game of basketball. Using OpenCV libraries, I was able to track a ball using color masking, pixel arrays, and post-processing kernels to create a real-time ball tracker. Utilizing the X/Y coordinates, I was able to figure out the radius and center point of the ball. Then using an android app as a pedometer, I was able to detect the real-time steps of a player. Comparing the real-time dribbles with real-time steps, the program can detect travels.

## Future of This Project
There are many future applications for this project. Currently, the program detects travel, one aspect of basketball violations. Other similar easily implementable applications include double dribble detection, carry detection, stepping out of bounds, and accounting for gather steps. More advanced applications include technical-foul detection and time violations.

## Plan to Advance this Project
After conversing with NBA team 76ers President, Daryl Morey, a commercial implementation of computer vision basketball referees by the NBA or NCAA will need some major improvements, which I am actively working on. Currently, the detection of the basketball is done with a color mask set with pre-defined HSV values based on experimental data. This will not work in all environments due to variations in lighting, meaning that a different method of detecting the ball is required such as training a custom machine learning model and using pose estimations and object detection. Also, using a pedometer to detect steps is impractical with 10 players on a court. A solution is using computer vision to detect jersey numbers and player faces to understand who has the ball and train a machine learning model to detect their steps. On this scale, we would need to stream many camera angles, which are conveniently already available at the NBA and NCAA games.

## News References:
https://www.hackster.io/news/ai-basketball-referee-detects-traveling-ed1ed45f8ccd
