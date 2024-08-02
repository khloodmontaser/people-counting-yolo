# People Counting in a Video

This script is designed to count the number of people in a given video using the YOLOv5 object detection model and OpenCV. It focuses on detecting people within a designated region of interest (ROI).

## Used Libraries

OpenCV
NumPy
PyTorch
## Code Explanation

Inputs Section
The input section includes the video source and the loading of the YOLOv5 model from the PyTorch hub.

Logic Section
This section contains the main logic for:

Drawing the region of interest (ROI).
Detecting people using YOLOv5.
Checking if detected people are within the ROI.
Output Section
The output section displays the number of people detected within the ROI on the video frame.






