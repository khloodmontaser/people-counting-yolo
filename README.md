# people-counting-yolo

This script is designed to count the number of people in a given video using the YOLOv5 object detection model and OpenCV. The detection is performed within a designated region of interest (ROI) in the video.

Used Libraries
OpenCV: For video processing and image manipulation.
NumPy: For numerical operations.
PyTorch: For loading and utilizing the YOLOv5 model.
Code Explanation
Inputs Section
The input section includes:

Video Source: The path or source of the input video.
Model Loading: The YOLOv5 model is loaded from the PyTorch hub for object detection.
Logic Section
The logic section contains:

Region of Interest (ROI): Drawing the ROI on the video frame.
Person Detection: Using YOLOv5 to detect people within the frame.
ROI Check: Verifying if the detected people are within the ROI.
Output Section
The output section:

Display: Shows the number of people detected within the ROI on the video frame.
Image
(Include any relevant images or screenshots here)

