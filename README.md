# Real-Time Phone Detection Using YOLO and OpenCV
The objective of this project is to develop a real-time phone detection system using computer vision techniques. By leveraging the YOLO (You Only Look Once) algorithm, this system will identify and highlight phones in a live video feed from a webcam. Additionally, it will provide visual notifications on the screen whenever a phone is detected.

# Project Components
YOLO Algorithm: A state-of-the-art object detection system that uses deep learning to detect objects in images and videos in real-time.

OpenCV Library: An open-source computer vision and machine learning software library that provides tools for image processing, video capture, and manipulation.

Webcam/Camera: A device to capture live video feed, which will be analyzed for phone detection.

# Requirements
Software Requirements:

Python (version 3.x)

OpenCV (install via pip: pip install opencv-python opencv-python-headless)

NumPy (install via pip: pip install numpy)

YOLOv3 weights and configuration files (e.g., yolov3.weights, yolov3.cfg, and coco.names).

# Hardware Requirements:
A computer with a webcam.

Sufficient processing power (GPU recommended for optimal performance).

# Methodology
// Setup:

Install the necessary libraries and download YOLO weights, configuration, and COCO class names.
Ensure that the webcam is functional and accessible through OpenCV.

// Load YOLO Model:

Use OpenCV's DNN module to load the pre-trained YOLO model, including its weights and configuration.

// Capture Video Feed:

Utilize OpenCV to capture video from the webcam.
Continuously read frames from the video feed for processing.

// Image Preprocessing:

Convert the captured frames into a format suitable for YOLO processing (resizing and normalizing the pixel values).

// Object Detection:

Run the YOLO model on the preprocessed image to detect objects.
Extract class IDs, confidence scores, and bounding box coordinates for detected objects.

// Filter Detections:

Identify objects classified as phones using the class ID corresponding to "cell phone".
Apply Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes.

// Display Results:

Draw bounding boxes around detected phones in the video feed.
Overlay labels and confidence scores on the bounding boxes.
Display a notification message when a phone is detected.

// User Interaction:

Provide an option to exit the video feed by pressing the 'q' key.
