# Intelligent Video Analytics : Real-Time Person Detection and Tracking
Real-time multi-person tracking system combining YOLOv8 for detection and DeepSORT for identity-preserving tracking. Includes a Streamlit interface for video upload, webcam input, ID selection, and processed video preview.This project is an interactive web application for real-time person detection and tracking in videos, built with Streamlit, YOLOv8, and DeepSORT.
The application allows users to upload a video file or use a live webcam stream to detect and track individuals throughout the video. Using YOLOv8 for object detection and DeepSORT for multi-object tracking, each person in the scene is assigned a unique, persistent ID. This enables the system to follow individuals consistently across frames, even in crowded environments.

The app provides an intuitive workflow:

Scan Mode – Analyze the first few frames of a video to automatically detect and collect all person IDs.

Selection Mode – Choose to track all detected individuals or focus on a single person by selecting their ID.

Tracking Mode – Process the entire video, overlay bounding boxes and IDs, and save the annotated output.

With a modern custom-designed Streamlit interface, the app includes progress indicators, real-time previews, and session controls for a smooth user experience.

This tool is useful for:

🎥 Video analytics – Tracking individuals in recorded surveillance or CCTV footage.

🧪 Research and experiments – Computer vision projects requiring person re-identification.

🎓 Education – Demonstrating the practical integration of deep learning models in real-world tracking tasks.

⚙️ Prototyping – Building blocks for larger intelligent video analysis systems.

By default, the application uses yolov8n.pt, a lightweight model optimized for CPU-based systems, ensuring broad compatibility. On GPU-enabled platforms, larger models (yolov8s, yolov8m, etc.) can be used for higher accuracy.


✨ Features

Video & Webcam Input
Upload video files (.mp4, .avi, .mov) or use a live webcam stream for real-time tracking.

Human Detection (YOLOv8)
Powered by Ultralytics YOLOv8 for fast and accurate person detection in each frame.

Multi-Object Tracking (DeepSORT)
Assigns unique IDs to individuals and tracks them consistently across frames.

Person ID Scanning
Automatically scans the first few frames to collect all detected person IDs for selection.

Flexible Tracking Modes
Choose to track all detected persons or focus on a specific individual by ID.

Live Annotated Preview
Displays bounding boxes and person IDs directly on frames during both scanning and tracking.

Output Video Export
Saves the processed and annotated video in .mp4 format for later review.

Custom Streamlit UI
Modern dark-themed interface with styled buttons, alerts, and sidebar for an improved user experience.

Progress Feedback
Real-time progress bar and live frame updates while the video is being processed.

Session Management
Reset or clear the current session state with a single click to start fresh with new inputs.
