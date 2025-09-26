# Intelligent Video Analytics : Real-Time Person Detection and Tracking
Real-time multi-person tracking system combining YOLOv8 for detection and DeepSORT for identity-preserving tracking. Includes a Streamlit interface for video upload, webcam input, ID selection, and processed video preview.

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
