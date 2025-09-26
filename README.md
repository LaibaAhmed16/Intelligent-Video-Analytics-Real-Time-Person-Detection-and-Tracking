# Intelligent Video Analytics : Real-Time Person Detection and Tracking
Real-time multi-person tracking system combining YOLOv8 for detection and DeepSORT for identity-preserving tracking. Includes a Streamlit interface for video upload, webcam input, ID selection, and processed video preview.

# #✨ Features

🎥 Video Upload & Webcam Support – Upload .mp4, .avi, .mov files or run live with your webcam.

🕵️ Person Detection with YOLOv8 – Detects humans in video frames with real-time performance.

🧭 Multi-Object Tracking (DeepSORT) – Assigns unique IDs to each person and tracks them across frames.

🔎 Scan Mode for IDs – Analyze initial frames to collect all detected person IDs before tracking.

🎯 Track All or Select Specific Person – Option to track everyone in the scene or focus on one chosen ID.

🖼️ Annotated Previews – See bounding boxes and IDs overlaid on frames during scanning and tracking.

💾 Save Processed Video – Automatically saves output with tracked persons to .mp4 format.

🎨 Custom Streamlit UI Styling – Modern dark theme with styled buttons, alerts, and sidebar.

📊 Progress Indicator – Real-time frame-by-frame processing with a progress bar.

🔄 Reset / Clear Session – One-click reset to restart with a new video or webcam input.
