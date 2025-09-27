# Intelligent Video Analytics : Real-Time Person Detection and Tracking
Real-time multi-person tracking system combining YOLOv8 for detection and DeepSORT for identity-preserving tracking. Includes a Streamlit interface for video upload, webcam input, ID selection, and processed video preview.This project is an interactive web application for real-time person detection and tracking in videos, built with Streamlit, YOLOv8, and DeepSORT.

This project is an interactive web application for real-time person detection and tracking in videos, built with **Streamlit**, **YOLOv8**, and **DeepSORT**.

The application allows users to:
- Upload a video file or use a live webcam stream  
- Detect and track individuals throughout the video  
- Assign unique, persistent IDs to each person using YOLOv8 (object detection) + DeepSORT (multi-object tracking)  
- Follow individuals consistently across frames, even in crowded environments  

---

## 🚀 Workflow

The app provides an intuitive workflow:

1. **Scan Mode** – Analyze the first few frames of a video to automatically detect and collect all person IDs.  
2. **Selection Mode** – Choose to track all detected individuals or focus on a single person by selecting their ID.  
3. **Tracking Mode** – Process the entire video, overlay bounding boxes and IDs, and save the annotated output.  

## 💻 Hardware & Model Compatibility

| Platform              | Recommended Model            | GPU Support | Notes                                                                 |
|-----------------------|------------------------------|-------------|----------------------------------------------------------------------|
| **CPU-only PC/Laptop** | `yolov8n.pt`                 | ❌          | Works fine (~10–15 FPS, lightweight model optimized for CPUs).        |
| **Desktop w/ GPU**     | `yolov8s.pt` or `yolov8m.pt` | ✅ CUDA     | Best performance, smoother detection, and higher accuracy.            |
| **Jetson Nano**        | `yolov8n.pt`                 | ✅ CUDA     | Achieves near real-time performance with lightweight models.          |
| **Raspberry Pi**       | `yolov8n` (exported to NCNN) | ❌ CPU only | Limited FPS, suitable for demos but not heavy real-time applications. |

⚡ **Default model in this project:** `yolov8n.pt` (fastest & most efficient for general use).  
📌 *Performance depends on hardware specs, model size, and video resolution.*

---

## 🎨 Features

- **Video & Webcam Input**  
    Upload video files (`.mp4`, `.avi`, `.mov`) or use a live webcam stream for real-time tracking.  

- **Human Detection (YOLOv8)**  
    Powered by Ultralytics YOLOv8 for fast and accurate person detection in each frame.  

- **Multi-Object Tracking (DeepSORT)**  
    Assigns unique IDs to individuals and tracks them consistently across frames.  

- **Person ID Scanning**  
   Automatically scans the first few frames to collect all detected person IDs for selection.  

- **Flexible Tracking Modes**  
    Choose to track all detected persons or focus on a specific individual by ID.  

- **Live Annotated Preview**  
    Displays bounding boxes and person IDs directly on frames during both scanning and tracking.  

- **Output Video Export**  
    Saves the processed and annotated video in `.mp4` format for later review.  

- **Custom Streamlit UI**  
    Modern dark-themed interface with styled buttons, alerts, and sidebar for an improved user experience.  

- **Progress Feedback**  
   Real-time progress bar and live frame updates while the video is being processed.  

- **Session Management**  
    Reset or clear the current session state with a single click to start fresh with new inputs.  
  

---

## 🔧 Use Cases

This tool is useful for:

- 🎥 **Video analytics** – Tracking individuals in recorded surveillance or CCTV footage  
- 🧪 **Research and experiments** – Computer vision projects requiring person re-identification  
- 🎓 **Education** – Demonstrating the practical integration of deep learning models in real-world tracking tasks  
- ⚙️ **Prototyping** – Building blocks for larger intelligent video analysis systems  

---

## ⚡ Model Information

- By default, the application uses **`yolov8n.pt`**, a lightweight model optimized for **CPU-based systems**, ensuring broad compatibility.  
- On **GPU-enabled platforms**, larger models (`yolov8s`, `yolov8m`, etc.) can be used for **higher accuracy**.  


