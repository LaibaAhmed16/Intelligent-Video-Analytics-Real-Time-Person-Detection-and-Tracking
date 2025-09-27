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

  ## 🤖 Models & Architecture

This project integrates **YOLOv8**, developed by Ultralytics, with the **DeepSORT** tracking algorithm to achieve accurate and efficient person detection and tracking in video streams. By combining these two models, the system is able to both detect humans in each frame and assign consistent IDs to individuals as they move across time.

---

### 🔍 YOLOv8 – Object Detection

**YOLOv8 (You Only Look Once, version 8)** is a state-of-the-art object detection model that performs fast and accurate detection in a single stage. It is pretrained on the **COCO dataset**, which includes the *person* class, making it directly applicable for human detection. In this project, YOLOv8 scans each video frame and generates bounding boxes around detected persons along with confidence scores.

The main advantages of YOLOv8 are its **speed, scalability, and accuracy**. The project defaults to `yolov8n.pt` (Nano model), which is lightweight and optimized for CPU use, achieving around 10–15 FPS on standard laptops. On systems equipped with GPUs, larger variants such as `yolov8s.pt` or `yolov8m.pt` can be used to deliver higher accuracy and smoother real-time performance.

---

### 🧭 DeepSORT – Object Tracking

**DeepSORT (Simple Online and Realtime Tracking with Deep Association Metrics)** is an extension of the SORT tracking algorithm that enhances robustness by incorporating appearance-based embeddings. It combines a **Kalman filter** for motion prediction with a **Hungarian algorithm** for detection-to-track assignment, ensuring that detections from YOLOv8 are consistently linked across frames.

This approach allows each detected person to be assigned a **unique, persistent ID**, even when partial occlusions or re-entries occur. In the context of this project, DeepSORT provides reliable multi-person tracking and enables features such as selecting a specific ID to focus on during analysis.

---

### 🏗️ Workflow in This Project

The pipeline operates as follows: each incoming frame from a video or webcam is processed by YOLOv8 to detect persons. The resulting bounding boxes and confidence scores are then passed to DeepSORT, which matches them with existing tracked objects. DeepSORT maintains unique IDs for each individual and ensures consistency across frames. Finally, the frame is annotated with bounding boxes and IDs, displayed in real time via Streamlit, and optionally saved to an output video file.

---

### ⚡ Performance Notes

This integration achieves a strong balance between **accuracy and efficiency**. Using `yolov8n.pt` ensures portability and smooth performance on most hardware, including CPU-only laptops and edge devices such as the Jetson Nano. With GPU acceleration enabled, models such as `yolov8s.pt` or `yolov8m.pt` provide higher accuracy while maintaining near real-time speeds. DeepSORT runs efficiently alongside YOLOv8 and introduces minimal overhead, making the system suitable for both lightweight and high-performance environments.



