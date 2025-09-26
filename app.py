"""
YOLOv8 + DeepSORT Person Tracking Web App
-----------------------------------------

This Streamlit application allows interactive person detection and tracking 
in video files or live webcam streams. It integrates YOLOv8 for object detection 
and DeepSORT for multi-object tracking.

Main Features:
    • Upload a video file or use a live webcam stream.
    • Automatically scan the first N frames to collect unique person IDs.
    • Track either all detected persons or one specific person chosen by ID.
    • Save the processed video with bounding boxes and preview results in-app.
    • Modern user interface with custom Streamlit CSS.

Dependencies:
    - streamlit
    - opencv-python
    - torch
    - ultralytics (YOLOv8)
    - deep_sort_realtime
    - numpy

Usage:
    streamlit run app.py
"""

# ---------------- IMPORT LIBRARIES ----------------
# Import necessary libraries for web interface, computer vision, deep learning,
# file handling, and person tracking.

import streamlit as st
import cv2
import tempfile
import torch
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------- CONFIG ----------------
# These constants define thresholds and parameters for detection and scanning.
CONF_THRESH = 0.5     # Confidence threshold for YOLO person detections
PERSON_CLASS_ID = 0  # YOLO class ID for 'person'
SCAN_FRAMES = 30     # how many frames to scan initially to collect IDs

# ---------------- CUSTOM STYLES ----------------
# Inject custom CSS to enhance the appearance of the Streamlit app.
# The styles define background gradient, text colors, button design, 
# widget appearance, and video frame styling.

st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    /* Titles */
    h1, h2, h3 {
        color: #fbc531 !important;
        font-weight: 700 !important;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: #1e272e;
    }

    /* Buttons */
    .stButton>button {
        background-color: #fbc531;
        color: #0b0b0b;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        padding: 0.6em 1.2em;
        transition: all 0.22s ease;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    }
    .stButton>button:hover {
        background-color: #e1b12c;
        color: #0b0b0b;
        transform: translateY(-2px);
    }

    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 10px;
        padding: 1em;
        font-weight: 500;
    }

    /* Video frames */
    img {
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }

    /* ============================
       FORCE SIDEBAR & WIDGET TEXT
       ============================ */

    /* File uploader label (outside box, keep gold) */
section[data-testid="stFileUploader"] label {
    color: #fbc531 !important;
    font-weight: 600 !important;
}

/* File uploader inner box text (inside white box, make black) */
section[data-testid="stFileUploader"] div[role="button"] * {
    color: black !important;
}

    /* Common widget labels in main area */
    .stCheckbox label, .stRadio label, .stSelectbox label, .stFileUploader label,
    .stCheckbox div, .stRadio div, .stSelectbox div {
        color: #fbc531 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    /* Dropdown text and options */
    select, option, .stSelectbox [data-baseweb="select"] * {
        color: black !important;
        opacity: 1 !important;
    }
            
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
# Configure the Streamlit page with a title, favicon, and layout.
st.set_page_config(page_title="YOLOv8 + DeepSORT Tracker", page_icon="🎥", layout="wide")
st.title("🎥 Interactive Person Tracking (YOLOv8 + DeepSORT)")

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load the YOLOv8 model for person detection.
    
    The function checks if a GPU is available and loads the 
    lightweight 'yolov8n.pt' model. Streamlit caching ensures 
    that the model is only loaded once per session.
    
    Returns:
        YOLO: Pretrained YOLOv8 model object.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")  # model will be cached by Streamlit
    return model

# Load YOLO model once and reuse throughout the app.
model = load_models()

# ---------------- Session state defaults ----------------
# Initialize default session state variables to track user input, app status,
# and outputs. This prevents errors when keys are missing.

if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'use_webcam' not in st.session_state:
    st.session_state.use_webcam = False
if 'scan_done' not in st.session_state:
    st.session_state.scan_done = False
if 'id_choices' not in st.session_state:
    st.session_state.id_choices = []
if 'selected_person_id' not in st.session_state:
    st.session_state.selected_person_id = None
if 'start_tracking' not in st.session_state:
    st.session_state.start_tracking = False
if 'output_path' not in st.session_state:
    st.session_state.output_path = None

# ---------------- Helpers ----------------
def detect_detections(frame, conf_thresh=CONF_THRESH):
    """
    Run YOLOv8 on a single video frame to detect persons.

    The results are converted into a format compatible with DeepSORT,
    which requires bounding boxes in (x, y, w, h) format along with
    confidence scores and class IDs.

    Args:
        frame (ndarray): Input video frame in BGR format.
        conf_thresh (float): Minimum detection confidence threshold.

    Returns:
        tuple: 
            detections (list): DeepSORT-style detections 
                               [[x, y, w, h], confidence, class_id].
            results (YOLO object): Full YOLO detection results for debugging/inspection.

    """
    results = model(frame, conf=conf_thresh, verbose=False)[0]
    detections = []
    for box in results.boxes:
        # Support multiple shapes of attributes robustly
        try:
            cls = int(box.cls[0])
        except Exception:
            cls = int(box.cls)
        if cls != PERSON_CLASS_ID:
            continue
        try:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
        except Exception:
            x1, y1, x2, y2 = map(float, box.xyxy.tolist())
        try:
            conf = float(box.conf[0])
        except Exception:
            conf = float(box.conf)
        if conf < conf_thresh:
            continue
        detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])
    return detections, results

def annotate_frame(frame, tracks, show_all=True, selected_id=None):
    """
    Draw bounding boxes and ID labels on the given frame.

    Args:
        frame (ndarray): Frame to annotate (BGR format).
        tracks (list): Active DeepSORT track objects.
        show_all (bool): If True, annotate all tracks. If False, only show selected ID.
        selected_id (int | None): Specific person ID to highlight.

    Returns:
        ndarray: Frame with bounding boxes and labels drawn.
    """
    out = frame.copy()
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        if (not show_all) and (selected_id is not None) and (tid != selected_id):
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f"Person | ID {tid}", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out

def scan_video_for_ids(video_path_or_index, use_webcam=False, n_frames=SCAN_FRAMES):
    """
    Scan the first N frames of a video or webcam stream to collect 
    unique person IDs using DeepSORT.

    This step is used to let the user choose which ID to track later.

    Args:
        video_path_or_index (str | int): Path to video file or webcam index.
        use_webcam (bool): If True, capture frames from webcam.
        n_frames (int): Number of frames to scan for IDs.

    Returns:
        tuple: 
            id_choices (list[int]): List of detected unique person IDs.
            annotated_preview (ndarray): Example annotated frame with IDs.
    """
    id_choices = []
    annotated_preview = None
    scan_tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)

    cap = cv2.VideoCapture(0 if use_webcam else video_path_or_index)

    frames_read = 0
    try:
        while frames_read < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1
            detections, _ = detect_detections(frame)
            tracks = scan_tracker.update_tracks(detections, frame=frame)
            # collect ids
            for tr in tracks:
                if tr.is_confirmed() and tr.track_id not in id_choices:
                    id_choices.append(tr.track_id)
            # keep the last annotated frame for preview
            annotated_preview = annotate_frame(frame, tracks, show_all=True)
    finally:
        cap.release()

    return id_choices, annotated_preview

# ---------------- UI: Upload / Webcam ----------------
# This section handles video input from either:
#   • File uploader (user selects a local video file)
#   • Live webcam stream (real-time detection & tracking)


# The sidebar is used for input controls to keep the main app area clean.
st.sidebar.header("Input")

# File uploader widget to allow uploading video files
video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
# Checkbox to switch to webcam mode
use_webcam_checkbox = st.sidebar.checkbox("Use Webcam (Live Mode)", value=False)

# Reset if user changes input mode or uploads a new file
if use_webcam_checkbox != st.session_state.use_webcam:
    st.session_state.use_webcam = use_webcam_checkbox
    st.session_state.video_path = None
    st.session_state.scan_done = False
    st.session_state.id_choices = []
    st.session_state.selected_person_id = None
    st.session_state.start_tracking = False

if video_file is not None:
    # Save uploaded file to a temporary location for OpenCV to read
    if st.session_state.video_path is None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
        tfile.write(video_file.read())
        tfile.flush()
        tfile.close()
        st.session_state.video_path = tfile.name
        # Reset scan/tracking state on new upload
        st.session_state.scan_done = False
        st.session_state.id_choices = []
        st.session_state.selected_person_id = None
        st.session_state.start_tracking = False

# Provide explanation
# step-by-step guide for users so they understand the workflow.
st.markdown(
    "### Workflow\n"
    "1. Scan the first few frames to detect persons and collect their tracker IDs.\n"
    "2. Choose whether to track **all persons** or **a specific person** (select via dropdown).\n"
    "3. Press **Start tracking** to process the whole video and save the output."
)

# ---------------- Scan step ----------------
# User clicks a button to scan the first N frames of the video or webcam
# This collects unique DeepSORT track IDs and shows an annotated preview.
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("🔎 Scan for person IDs"):
# Reset states before scanning
        st.session_state.scan_done = False
        st.session_state.id_choices = []
        st.session_state.selected_person_id = None
        st.session_state.start_tracking = False

        # Do scan
        try:
            if st.session_state.use_webcam:
                ids, preview = scan_video_for_ids(0, use_webcam=True, n_frames=SCAN_FRAMES)
            else:
                if st.session_state.video_path is None:
                    st.warning("Please upload a video first (or enable Webcam).")
                    ids, preview = [], None
                else:
                    ids, preview = scan_video_for_ids(st.session_state.video_path, use_webcam=False, n_frames=SCAN_FRAMES)
            # Save results in session state
            st.session_state.id_choices = ids
            st.session_state.scan_done = True
            st.success(f"Scan finished — found IDs: {ids if ids else 'None'}")
            if preview is not None:
                # use st.session_state to hold preview image (converted to RGB) for display
                st.session_state._preview_img = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.session_state.scan_done = False
            st.error(f"Error during scanning: {e}")

with col2:
    # Show detected IDs and preview image after scanning
    if st.session_state.scan_done:
        st.write("Detected person IDs (from scan):")
        if st.session_state.id_choices:
            st.write(st.session_state.id_choices)
            # show preview image if available
            if '_preview_img' in st.session_state and st.session_state._preview_img is not None:
                st.image(st.session_state._preview_img, use_column_width=True)
        else:
            st.info("No persons detected during the scan. Try scanning more frames or a different video.")

# ---------------- Selection: Track all vs specific ----------------
# After scanning, the user chooses between:
#   • Tracking all persons
#   • Tracking one specific person (by ID from scan)
if st.session_state.scan_done:
    st.markdown("---")
    track_mode = st.radio("Choose tracking mode:", ("Track all persons", "Track specific person (use dropdown)"))
    if track_mode == "Track specific person (use dropdown)":
        if not st.session_state.id_choices:
            st.warning("No IDs available to select. Please re-scan the video.")
        else:
            # show selectbox with the detected IDs
            # Keep previously selected ID if it still exists
            default_idx = 0
            try:
                if st.session_state.selected_person_id in st.session_state.id_choices:
                    default_idx = st.session_state.id_choices.index(st.session_state.selected_person_id)
            except Exception:
                default_idx = 0
            selected = st.selectbox("Select person ID to track:", options=st.session_state.id_choices, index=default_idx)
            st.session_state.selected_person_id = selected
    else:
        # Track all
        st.session_state.selected_person_id = None

    # Start tracking button
    if st.button("▶️ Start tracking"):
        st.session_state.start_tracking = True
        st.session_state.output_path = os.path.join(tempfile.gettempdir(), "output_tracked.mp4")
        # ensure previous output is removed to avoid stale file
        try:
            if os.path.exists(st.session_state.output_path):
                os.remove(st.session_state.output_path)
        except Exception:
            pass

# ---------------- Main tracking execution ----------------
# This section runs YOLO + DeepSORT frame by frame across the entire video
# or continuously for webcam input. Each processed frame is displayed in the app
# and simultaneously written to an output video file.

if st.session_state.start_tracking:
    st.info("Processing and tracking — this will run until the video ends. (This may take a while depending on video/model/hardware.)")
    stframe = st.empty()  # Placeholder for video frames
    progress_bar = st.progress(0)  # Progress bar for feedback

    # Prepare capture
    cap = cv2.VideoCapture(0 if st.session_state.use_webcam else st.session_state.video_path)
    ret, sample_frame = cap.read()
    if not ret:
        st.error("Could not read from the selected input. Make sure the video/webcam is available.")
        cap.release()
        st.session_state.start_tracking = False
    else:
        # Extract video properties
        height, width = sample_frame.shape[:2]
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        # Rewind video file to start if not webcam
        if not st.session_state.use_webcam:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(st.session_state.output_path, fourcc, fps, (width, height))

        # Create fresh tracker for the real tracking run
        tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)
        # For progress estimation (not available for webcam)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not st.session_state.use_webcam else None
        processed = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed += 1
                # Run detection + tracking
                detections, _ = detect_detections(frame)
                tracks = tracker.update_tracks(detections, frame=frame)

                # draw according to selection
                if st.session_state.selected_person_id is None:
                    # track all
                    annotated = annotate_frame(frame, tracks, show_all=True, selected_id=None)
                else:
                    annotated = annotate_frame(frame, tracks, show_all=False, selected_id=st.session_state.selected_person_id)

                # write and show
                out_writer.write(annotated)
                stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

                # update progress
                if total_frames:
                    progress = min(1.0, processed / total_frames)
                    progress_bar.progress(progress)
                else:
                    # for webcam, show an indeterminate progress by incrementing a small value
                    progress_bar.progress(min(1.0, (processed % 100) / 100.0))

        except Exception as e:
            st.error(f"Error during tracking: {e}")
        finally:
            cap.release()
            out_writer.release()
            st.session_state.start_tracking = False
            progress_bar.empty()
            # Show success message and playback
            if os.path.exists(st.session_state.output_path):
                st.success(f"✅ Tracking finished. Output saved at: {st.session_state.output_path}")
                st.video(st.session_state.output_path)
            else:
                st.error("Tracking finished but output file was not created.")

# ---------------- Small utilities ----------------
# Utility button to clear session state and restart the app from scratch.
st.markdown("---")
if st.button("🔄 Reset / Clear state"):
    # clear session_state keys
    keys = ['video_path', 'scan_done', 'id_choices', 'selected_person_id', 'start_tracking', 'output_path', '_preview_img']
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()
