import base64
import hashlib
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock

import cv2
import numpy as np
import serial.tools.list_ports as list_ports
import streamlit as st
import torch
from av import VideoFrame
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

RFID_DIR = Path(__file__).resolve().parent / "RF ID"
if str(RFID_DIR) not in sys.path:
    sys.path.insert(0, str(RFID_DIR))

from rfid_bridge import RFIDMonitor
from video_utils import create_output_folder, draw_detection_labels, ensure_goggles_class, sanitize_name

st.set_page_config(page_title="Safety Detection", layout="wide")
st.title("Safety Detection")
st.markdown(
    """
    <style>
    .alert-card {
        background: rgba(10, 22, 33, 0.96);
        border: 1px solid rgba(82, 150, 255, 0.7);
        border-radius: 18px;
        padding: 0.7rem 0.9rem 0.5rem 0.9rem;
        margin-top: 0.5rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
        max-width: 420px;
    }
    .alert-header {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.9rem;
    }
    .alert-header-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: rgba(255, 103, 103, 0.18);
        color: #ff6b6b;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: 700;
        border: 1px solid rgba(255, 103, 103, 0.4);
    }
    .alert-header-text {
        font-size: 1.3rem;
        font-weight: 700;
        color: #edf6ff;
        letter-spacing: 0.02em;
    }
    .alert-shell {
        background: rgba(13, 25, 38, 0.96);
        border: 1px solid rgba(111, 149, 179, 0.5);
        border-radius: 18px;
        padding: 0.9rem 0.9rem 0.75rem 0.9rem;
        max-width: 430px;
        margin-top: 0.4rem;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }
    .alert-shell .summary-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #d6eaf8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0 0 0.35rem 0;
    }
    .alert-shell .summary-head {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.8rem;
    }
    .alert-shell .summary-head .icon {
        width: 28px;
        height: 28px;
        border-radius: 9px;
        background: rgba(255, 102, 102, 0.14);
        color: #ff6969;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        border: 1px solid rgba(255, 102, 102, 0.42);
    }
    .alert-shell .summary-head .title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #ebf4ff;
    }
    .alert-shell .summary-grid {
        display: grid;
        grid-template-columns: 96px 1fr;
        row-gap: 0.35rem;
        column-gap: 0.85rem;
        font-size: 0.83rem;
        color: #d9ebff;
        margin-bottom: 0.8rem;
    }
    .alert-shell .summary-grid .label {
        color: #a8bfd7;
    }
    .alert-shell .summary-grid .value {
        color: #edf7ff;
        font-weight: 600;
        text-align: left;
    }
    .alert-list {
        max-height: 62vh;
        overflow-y: auto;
        padding-right: 0.3rem;
    }
    .alert-list::-webkit-scrollbar {
        width: 8px;
    }
    .alert-list::-webkit-scrollbar-thumb {
        background: rgba(160, 183, 202, 0.45);
        border-radius: 999px;
    }
    .alert-item-box {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        background: transparent;
        border: 1px solid rgba(162, 187, 214, 0.16);
        border-radius: 12px;
        padding: 0.5rem 0.45rem;
        margin: 0.55rem 0;
    }
    .alert-image-box {
        width: 110px;
        min-width: 110px;
        height: 122px;
        overflow: hidden;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.04);
    }
    .alert-image-box img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover;
        display: block;
        border-radius: 10px;
    }
    .alert-item-copy {
        flex: 1;
        color: #eaf6ff;
        line-height: 1.5;
        font-size: 0.82rem;
    }
    .alert-item-copy .title {
        font-size: 0.95rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.25;
        margin-bottom: 0.2rem;
    }
    .alert-item-copy .small {
        color: #d8edf9;
        font-size: 0.8rem;
    }
    .alert-item-copy .meta {
        color: #b8cfe6;
        font-size: 0.72rem;
        margin-top: 0.35rem;
    }
    .alert-summary {
        font-size: 0.95rem;
        color: #dfeaf5;
        line-height: 1.7;
    }
    .alert-summary strong {
        color: #ffffff;
    }
    .alert-item {
        display: flex;
        align-items: flex-start;
        gap: 0.9rem;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .alert-item:last-child {
        border-bottom: none;
    }
    .alert-image {
        width: 160px;
        min-width: 160px;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .alert-image img {
        width: 100% !important;
        height: auto !important;
        display: block;
        border-radius: 10px;
    }
    .alert-copy {
        flex: 1;
        color: #edf5ff;
        font-size: 0.8rem;
        line-height: 1.45;
    }
    .alert-copy b {
        color: #ffffff;
        font-size: 0.9rem;
    }
    .alert-copy .caption {
        color: #b9d3eb;
        font-size: 0.72rem;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.header("Settings")

MODELS = {
    "PPE Kit": ("best.pt", True),
    "Fire and Smoke": ("best_smoke_fire.pt", False),
}
INFERENCE_BATCH_SIZE = 4
MAX_CAMERA_INDEX = 10
HEAD_REGION_HEIGHT = 0.22
IGNORED_PPE_CLASSES = {"goggles", "no_goggle"}
FIREBASE_DATABASE_URL = os.environ.get(
    "FIREBASE_DATABASE_URL", "https://minniepro-e2b29-default-rtdb.firebaseio.com"
).strip()
FIREBASE_CREDENTIALS = os.environ.get("FIREBASE_CREDENTIALS", "").strip()
if FIREBASE_CREDENTIALS:
    FIREBASE_CREDENTIALS_PATH = Path(FIREBASE_CREDENTIALS)
else:
    _project_dir = Path(__file__).resolve().parent
    _preferred_firebase_credentials = (
        _project_dir / "minniepro-e2b29-firebase-adminsdk-fbsvc-791c9195cb.json"
    )
    _default_firebase_credentials = _project_dir / "firebase-service-account.json"
    _discovered_firebase_credentials = sorted(_project_dir.glob("*-firebase-adminsdk-*.json"))
    if _preferred_firebase_credentials.is_file():
        FIREBASE_CREDENTIALS_PATH = _preferred_firebase_credentials
    elif _default_firebase_credentials.is_file() or len(_discovered_firebase_credentials) != 1:
        FIREBASE_CREDENTIALS_PATH = _default_firebase_credentials
    else:
        FIREBASE_CREDENTIALS_PATH = _discovered_firebase_credentials[0]
class TrackedPerson:
    def __init__(self, track_id: int, box: np.ndarray, hist: np.ndarray | None):
        self.track_id: int = int(track_id)
        self.box: np.ndarray = np.array(box, dtype=np.float32)
        self.hist: np.ndarray | None = hist
        self.last_seen: float = time.time()
        self.first_seen: float = time.time()
        self.hits: int = 1
        self.is_active: bool = True

    def update(self, box: np.ndarray, hist: np.ndarray | None) -> None:
        self.box = 0.65 * np.array(box, dtype=np.float32) + 0.35 * self.box
        if hist is not None:
            if self.hist is not None:
                self.hist = 0.75 * hist + 0.25 * self.hist
            else:
                self.hist = hist
        self.last_seen = time.time()
        self.hits += 1
        self.is_active = True

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, float(self.box[2] - self.box[0])) * max(0.0, float(self.box[3] - self.box[1]))


class StablePersonTracker:
    """Stable multi-person tracker designed to prevent ID jitter and preserve IDs upon re-entry."""

    def __init__(self, memory_seconds: float = 120.0):
        self.tracks: list[TrackedPerson] = []
        self.memory_seconds: float = memory_seconds
        self.lock = Lock()

    def reset(self) -> None:
        with self.lock:
            self.tracks.clear()

    @staticmethod
    def extract_hist(image: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, int(box[0])))
        y1 = max(0, min(h - 1, int(box[1])))
        x2 = max(0, min(w, int(box[2])))
        y2 = max(0, min(h, int(box[3])))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        crop = image[y1:y2, x1:x2]
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist
        except Exception:
            return None

    @staticmethod
    def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def compute_hist_similarity(histA: np.ndarray | None, histB: np.ndarray | None) -> float:
        if histA is None or histB is None:
            return 0.5
        try:
            score = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
            return max(0.0, min(1.0, float(score)))
        except Exception:
            return 0.5

    def update(self, result, image: np.ndarray, names: dict[int, str]) -> None:
        with self.lock:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                now = time.time()
                for t in self.tracks:
                    if now - t.last_seen > 1.5:
                        t.is_active = False
                return

            device = boxes.data.device
            coords = boxes.xyxy.tolist()
            cls_ids = [int(c) for c in boxes.cls.tolist()]
            confs = boxes.conf.tolist()
            labels = [str(names.get(c, "")).strip().lower().replace("-", "_").replace(" ", "_") for c in cls_ids]

            person_indices = [
                i for i, label in enumerate(labels)
                if label in {"person", "worker"}
            ]

            now = time.time()
            if len(self.tracks) > 1:
                self.tracks = [t for t in self.tracks if (now - t.last_seen) < self.memory_seconds]

            det_boxes = [np.array(coords[i], dtype=np.float32) for i in person_indices]
            det_hists = [self.extract_hist(image, b) for b in det_boxes]

            img_h, img_w = image.shape[:2] if image is not None else (720, 1280)
            diag = np.hypot(img_w, img_h)

            matched_det_to_track: dict[int, TrackedPerson] = {}
            matched_track_ids: set[int] = set()

            if det_boxes and self.tracks:
                candidates = []
                for d_idx, (d_box, d_hist) in enumerate(zip(det_boxes, det_hists)):
                    d_center = ((d_box[0] + d_box[2]) / 2.0, (d_box[1] + d_box[3]) / 2.0)
                    d_area = max(0.0, float(d_box[2] - d_box[0])) * max(0.0, float(d_box[3] - d_box[1]))
                    for t_idx, track in enumerate(self.tracks):
                        iou = self.compute_iou(d_box, track.box)
                        dist = np.hypot(d_center[0] - track.center[0], d_center[1] - track.center[1])
                        norm_dist = dist / diag if diag > 0 else 1.0
                        dist_sim = max(0.0, 1.0 - norm_dist * 2.0)
                        area_ratio = (min(d_area, track.area) / max(d_area, track.area)) if max(d_area, track.area) > 0 else 1.0
                        color_sim = self.compute_hist_similarity(d_hist, track.hist)

                        if iou > 0.3:
                            sim = 0.5 * iou + 0.3 * color_sim + 0.2 * area_ratio
                        elif (now - track.last_seen) < 2.5:
                            sim = 0.45 * dist_sim + 0.35 * color_sim + 0.2 * area_ratio
                        else:
                            sim = 0.65 * color_sim + 0.20 * dist_sim + 0.15 * area_ratio

                        if len(det_boxes) == 1 and len(self.tracks) == 1 and color_sim > 0.15:
                            sim = max(sim, 0.90)

                        candidates.append((sim, d_idx, t_idx))

                candidates.sort(key=lambda x: x[0], reverse=True)
                assigned_dets = set()
                assigned_tracks = set()

                for sim, d_idx, t_idx in candidates:
                    if d_idx in assigned_dets or t_idx in assigned_tracks:
                        continue
                    if sim >= 0.35 or (len(det_boxes) == 1 and len(self.tracks) == 1):
                        matched_track = self.tracks[t_idx]
                        matched_track.update(det_boxes[d_idx], det_hists[d_idx])
                        matched_det_to_track[d_idx] = matched_track
                        matched_track_ids.add(matched_track.track_id)
                        assigned_dets.add(d_idx)
                        assigned_tracks.add(t_idx)

            for d_idx, (d_box, d_hist) in enumerate(zip(det_boxes, det_hists)):
                if d_idx not in matched_det_to_track:
                    active_ids = {t.track_id for t in self.tracks if t.is_active or t.track_id in matched_track_ids}
                    new_id = 1
                    while new_id in active_ids:
                        new_id += 1
                    new_track = TrackedPerson(new_id, d_box, d_hist)
                    self.tracks.append(new_track)
                    matched_det_to_track[d_idx] = new_track
                    matched_track_ids.add(new_id)

            for t in self.tracks:
                if t.track_id not in matched_track_ids:
                    if now - t.last_seen > 1.5:
                        t.is_active = False

            new_rows = []
            for i in range(len(coords)):
                x1, y1, x2, y2 = coords[i]
                conf = confs[i]
                cls_id = cls_ids[i]
                if i in person_indices:
                    d_pos = person_indices.index(i)
                    track = matched_det_to_track[d_pos]
                    track_id = float(track.track_id)
                else:
                    track_id = -1.0
                new_rows.append([x1, y1, x2, y2, track_id, conf, float(cls_id)])

            new_tensor = torch.tensor(new_rows, dtype=torch.float32, device=device)
            result.update(boxes=new_tensor)


firebase_app = None
firebase_init_lock = Lock()


@st.cache_resource
def load_model(model_path: str):
    model = YOLO(model_path)
    return model


def load_tracking_model(model_path: str, tracking_key: tuple, names: dict[int, str]):
    tracking_models = st.session_state.setdefault("tracking_models", {})
    if tracking_key not in tracking_models:
        tracking_model = YOLO(model_path)
        tracking_model.model.names = names
        tracking_models[tracking_key] = tracking_model
    return tracking_models[tracking_key]


if "selected_detector" not in st.session_state:
    st.session_state.selected_detector = "PPE Kit"

model_columns = st.sidebar.columns(2)
for column, detector_name in zip(model_columns, MODELS):
    if column.button(
        detector_name,
        type="primary" if st.session_state.selected_detector == detector_name else "secondary",
        use_container_width=True,
    ):
        st.session_state.selected_detector = detector_name

model_path, is_ppe_model = MODELS[st.session_state.selected_detector]
model = load_model(model_path)
model_names = ensure_goggles_class(model.names) if is_ppe_model else model.names
if is_ppe_model:
    model.model.names = model_names
    tracking_classes = [
        class_id for class_id, label in model_names.items()
        if label.lower() not in IGNORED_PPE_CLASSES
    ]
else:
    tracking_classes = None

st.sidebar.caption(f"Selected model: **{st.session_state.selected_detector}**")
# Backend inference threshold; intentionally not exposed as a dashboard control.
conf_threshold = 0.45
camera_alert_delay = st.sidebar.number_input(
    "Camera Alert Delay (seconds)",
    min_value=0,
    max_value=60,
    value=10,
    step=1,
    help="Grace period after camera starts before alert notifications are generated.",
)
source = st.sidebar.radio("Detection option", ("Camera", "Video", "Web"))

requires_rfid = is_ppe_model
if requires_rfid:
    if "rfid_monitor" not in st.session_state:
        st.session_state.rfid_monitor = None

    manual_ppe_camera_running = st.session_state.get("ppe_manual_camera_running", False)
    if source != "Video" and not manual_ppe_camera_running:
        available_ports = sorted({port.device for port in list_ports.comports()})
        non_bluetooth_ports = sorted(
            {
                port.device
                for port in list_ports.comports()
                if "Bluetooth" not in port.description
            }
        )
        preferred_port_index = 0
        if non_bluetooth_ports:
            preferred_port_index = available_ports.index(non_bluetooth_ports[0]) if non_bluetooth_ports[0] in available_ports else 0
        rfid_port = st.sidebar.selectbox(
            "Arduino COM port",
            options=available_ports,
            index=preferred_port_index,
            help="Choose the COM port assigned to the Arduino board.",
        )
        rfid_connect = st.sidebar.button("Connect Arduino RFID")

        if rfid_connect:
            if st.session_state.rfid_monitor is not None:
                st.session_state.rfid_monitor.stop()
            monitor = RFIDMonitor(rfid_port)
            if monitor.start():
                st.session_state.rfid_monitor = monitor
                st.sidebar.success(f"Arduino connected to {rfid_port} at 9600 baud.")
            else:
                st.session_state.rfid_monitor = None
                st.sidebar.error(monitor.status.error)

    rfid_monitor = st.session_state.rfid_monitor
    if source != "Video" and not manual_ppe_camera_running:
        if rfid_monitor is None:
            st.session_state["_rfid_match_seen"] = False
            st.sidebar.info("Arduino connected state: waiting for RFID tag detection.")
        else:
            @st.fragment(run_every=0.5)
            def render_rfid_status():
                rfid_status = rfid_monitor.status
                if rfid_status.error:
                    st.error(rfid_status.error)
                elif rfid_status.matched:
                    st.success("RFID CARD: MATCHED — camera unlock granted.")
                    if not st.session_state.get("_rfid_match_seen", False):
                        st.session_state["_rfid_match_seen"] = True
                        st.rerun()
                else:
                    st.session_state["_rfid_match_seen"] = False
                    st.info("Waiting for RFID CARD: MATCHED from Arduino.")

            with st.sidebar:
                render_rfid_status()

    RFID_CAMERA_ON_SECONDS = 15

    if "rfid_session_started_at" not in st.session_state:
        st.session_state.rfid_session_started_at = None
    if "rfid_session_deadline" not in st.session_state:
        st.session_state.rfid_session_deadline = None

    if source != "Video" and not manual_ppe_camera_running:
        if rfid_monitor is not None and rfid_monitor.status.matched:
            if st.session_state.rfid_session_started_at is None:
                st.session_state.rfid_session_started_at = time.monotonic()
                st.session_state.rfid_session_deadline = time.monotonic() + RFID_CAMERA_ON_SECONDS
            session_started_at = st.session_state.rfid_session_started_at
            if session_started_at is not None and st.session_state.rfid_session_deadline is not None:
                if time.monotonic() >= st.session_state.rfid_session_deadline:
                    st.session_state.rfid_session_started_at = None
                    st.session_state.rfid_session_deadline = None
                    rfid_monitor.status.matched = False
                    st.sidebar.info("Camera closed after 15 seconds. Waiting for the next RFID tag.")
                    st.rerun()
        else:
            st.session_state.rfid_session_started_at = None
            st.session_state.rfid_session_deadline = None

    rfid_ready = rfid_monitor is not None and rfid_monitor.status.matched
else:
    rfid_monitor = st.session_state.get("rfid_monitor")
    if rfid_monitor is not None:
        rfid_monitor.stop()
    st.session_state.rfid_monitor = None
    st.session_state["_rfid_match_seen"] = False
    st.session_state.rfid_session_started_at = None
    st.session_state.rfid_session_deadline = None
    rfid_monitor = None
    rfid_ready = True

manual_ppe_camera_running = bool(st.session_state.get("ppe_manual_camera_running", False))


def get_ppe_display_label(label: str) -> str:
    cleaned = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "helmet": "Having_Helmet",
        "hardhat": "Having_Helmet",
        "hard_hat": "Having_Helmet",
        "no_helmet": "No_helmet",
        "vest": "Having_Vest",
        "none": "No_vest",
        "no_vest": "No_vest",
        "no_ppe": "No_vest",
        "boots": "Having_Boot",
        "boot": "Having_Boot",
        "no_boots": "No_boots",
        "no_boot": "No_boots",
        "gloves": "Having_Gloves",
        "glove": "Having_Gloves",
        "no_gloves": "No_gloves",
        "no_glove": "No_gloves",
    }
    return mapping.get(cleaned, label)


def summarize_detection(results, names):
    boxes = results[0].boxes
    labels = boxes.cls.tolist() if boxes is not None else []
    total = len(labels)
    counts = {}
    for lbl in labels:
        name = names[int(lbl)]
        counts[name] = counts.get(name, 0) + 1
    danger_keywords = ["danger", "hazard", "no ", "missing", "unsafe"]
    danger_count = sum(cnt for name, cnt in counts.items() if any(kw in name.lower() for kw in danger_keywords))
    return total, counts, danger_count


def alert_items_for_result(result, names, is_ppe: bool) -> tuple[set[str], set[str]]:
    if result.boxes is None or len(result.boxes) == 0:
        return set(), set()

    class_ids = [int(class_id) for class_id in result.boxes.cls.tolist()]
    labels = [str(names[class_id]).strip().lower().replace("-", "_").replace(" ", "_") for class_id in class_ids]

    if not is_ppe:
        hazards = {label for label in labels if label in {"smoke", "fire"}}
        return hazards, set()

    missing_items = set()
    present_items = set()

    for label in labels:
        if label in {"helmet", "hardhat", "hard_hat"}:
            present_items.add("Having_Helmet")
        elif label == "vest":
            present_items.add("Having_Vest")
        elif label in {"boots", "boot"}:
            present_items.add("Having_Boot")
        elif label in {"gloves", "glove"}:
            present_items.add("Having_Gloves")
        elif label in {"none", "no_vest", "no_ppe"}:
            missing_items.add("No_vest")
        elif label == "no_helmet":
            missing_items.add("No_helmet")
        elif label in {"no_boots", "no_boot"}:
            missing_items.add("No_boots")
        elif label in {"no_gloves", "no_glove"}:
            missing_items.add("No_gloves")

    has_person = any(label in {"person", "worker"} for label in labels)
    if has_person:
        if "Having_Helmet" not in present_items:
            missing_items.add("No_helmet")
        if "Having_Vest" not in present_items:
            missing_items.add("No_vest")

    return missing_items, present_items


def ppe_alerts_by_person(
    result, names, include_compliant: bool = False
) -> list[tuple[int, set[str], set[str], bool]]:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []

    coordinates = boxes.xyxy.tolist()
    class_ids = [int(class_id) for class_id in boxes.cls.tolist()]
    labels = [
        str(names[class_id]).strip().lower().replace("-", "_").replace(" ", "_")
        for class_id in class_ids
    ]
    track_ids = [int(track_id) for track_id in boxes.id.tolist()]
    class_names = names.values() if isinstance(names, dict) else names
    ppe_classes = {
        str(label).strip().lower().replace("-", "_").replace(" ", "_")
        for label in class_names
        if str(label).strip().lower() not in {"person", "worker", "none"} | IGNORED_PPE_CLASSES
        and not str(label).strip().lower().startswith("no_")
    }
    helmet_labels = {"helmet", "hardhat", "hard_hat"}
    person_records = [
        (track_id, box)
        for track_id, box, label in zip(track_ids, coordinates, labels)
        if label in {"person", "worker"} and track_id > 0
    ]
    alerts = []

    for track_id, person_box in person_records:
        person_present = set()
        missing_items = set()
        has_explicit_missing_ppe = False
        person_width = person_box[2] - person_box[0]
        person_height = person_box[3] - person_box[1]
        px1 = person_box[0] - person_width * 0.12
        px2 = person_box[2] + person_width * 0.12
        py1 = person_box[1] - person_height * 0.10
        py2 = person_box[3] + person_height * 0.10

        for box, label in zip(coordinates, labels):
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            if not (px1 <= center_x <= px2 and py1 <= center_y <= py2):
                continue

            if label == "none":
                missing_items.add("No_vest")
                has_explicit_missing_ppe = True
            elif label.startswith("no_") and label != "no_goggle":
                missing_items.add(get_ppe_display_label(label))
                has_explicit_missing_ppe = True
            elif label in ppe_classes:
                if label in helmet_labels and not (
                    person_box[1] - person_height * 0.08 <= center_y
                    <= person_box[1] + person_height * HEAD_REGION_HEIGHT
                ):
                    continue
                person_present.add(get_ppe_display_label(label))

        for ppe_item in ["helmet", "vest", "boots", "gloves"]:
            display_present = get_ppe_display_label(ppe_item)
            display_missing = get_ppe_display_label(f"no_{ppe_item}")
            if display_present not in person_present:
                missing_items.add(display_missing)

        if missing_items or include_compliant:
            alerts.append(
                (track_id, missing_items, person_present, has_explicit_missing_ppe)
            )

    return alerts


def update_person_tracking_state(
    result, names, tracking_state, alerted_person_ids: set | None = None
) -> None:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        current_person_ids = set()
    else:
        class_ids = [int(class_id) for class_id in boxes.cls.tolist()]
        labels = [
            str(names[class_id]).strip().lower().replace("-", "_").replace(" ", "_")
            for class_id in class_ids
        ]
        current_person_ids = {
            int(track_id)
            for track_id, label in zip(boxes.id.tolist(), labels)
            if label in {"person", "worker"} and int(track_id) > 0
        }

    frame_time = time.monotonic()
    counts = tracking_state["detection_counts"]
    last_seen = tracking_state["last_seen"]
    for person_id in current_person_ids:
        counts[person_id] = counts.get(person_id, 0) + 1
        last_seen[person_id] = frame_time

    for person_id, last_detection_time in list(last_seen.items()):
        if person_id not in current_person_ids and frame_time - last_detection_time > 8.0:
            last_seen.pop(person_id, None)
            counts.pop(person_id, None)
            if alerted_person_ids is not None:
                alerted_person_ids.discard(person_id)


def filter_result_by_confidence(result, confidence: float, names: dict) -> None:
    if result.boxes is None or len(result.boxes) == 0:
        return

    keep_indices = []
    for index, (score, cls_id) in enumerate(zip(result.boxes.conf.tolist(), result.boxes.cls.tolist())):
        label = str(names.get(int(cls_id), "")).strip().lower().replace("-", "_").replace(" ", "_")
        # Small items like gloves have lower peak confidence (0.14 - 0.42)
        if "glove" in label:
            required_thresh = min(confidence, 0.14)
        elif "boot" in label:
            required_thresh = min(confidence, 0.20)
        else:
            required_thresh = confidence
        if score >= required_thresh:
            keep_indices.append(index)

    if len(keep_indices) != len(result.boxes):
        result.update(boxes=result.boxes.data[keep_indices])


def send_alert_snapshot_to_firebase(
    image,
    captured_at: str,
    missing_items: set[str],
    present_items: set[str],
    alert_items: set[str],
    source_name: str,
) -> dict:
    global firebase_app
    if not FIREBASE_DATABASE_URL or not FIREBASE_CREDENTIALS_PATH.is_file():
        return {"configured": False, "saved": False}

    try:
        import firebase_admin
        from firebase_admin import credentials, db, messaging

        with firebase_init_lock:
            if firebase_app is None:
                try:
                    firebase_app = firebase_admin.get_app("ppe-alerts")
                except ValueError:
                    firebase_app = firebase_admin.initialize_app(
                        credentials.Certificate(str(FIREBASE_CREDENTIALS_PATH)),
                        {"databaseURL": FIREBASE_DATABASE_URL},
                        name="ppe-alerts",
                    )

        height, width = image.shape[:2]
        scale = min(1, 512 / max(height, width))
        if scale < 1:
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        encoded, image_buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        if not encoded:
            raise RuntimeError("Could not encode the alert snapshot image.")

        snapshot_data = "data:image/jpeg;base64," + base64.b64encode(
            image_buffer.tobytes()
        ).decode("ascii")
        alert_record = db.reference("ppe_alerts", app=firebase_app).push({
            "captured_at": captured_at,
            "alert_items": sorted(alert_items),
            "missing_items": sorted(missing_items),
            "present_items": sorted(present_items),
            "snapshot_data": snapshot_data,
            "source": source_name,
        })

        normalized_items = sorted(item.lower() for item in alert_items)
        notification_targets = []
        if any(item in {"smoke", "fire"} for item in normalized_items):
            notification_targets.append(
                ("fire_safety_alerts", "Fire Safety Alert", "fire_safety_alerts")
            )
        if any(item.startswith("no_") for item in normalized_items) or missing_items:
            notification_targets.append(
                ("ppe_safety_alerts", "PPE Safety Alert", "ppe_safety_alerts")
            )

        notifications_sent = []
        notification_errors = []
        for topic, title, channel_id in notification_targets:
            try:
                messaging.send(
                    messaging.Message(
                        topic=topic,
                        notification=messaging.Notification(
                            title=title,
                            body=(
                                f"{', '.join(normalized_items)} at {captured_at}. "
                                "Open the app for the snapshot."
                            ),
                        ),
                        data={
                            "record_id": alert_record.key,
                            "captured_at": captured_at,
                            "source": source_name,
                        },
                        android=messaging.AndroidConfig(
                            priority="high",
                            notification=messaging.AndroidNotification(channel_id=channel_id),
                        ),
                    ),
                    app=firebase_app,
                )
                notifications_sent.append(topic)
            except Exception as error:
                logging.exception("Firebase alert was saved but push to %s failed.", topic)
                notification_errors.append(f"{topic}: {error}")

        return {
            "configured": True,
            "saved": True,
            "record_id": alert_record.key,
            "notifications_sent": notifications_sent,
            "notification_errors": notification_errors,
        }
    except Exception as error:
        logging.exception("Could not upload alert snapshot to Firebase.")
        return {"configured": True, "saved": False, "error": str(error)}


def create_alert_entry(
    image,
    missing_items,
    present_items,
    alert_items,
    source_name: str,
    person_id: int | None = None,
    crop_image: np.ndarray | None = None,
    captured_at: str | None = None,
    video_timestamp: str | None = None,
    visitor_message: str | None = None,
) -> dict:
    captured_at = captured_at or datetime.now().astimezone().isoformat(timespec="milliseconds")
    upload_img = crop_image if crop_image is not None else image
    firebase_sync = (
        send_alert_snapshot_to_firebase(
            upload_img,
            captured_at,
            missing_items,
            present_items,
            alert_items,
            source_name,
        )
        if missing_items or alert_items
        else {"configured": False, "saved": False}
    )
    return {
        "person_id": person_id,
        "captured_at": captured_at,
        "alert_items": sorted(alert_items),
        "missing_items": sorted(missing_items),
        "present_items": sorted(present_items),
        "image": image.copy(),
        "crop_image": crop_image.copy() if crop_image is not None else None,
        "firebase_sync": firebase_sync,
        "source": source_name,
        "video_timestamp": video_timestamp,
        "visitor_message": visitor_message,
    }


def append_alert_entry(alert_entries, entry) -> None:
    alert_entries.append(entry)
    del alert_entries[:-15]


def render_alert_dashboard(alert_container, alert_entries):
    with alert_container.container():
        if not alert_entries:
            st.caption("No alerts recorded in this session.")
            return

        with st.container(height=560, border=True):
            st.markdown("### 🚨 Alert")
            st.caption(f"{len(alert_entries)} alert(s) · scroll to browse")

            for index, entry in enumerate(reversed(alert_entries), start=1):
                alert_items = set(entry.get("alert_items") or [])
                hazard_items = alert_items & {"fire", "smoke"}
                is_hazard_alert = bool(hazard_items) and not entry.get("missing_items")
                if is_hazard_alert:
                    hazard_names = ", ".join(
                        label.title() for label in ("fire", "smoke") if label in hazard_items
                    )
                else:
                    missing_str = ", ".join(entry.get("missing_items") or entry["alert_items"])
                present_str = ", ".join(entry.get("present_items") or [])
                img_to_show = entry.get("crop_image") if entry.get("crop_image") is not None else entry["image"]
                sync = entry.get("firebase_sync") or {}

                with st.container(border=True):
                    st.caption(f"Alert {index} · {entry['captured_at']}")
                    if entry.get("video_timestamp"):
                        st.caption(f"Video position: {entry['video_timestamp']}")
                    col_image, col_details = st.columns([1, 1.35], gap="small")
                    with col_image:
                        st.image(img_to_show, channels="BGR", width=130)
                    with col_details:
                        if is_hazard_alert:
                            st.markdown(f"**🔥 {hazard_names} Detected**")
                        elif entry.get("visitor_message") == "Welcome":
                            st.markdown("**PPE Status**  \nCompliant")
                            st.markdown(f"**Present PPE**  \n{present_str or 'None'}")
                            st.success("Welcome")
                        else:
                            st.markdown(f"**Missing PPE**  \n{missing_str or 'None'}")
                            st.markdown(f"**Present PPE**  \n{present_str or 'None'}")
                            if entry.get("visitor_message") == "Go Back and come after 30min":
                                st.error("Go Back and come after 30min")
                        firebase_status = "Saved" if sync.get("saved") else "Not saved"
                        st.markdown(f"**Firebase**  \n{firebase_status}")

                    if sync.get("saved"):
                        st.caption(
                            f"Record {sync['record_id']}; notifications sent: "
                            f"{', '.join(sync.get('notifications_sent', [])) or 'none'}."
                        )
                        for notification_error in sync.get("notification_errors", []):
                            st.warning(f"Push notification failed: {notification_error}")
                    elif sync.get("configured"):
                        st.error(f"Firebase save failed: {sync.get('error', 'Unknown error')}")



def resolve_ppe_conflicts(result, names: dict[int, str]) -> None:
    """Eliminates conflicting duplicate predictions such as Having_Vest vs No_vest on the same chest."""
    if result.boxes is None or len(result.boxes) <= 1:
        return

    boxes = result.boxes.xyxy.tolist()
    class_ids = [int(c) for c in result.boxes.cls.tolist()]
    confs = result.boxes.conf.tolist()
    n = len(boxes)
    suppressed = set()

    def box_iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / max(1.0, min(area1, area2))

    for i in range(n):
        if i in suppressed:
            continue
        name_i = str(names.get(class_ids[i], "")).lower().replace("-", "_")
        for j in range(i + 1, n):
            if j in suppressed:
                continue
            name_j = str(names.get(class_ids[j], "")).lower().replace("-", "_")

            iou = box_iou(boxes[i], boxes[j])
            if iou > 0.35:
                is_vest_i = name_i in {"vest", "having_vest"}
                is_none_i = name_i in {"none", "no_vest", "no_ppe"}
                is_vest_j = name_j in {"vest", "having_vest"}
                is_none_j = name_j in {"none", "no_vest", "no_ppe"}

                if (is_vest_i and is_none_j) or (is_none_i and is_vest_j):
                    none_idx = i if is_none_i else j
                    vest_idx = j if is_none_i else i
                    if confs[none_idx] >= 0.40:
                        suppressed.add(vest_idx)
                    elif confs[vest_idx] > confs[none_idx] + 0.25:
                        suppressed.add(none_idx)
                    else:
                        suppressed.add(vest_idx)
                elif name_i == name_j and name_i in {"boots", "boot", "having_boot"}:
                    worse = j if confs[i] >= confs[j] else i
                    suppressed.add(worse)

    if suppressed:
        keep = [k for k in range(n) if k not in suppressed]
        result.update(boxes=result.boxes.data[keep])

def plot_detections(result, names, is_ppe: bool):
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return result.plot(labels=False)

    if is_ppe:
        filter_helmets_on_heads(result, names)
        resolve_ppe_conflicts(result, names)

    boxes = result.boxes.xyxy.tolist()
    class_ids = [int(c) for c in result.boxes.cls.tolist()]
    confs = result.boxes.conf.tolist()
    track_ids = result.boxes.id.tolist() if result.boxes.is_track else [None] * len(boxes)

    for box, class_id, confidence, track_id in zip(boxes, class_ids, confs, track_ids):
        raw_name = names.get(class_id, "")
        clean_name = str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")

        if clean_name in {"person", "worker"}:
            track_id_str = f" ID {int(track_id)}" if (track_id is not None and int(track_id) > 0) else ""
            label = f"Person{track_id_str} {confidence:.2f}"
        else:
            display_name = get_ppe_display_label(raw_name)
            label = f"{display_name} {confidence:.2f}"

        detections.append((*box, label))

    return draw_detection_labels(result.plot(labels=False), detections)


def filter_helmets_on_heads(result, names: dict[int, str]) -> None:
    if result.boxes is None or len(result.boxes) == 0:
        return

    boxes = result.boxes.xyxy.tolist()
    class_ids = result.boxes.cls.tolist()
    person_boxes = [
        box
        for box, class_id in zip(boxes, class_ids)
        if names[int(class_id)].strip().lower() in {"person", "worker"}
    ]
    helmet_class_ids = {
        class_id
        for class_id, name in names.items()
        if name.strip().lower().replace("-", "_").replace(" ", "_") in {"helmet", "hardhat", "hard_hat"}
    }

    keep_indices = []
    for index, (box, class_id) in enumerate(zip(boxes, class_ids)):
        if int(class_id) not in helmet_class_ids:
            keep_indices.append(index)
            continue

        helmet_center_x = (box[0] + box[2]) / 2
        helmet_center_y = (box[1] + box[3]) / 2
        is_on_a_head = any(
            person[2] > person[0]
            and person[3] > person[1]
            and person[0] <= helmet_center_x <= person[2]
            and person[1] <= helmet_center_y <= person[1] + (person[3] - person[1]) * HEAD_REGION_HEIGHT
            for person in person_boxes
        )
        if is_on_a_head:
            keep_indices.append(index)

    if len(keep_indices) != len(boxes):
        result.update(boxes=result.boxes.data[keep_indices])


def find_available_cameras(max_index: int = MAX_CAMERA_INDEX) -> list[int]:
    available_cameras = []
    for camera_index in range(max_index):
        capture = cv2.VideoCapture(camera_index)
        try:
            opened, frame = capture.read() if capture.isOpened() else (False, None)
            if opened and frame is not None and frame.size > 0:
                available_cameras.append(camera_index)
        finally:
            capture.release()
    return available_cameras


def process_video(
    uploaded_video,
    output_dir: Path,
    conf: float,
    detector,
    names,
    is_ppe: bool,
    vid_stride: int = 2,
    imgsz: int = 480,
    progress_callback=None,
):
    input_path = output_dir / f"{sanitize_name(uploaded_video.name)}_input{Path(uploaded_video.name).suffix or '.mp4'}"
    with input_path.open("wb") as video_file:
        video_file.write(uploaded_video.getvalue())

    metadata_capture = cv2.VideoCapture(str(input_path))
    if not metadata_capture.isOpened():
        raise ValueError("Unable to read the uploaded video file.")

    fps = metadata_capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(metadata_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(metadata_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata_capture.release()
    output_path = output_dir / f"{sanitize_name(uploaded_video.name)}_processed.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    if not writer.isOpened():
        raise ValueError("Video writer failed to initialize. Try a different video format.")

    frame_count = 0
    written_frames = 0
    summary_counts = {}
    item_frame_counts = {}
    hazard_first_detections = {}
    total_detected = 0
    alert_image = None
    best_alert_score = 0.0

    video_tracker = StablePersonTracker() if is_ppe else None
    predict_conf = min(conf, 0.10) if is_ppe else conf
    person_stats = {}

    with torch.inference_mode():
        results_stream = detector.predict(
            source=str(input_path),
            conf=predict_conf,
            vid_stride=vid_stride,
            imgsz=imgsz,
            stream=True,
            verbose=False,
        )
        try:
            for result in results_stream:
                if is_ppe:
                    filter_result_by_confidence(result, conf, names)
                    resolve_ppe_conflicts(result, names)
                frame_count += 1
                if is_ppe and video_tracker is not None:
                    video_tracker.update(result, result.orig_img, names)
                annotated = plot_detections(result, names, is_ppe)

                repeat = min(vid_stride, total_frames - written_frames) if total_frames > 0 else vid_stride
                for _ in range(max(1, repeat)):
                    writer.write(annotated)
                    written_frames += 1

                if progress_callback and total_frames > 0:
                    pct = min(1.0, written_frames / total_frames)
                    progress_callback(
                        pct,
                        f"Processing video: {int(pct * 100)}% ({written_frames}/{total_frames} frames)",
                    )

                if is_ppe and result.boxes is not None and result.boxes.id is not None:
                    for pid, missing, present, explicit in ppe_alerts_by_person(
                        result, names, include_compliant=True
                    ):
                        if pid not in person_stats:
                            person_stats[pid] = {
                                "missing_counts": {},
                                "present_counts": {},
                                "total_frames": 0,
                                "best_image": None,
                                "best_crop": None,
                            }
                        pdata = person_stats[pid]
                        pdata["total_frames"] += 1
                        for m in missing:
                            pdata["missing_counts"][m] = pdata["missing_counts"].get(m, 0) + 1
                        for p in present:
                            pdata["present_counts"][p] = pdata["present_counts"].get(p, 0) + 1

                        if pdata["best_crop"] is None or explicit or len(missing) > 0:
                            pdata["best_image"] = annotated.copy()
                            for b, tid in zip(result.boxes.xyxy.tolist(), result.boxes.id.tolist()):
                                if int(tid) == pid:
                                    h_img, w_img = annotated.shape[:2]
                                    pad = 20
                                    cx1 = max(0, int(b[0]) - pad)
                                    cy1 = max(0, int(b[1]) - pad)
                                    cx2 = min(w_img, int(b[2]) + pad)
                                    cy2 = min(h_img, int(b[3]) + pad)
                                    if cx2 > cx1 and cy2 > cy1:
                                        pdata["best_crop"] = annotated[cy1:cy2, cx1:cx2].copy()

                if result.boxes is not None:
                    for cls_id, box_conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                        name = names[int(cls_id)]
                        disp_name = get_ppe_display_label(name)
                        item_frame_counts[disp_name] = item_frame_counts.get(disp_name, 0) + 1
                        summary_counts[disp_name] = summary_counts.get(disp_name, 0) + 1
                        total_detected += 1
                        if not is_ppe and disp_name in {"smoke", "fire"}:
                            video_seconds = (frame_count - 1) * vid_stride / fps
                            total_milliseconds = round(video_seconds * 1000)
                            hours, remainder = divmod(total_milliseconds, 3_600_000)
                            minutes, remainder = divmod(remainder, 60_000)
                            seconds, milliseconds = divmod(remainder, 1000)
                            video_timestamp = (
                                f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
                            )
                            hazard_first_detections.setdefault(
                                disp_name,
                                {
                                    "frame": frame_count,
                                    "captured_at": datetime.now().astimezone().isoformat(
                                        timespec="milliseconds"
                                    ),
                                    "video_timestamp": video_timestamp,
                                    "image": annotated.copy(),
                                },
                            )
                        if is_ppe and disp_name.startswith("No_"):
                            if box_conf > best_alert_score:
                                best_alert_score = box_conf
                                alert_image = annotated.copy()
                        elif not is_ppe and disp_name in {"smoke", "fire"}:
                            if box_conf > best_alert_score:
                                best_alert_score = box_conf
                                alert_image = annotated.copy()

                if alert_image is None and frame_count == 1:
                    alert_image = annotated.copy()

            while written_frames < total_frames and alert_image is not None:
                writer.write(annotated)
                written_frames += 1
        finally:
            results_stream.close()
            writer.release()

    alert_items = set()
    missing_items = set()
    present_items = set()
    person_alerts = []

    if is_ppe:
        for pid, pdata in sorted(person_stats.items()):
            p_frames = pdata["total_frames"]
            if p_frames < 3:
                continue
            p_thresh = max(2, int(p_frames * 0.10))
            gloves_thresh = max(2, int(p_frames * 0.04))
            p_present = set()
            for item, cnt in pdata["present_counts"].items():
                req = gloves_thresh if item == "Having_Gloves" else p_thresh
                if cnt >= req:
                    p_present.add(item)

            p_missing = set()
            for item in ["Having_Helmet", "Having_Vest", "Having_Boot", "Having_Gloves"]:
                display_missing = get_ppe_display_label(f"no_{item.replace('Having_', '').lower()}")
                if item not in p_present:
                    p_missing.add(display_missing)

            for item, cnt in pdata["missing_counts"].items():
                req = gloves_thresh if item == "No_gloves" else p_thresh
                if cnt >= req:
                    p_missing.add(item)

            # Eliminate contradictions: confirmed worn items must not be marked missing
            for item in ["Having_Helmet", "Having_Vest", "Having_Boot", "Having_Gloves"]:
                if item in p_present:
                    disp_no = get_ppe_display_label(f"no_{item.replace('Having_', '').lower()}")
                    p_missing.discard(disp_no)

            if p_missing:
                missing_items.update(p_missing)
                alert_items.update(p_missing)
                snap = pdata["best_image"] if pdata["best_image"] is not None else alert_image
                entry = create_alert_entry(
                    snap if snap is not None else annotated,
                    p_missing,
                    p_present,
                    p_missing,
                    "video_upload",
                    person_id=pid,
                    crop_image=pdata["best_crop"],
                    visitor_message="Go Back and come after 30min",
                )
                person_alerts.append(entry)
            else:
                snap = pdata["best_image"] if pdata["best_image"] is not None else alert_image
                person_alerts.append(
                    create_alert_entry(
                        snap if snap is not None else annotated,
                        set(),
                        p_present,
                        set(),
                        "video_upload",
                        person_id=pid,
                        crop_image=pdata["best_crop"],
                        visitor_message="Welcome",
                    )
                )
            present_items.update(p_present)
    else:
        for hazard, first_detection in sorted(
            hazard_first_detections.items(), key=lambda item: item[1]["frame"]
        ):
            if item_frame_counts.get(hazard, 0) >= 3:
                alert_items.add(hazard)
                person_alerts.append(
                    create_alert_entry(
                        first_detection["image"],
                        set(),
                        set(),
                        {hazard},
                        "video_upload",
                        captured_at=first_detection["captured_at"],
                        video_timestamp=first_detection["video_timestamp"],
                    )
                )

    summary_text = [
        f"Source video: {uploaded_video.name}",
        f"Total frames written: {written_frames}",
        f"Keyframes analyzed: {frame_count} (speed: vid_stride={vid_stride}, imgsz={imgsz})",
        f"Total detections: {total_detected}",
    ]
    if summary_counts:
        summary_text.extend(f"- {name}: {count}" for name, count in sorted(summary_counts.items()))
    else:
        summary_text.append("- No objects detected.")

    if is_ppe and person_stats:
        summary_text.append("")
        summary_text.append("Per-person safety evaluation:")
        for pid, pdata in sorted(person_stats.items()):
            p_frames = pdata["total_frames"]
            if p_frames < 3:
                continue
            p_thresh = max(2, int(p_frames * 0.10))
            gloves_thresh = max(2, int(p_frames * 0.04))
            p_present = set()
            for item, cnt in pdata["present_counts"].items():
                req = gloves_thresh if item == "Having_Gloves" else p_thresh
                if cnt >= req:
                    p_present.add(item)
            p_missing = set()
            for item in ["Having_Helmet", "Having_Vest", "Having_Boot", "Having_Gloves"]:
                display_missing = get_ppe_display_label(f"no_{item.replace('Having_', '').lower()}")
                if item not in p_present:
                    p_missing.add(display_missing)
            for item, cnt in pdata["missing_counts"].items():
                req = gloves_thresh if item == "No_gloves" else p_thresh
                if cnt >= req:
                    p_missing.add(item)
            for item in ["Having_Helmet", "Having_Vest", "Having_Boot", "Having_Gloves"]:
                if item in p_present:
                    disp_no = get_ppe_display_label(f"no_{item.replace('Having_', '').lower()}")
                    p_missing.discard(disp_no)
            status_str = f"VIOLATIONS: {', '.join(sorted(p_missing))}" if p_missing else "FULLY COMPLIANT"
            summary_text.append(f"- Person ID {pid}: {status_str} (Wearing: {', '.join(sorted(p_present)) or 'none'})")

    summary_path = output_dir / "detection_summary.txt"
    summary_path.write_text("\n".join(summary_text), encoding="utf-8")

    return output_path, summary_path, alert_items, missing_items, present_items, alert_image, person_alerts


if requires_rfid or source != "Camera":
    st.session_state["fire_smoke_camera_running"] = False

if not requires_rfid or source != "Camera":
    manual_capture = st.session_state.pop("ppe_manual_camera_capture", None)
    if manual_capture is not None:
        manual_capture.release()
    st.session_state["ppe_manual_camera_running"] = False

if source == "Web":
    if not rfid_ready:
        st.warning("Web camera is locked. Scan a registered RFID card; the status updates automatically.")
    else:
        if "alert_entries" not in st.session_state:
            st.session_state.alert_entries = []
        if "web_alert_queue" not in st.session_state:
            st.session_state.web_alert_queue = Queue()
        web_alert_queue = st.session_state.web_alert_queue
        try:
            while True:
                append_alert_entry(st.session_state.alert_entries, web_alert_queue.get_nowait())
        except Empty:
            pass
        camera_column, alert_column = st.columns([2, 1])
        alert_panel = alert_column.empty()
        render_alert_dashboard(alert_panel, st.session_state.alert_entries)

        with camera_column:
            if requires_rfid:
                st.info(
                    "RFID CARD: MATCHED — browser camera started. "
                    "Allow camera access when your browser asks, then choose your external/USB camera."
                )
            else:
                st.info(
                    "Browser camera is ready. Allow camera access when your browser asks, "
                    "then choose your external/USB camera."
                )
            with st.expander("External camera not listed?"):
                st.markdown(
                    "- Allow camera access for this site in your browser, then reload the page.\n"
                    "- Connect the camera before reloading and close other apps that may be using it.\n"
                    "- On Windows, allow desktop apps to access the camera in **Settings → Privacy & security → Camera**.\n"
                    "- If this app is not opened on `localhost`, use HTTPS; browsers block camera access on insecure pages."
                )

        browser_alert_state = st.session_state.setdefault(
            "browser_alert_state", {"active_items": None, "start_time": None}
        )
        browser_alert_lock = Lock()
        browser_tracking_key = ("browser_camera", model_path)
        browser_tracking_model = (
            load_tracking_model(model_path, browser_tracking_key, model_names)
            if is_ppe_model
            else model
        )
        person_tracking_states = st.session_state.setdefault("person_tracking_states", {})
        browser_person_tracking_state = person_tracking_states.setdefault(
            browser_tracking_key, {"detection_counts": {}, "last_seen": {}}
        )
        alerted_person_ids = st.session_state.setdefault("alerted_person_ids", {})
        web_alerted_person_ids = alerted_person_ids.setdefault(browser_tracking_key, set())

        person_trackers = st.session_state.setdefault("person_trackers", {})
        browser_person_tracker = person_trackers.setdefault(
            browser_tracking_key, StablePersonTracker()
        )

        def process_browser_frame(frame):
            image = frame.to_ndarray(format="bgr24")
            with browser_alert_lock:
                if browser_alert_state.get("start_time") is None:
                    browser_alert_state["start_time"] = time.time()
                elapsed_time = time.time() - browser_alert_state["start_time"]
                is_warmup = elapsed_time < camera_alert_delay
                remaining_warmup = max(0, int(camera_alert_delay - elapsed_time) + 1)

            if is_ppe_model:
                results = model.predict(
                    source=image,
                    conf=min(conf_threshold, 0.10),
                    classes=tracking_classes,
                    verbose=False,
                )
                result = results[0]
                filter_result_by_confidence(result, conf_threshold, model_names)
                browser_person_tracker.update(result, image, model_names)
            else:
                results = model.predict(source=image, conf=conf_threshold, verbose=False)
                result = results[0]
            annotated = plot_detections(result, model_names, is_ppe_model)
            if is_warmup:
                cv2.putText(
                    annotated,
                    f"Camera Warmup: Alert starts in {remaining_warmup}s",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            frame_alerts, frame_present = alert_items_for_result(result, model_names, is_ppe_model)
            if is_ppe_model:
                with browser_alert_lock:
                    update_person_tracking_state(
                        result, model_names, browser_person_tracking_state, web_alerted_person_ids
                    )
                    if not is_warmup:
                        for (
                            person_id,
                            missing_items,
                            present_items,
                            has_explicit_missing_ppe,
                        ) in ppe_alerts_by_person(
                            result, model_names, include_compliant=True
                        ):
                            if person_id in web_alerted_person_ids:
                                continue
                            if (
                                browser_person_tracking_state["detection_counts"].get(person_id, 0) < 10
                                and not has_explicit_missing_ppe
                            ):
                                continue
                            crop_snap = None
                            if result.boxes is not None and result.boxes.id is not None:
                                for b, tid in zip(result.boxes.xyxy.tolist(), result.boxes.id.tolist()):
                                    if int(tid) == person_id:
                                        h_img, w_img = annotated.shape[:2]
                                        pad = 20
                                        cx1 = max(0, int(b[0]) - pad)
                                        cy1 = max(0, int(b[1]) - pad)
                                        cx2 = min(w_img, int(b[2]) + pad)
                                        cy2 = min(h_img, int(b[3]) + pad)
                                        if cx2 > cx1 and cy2 > cy1:
                                            crop_snap = annotated[cy1:cy2, cx1:cx2].copy()
                            entry = create_alert_entry(
                                annotated,
                                missing_items,
                                present_items,
                                missing_items,
                                "browser_camera",
                                person_id=person_id,
                                crop_image=crop_snap,
                                visitor_message=(
                                    "Go Back and come after 30min" if missing_items else "Welcome"
                                ),
                            )
                            web_alert_queue.put(entry)
                            web_alerted_person_ids.add(person_id)
            else:
                with browser_alert_lock:
                    if not is_warmup:
                        previous_items = set(browser_alert_state["active_items"] or ())
                        new_items = frame_alerts - previous_items
                        browser_alert_state["active_items"] = tuple(sorted(frame_alerts))
                        detected_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
                        for hazard in ("fire", "smoke"):
                            if hazard in new_items:
                                web_alert_queue.put(
                                    create_alert_entry(
                                        annotated,
                                        set(),
                                        set(),
                                        {hazard},
                                        "browser_camera",
                                        captured_at=detected_at,
                                    )
                                )
            return VideoFrame.from_ndarray(annotated, format="bgr24")

        with camera_column:
            webrtc_ctx = webrtc_streamer(
                key="web-camera",
                video_frame_callback=process_browser_frame,
                media_stream_constraints={"video": True, "audio": False},
            )
            if webrtc_ctx and not webrtc_ctx.state.playing:
                with browser_alert_lock:
                    browser_alert_state["start_time"] = None
                    browser_alert_state["active_items"] = None
                    web_alerted_person_ids.clear()
                    browser_person_tracker.reset()

elif source == "Video":
    st.info("Upload a video file for detection. An output folder will contain the annotated result and summary.")
    # Process every frame at 640p for maximum detection detail.
    vid_stride = 1
    imgsz = 640

    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv", "wmv"])

    if uploaded_video is not None:
        output_root = st.sidebar.text_input("Output folder", value="output")
        output_dir = create_output_folder(base_dir=output_root, name=uploaded_video.name)

        progress_container = st.empty()
        progress_bar = progress_container.progress(0, text="Starting video processing...")

        def update_progress(pct: float, message: str) -> None:
            progress_bar.progress(pct, text=message)

        with st.spinner("Processing video frames..."):
            (
                output_path,
                summary_path,
                alert_items,
                missing_items,
                present_items,
                alert_image,
                person_alerts,
            ) = process_video(
                uploaded_video,
                output_dir,
                conf_threshold,
                model,
                model_names,
                is_ppe_model,
                vid_stride=vid_stride,
                imgsz=imgsz,
                progress_callback=update_progress,
            )
        progress_container.empty()

        if "alert_entries" not in st.session_state:
            st.session_state.alert_entries = []
        video_alert_id = (
            uploaded_video.name,
            hashlib.sha256(uploaded_video.getvalue()).hexdigest(),
            st.session_state.selected_detector,
            conf_threshold,
        )
        if st.session_state.get("last_video_alert_id") != video_alert_id:
            if person_alerts:
                entries_to_add = person_alerts if is_ppe_model else reversed(person_alerts)
                for p_entry in entries_to_add:
                    append_alert_entry(st.session_state.alert_entries, p_entry)
            elif alert_items and alert_image is not None:
                append_alert_entry(
                    st.session_state.alert_entries,
                    create_alert_entry(
                        alert_image,
                        missing_items,
                        present_items,
                        alert_items,
                        "video_upload",
                    )
                )
            st.session_state.last_video_alert_id = video_alert_id
        st.success(f"Video processed successfully. Output folder: {output_dir}")
        alert_panel = st.empty()
        render_alert_dashboard(alert_panel, st.session_state.get("alert_entries", []))
        st.write(f"Processed video: {output_path.name}")
        st.write(f"Summary file: {summary_path.name}")
        st.video(str(output_path))

        with open(output_path, "rb") as video_file:
            st.download_button(
                label="Download processed video",
                data=video_file.read(),
                file_name=output_path.name,
                mime="video/mp4",
            )

        st.subheader("Folder contents")
        for item in sorted(output_dir.iterdir()):
            st.write(f"- {item.name}")

elif source == "Camera":
    st.info("Connect the webcam to the computer running this Streamlit app.")
    if "available_cameras" not in st.session_state or st.sidebar.button("Refresh camera list"):
        st.session_state.available_cameras = find_available_cameras()

    available_cameras = st.session_state.available_cameras
    camera_index = None
    if not available_cameras:
        st.warning("No cameras detected. Connect the external webcam and refresh the camera list.")
    else:
        preferred_camera = 1 if 1 in available_cameras else available_cameras[0]
        camera_index = st.sidebar.selectbox(
            "Camera device",
            options=available_cameras,
            index=available_cameras.index(preferred_camera),
            format_func=lambda index: f"Camera {index}",
        )
    if requires_rfid:
        manual_ppe_camera_running = st.session_state.setdefault("ppe_manual_camera_running", False)
        if st.sidebar.button(
            "Stop Manual PPE Camera" if manual_ppe_camera_running else "Start PPE Camera Without RFID",
            disabled=not available_cameras,
            use_container_width=True,
        ):
            if manual_ppe_camera_running:
                manual_capture = st.session_state.pop("ppe_manual_camera_capture", None)
                if manual_capture is not None:
                    manual_capture.release()
                st.session_state.ppe_manual_camera_running = False
                manual_ppe_camera_running = False
            else:
                manual_capture = cv2.VideoCapture(camera_index)
                if manual_capture.isOpened():
                    st.session_state.ppe_manual_camera_capture = manual_capture
                    st.session_state.ppe_manual_camera_running = True
                    st.session_state.ppe_manual_camera_started_at = time.time()
                    st.session_state.rfid_session_started_at = None
                    st.session_state.rfid_session_deadline = None
                    if rfid_monitor is not None:
                        rfid_monitor.status.matched = False
                    manual_ppe_camera_running = True
                else:
                    manual_capture.release()
                    st.sidebar.error("Could not open the selected camera device.")
        if manual_ppe_camera_running:
            st.sidebar.success("PPE camera is running without RFID. Use Stop Manual PPE Camera to close it.")
        else:
            st.sidebar.caption("Manual PPE camera mode runs continuously without RFID until stopped.")

    if requires_rfid:
        run = rfid_ready or manual_ppe_camera_running
    else:
        camera_running = st.session_state.setdefault("fire_smoke_camera_running", False)
        if st.sidebar.button(
            "Stop Camera" if camera_running else "Start Camera",
            use_container_width=True,
        ):
            camera_running = not camera_running
            st.session_state.fire_smoke_camera_running = camera_running
        run = camera_running

    if requires_rfid and rfid_ready and not manual_ppe_camera_running:
        st.sidebar.success("Webcam auto-started after RFID CARD: MATCHED.")
    elif not requires_rfid:
        if run:
            st.sidebar.success("Fire and Smoke camera is running.")
        else:
            st.sidebar.info("Click Start Camera when you are ready to begin Fire and Smoke detection.")
    if "alert_entries" not in st.session_state:
        st.session_state.alert_entries = []
    camera_tracking_key = (
        "live_camera",
        model_path,
        camera_index if available_cameras else None,
    )
    person_tracking_states = st.session_state.setdefault("person_tracking_states", {})
    camera_person_tracking_state = person_tracking_states.setdefault(
        camera_tracking_key, {"detection_counts": {}, "last_seen": {}}
    )
    alerted_person_ids = st.session_state.setdefault("alerted_person_ids", {})
    camera_alerted_person_ids = alerted_person_ids.setdefault(camera_tracking_key, set())
    camera_column, alert_column = st.columns([2, 1])
    alert_panel = alert_column.empty()
    render_alert_dashboard(alert_panel, st.session_state.alert_entries)
    with camera_column:
        FRAME_WINDOW = st.empty()
        status_text = st.empty()

    person_trackers = st.session_state.setdefault("person_trackers", {})
    camera_person_tracker = person_trackers.setdefault(
        camera_tracking_key, StablePersonTracker()
    )

    if manual_ppe_camera_running and available_cameras:
        @st.fragment(run_every=0.15)
        def render_manual_ppe_camera():
            manual_capture = st.session_state.get("ppe_manual_camera_capture")
            if not st.session_state.get("ppe_manual_camera_running") or manual_capture is None:
                return

            success, frame = manual_capture.read()
            if not success:
                st.warning("Camera stopped providing frames. Check the connection and restart it.")
                manual_capture.release()
                st.session_state.pop("ppe_manual_camera_capture", None)
                st.session_state.ppe_manual_camera_running = False
                return

            results = model.predict(
                source=frame,
                conf=min(conf_threshold, 0.10),
                classes=tracking_classes,
                verbose=False,
            )
            result = results[0]
            filter_result_by_confidence(result, conf_threshold, model_names)
            camera_person_tracker.update(result, frame, model_names)
            annotated = plot_detections(result, model_names, is_ppe_model)

            elapsed_time = time.time() - st.session_state.get(
                "ppe_manual_camera_started_at", time.time()
            )
            is_warmup = elapsed_time < camera_alert_delay
            remaining_warmup = max(0, int(camera_alert_delay - elapsed_time) + 1)
            if is_warmup:
                cv2.putText(
                    annotated,
                    f"Camera Warmup: Alert starts in {remaining_warmup}s",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
            FRAME_WINDOW.image(annotated, channels="BGR")

            update_person_tracking_state(
                result, model_names, camera_person_tracking_state, camera_alerted_person_ids
            )
            if not is_warmup:
                for person_id, missing_items, present_items, has_explicit_missing_ppe in ppe_alerts_by_person(
                    result, model_names, include_compliant=True
                ):
                    if person_id in camera_alerted_person_ids:
                        continue
                    if (
                        camera_person_tracking_state["detection_counts"].get(person_id, 0) < 10
                        and not has_explicit_missing_ppe
                    ):
                        continue
                    crop_snap = None
                    if result.boxes is not None and result.boxes.id is not None:
                        for box, track_id in zip(result.boxes.xyxy.tolist(), result.boxes.id.tolist()):
                            if int(track_id) == person_id:
                                image_height, image_width = annotated.shape[:2]
                                padding = 20
                                x1 = max(0, int(box[0]) - padding)
                                y1 = max(0, int(box[1]) - padding)
                                x2 = min(image_width, int(box[2]) + padding)
                                y2 = min(image_height, int(box[3]) + padding)
                                if x2 > x1 and y2 > y1:
                                    crop_snap = annotated[y1:y2, x1:x2].copy()
                    entry = create_alert_entry(
                        annotated,
                        missing_items,
                        present_items,
                        missing_items,
                        "live_camera_manual",
                        person_id=person_id,
                        crop_image=crop_snap,
                        visitor_message=(
                            "Go Back and come after 30min" if missing_items else "Welcome"
                        ),
                    )
                    append_alert_entry(st.session_state.alert_entries, entry)
                    camera_alerted_person_ids.add(person_id)
                render_alert_dashboard(alert_panel, st.session_state.alert_entries)

            labels = result.boxes.cls.tolist() if result.boxes is not None else []
            if is_warmup:
                status_text.warning(
                    f"⏳ Camera warming up: alert monitoring starts in {remaining_warmup}s... "
                    f"(Detected {len(labels)} object(s))"
                )
            elif not labels:
                status_text.warning("No PPE objects detected in current frame.")
            else:
                status_text.info(f"Detected {len(labels)} object(s)")

        render_manual_ppe_camera()

    if not run:
        camera_alerted_person_ids.clear()
        camera_person_tracking_state["detection_counts"].clear()
        camera_person_tracking_state["last_seen"].clear()
        camera_person_tracker.reset()

    if run and available_cameras and not manual_ppe_camera_running:
        active_alert_items = None
        camera_start_time = time.time()
        results_stream = None
        try:
            camera_probe = cv2.VideoCapture(camera_index)
            try:
                camera_opened, camera_frame = camera_probe.read()
            finally:
                camera_probe.release()
            if not camera_opened or camera_frame is None or camera_frame.size == 0:
                raise ConnectionError(
                    f"Camera {camera_index} opened but did not provide a readable frame."
                )

            if is_ppe_model:
                results_stream = model.predict(
                    source=camera_index,
                    conf=min(conf_threshold, 0.10),
                    classes=tracking_classes,
                    stream=True,
                    verbose=False,
                )
            else:
                results_stream = model.predict(
                    source=camera_index,
                    conf=conf_threshold,
                    stream=True,
                    verbose=False,
                )

            for result in results_stream:
                if st.session_state.rfid_session_deadline is not None and time.monotonic() >= st.session_state.rfid_session_deadline:
                    st.session_state.rfid_session_started_at = None
                    st.session_state.rfid_session_deadline = None
                    if rfid_monitor is not None:
                        rfid_monitor.status.matched = False
                    st.sidebar.info("Camera closed after 15 seconds. Waiting for the next RFID tag.")
                    st.rerun()
                    break

                elapsed_time = time.time() - camera_start_time
                is_warmup = elapsed_time < camera_alert_delay
                remaining_warmup = max(0, int(camera_alert_delay - elapsed_time) + 1)

                if is_ppe_model:
                    filter_result_by_confidence(result, conf_threshold, model_names)
                    camera_person_tracker.update(result, result.orig_img, model_names)
                annotated = plot_detections(result, model_names, is_ppe_model)
                if is_warmup:
                    cv2.putText(
                        annotated,
                        f"Camera Warmup: Alert starts in {remaining_warmup}s",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                FRAME_WINDOW.image(annotated, channels="BGR")
                frame_alerts, frame_present = alert_items_for_result(
                    result, model_names, is_ppe_model
                )
                if is_ppe_model:
                    new_alert_created = False
                    update_person_tracking_state(
                        result, model_names, camera_person_tracking_state, camera_alerted_person_ids
                    )
                    if not is_warmup:
                        for (
                            person_id,
                            missing_items,
                            present_items,
                            has_explicit_missing_ppe,
                        ) in ppe_alerts_by_person(
                            result, model_names, include_compliant=True
                        ):
                            if person_id in camera_alerted_person_ids:
                                continue
                            if (
                                camera_person_tracking_state["detection_counts"].get(person_id, 0) < 10
                                and not has_explicit_missing_ppe
                            ):
                                continue
                            crop_snap = None
                            if result.boxes is not None and result.boxes.id is not None:
                                for b, tid in zip(result.boxes.xyxy.tolist(), result.boxes.id.tolist()):
                                    if int(tid) == person_id:
                                        h_img, w_img = annotated.shape[:2]
                                        pad = 20
                                        cx1 = max(0, int(b[0]) - pad)
                                        cy1 = max(0, int(b[1]) - pad)
                                        cx2 = min(w_img, int(b[2]) + pad)
                                        cy2 = min(h_img, int(b[3]) + pad)
                                        if cx2 > cx1 and cy2 > cy1:
                                            crop_snap = annotated[cy1:cy2, cx1:cx2].copy()
                            entry = create_alert_entry(
                                annotated,
                                missing_items,
                                present_items,
                                missing_items,
                                "live_camera",
                                person_id=person_id,
                                crop_image=crop_snap,
                                visitor_message=(
                                    "Go Back and come after 30min" if missing_items else "Welcome"
                                ),
                            )
                            append_alert_entry(st.session_state.alert_entries, entry)
                            camera_alerted_person_ids.add(person_id)
                            new_alert_created = True
                        if new_alert_created:
                            render_alert_dashboard(alert_panel, st.session_state.alert_entries)
                else:
                    if not is_warmup:
                        previous_items = set(active_alert_items or ())
                        new_items = frame_alerts - previous_items
                        active_alert_items = tuple(sorted(frame_alerts))
                        detected_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
                        for hazard in ("fire", "smoke"):
                            if hazard in new_items:
                                entry = create_alert_entry(
                                    annotated,
                                    set(),
                                    set(),
                                    {hazard},
                                    "live_camera",
                                    captured_at=detected_at,
                                )
                                append_alert_entry(st.session_state.alert_entries, entry)
                                render_alert_dashboard(alert_panel, st.session_state.alert_entries)

                labels = result.boxes.cls.tolist() if result.boxes is not None else []
                if is_warmup:
                    status_text.warning(
                        f"⏳ Camera warming up: alert monitoring starts in {remaining_warmup}s... "
                        f"(Detected {len(labels)} object(s))"
                    )
                elif not labels:
                    status_text.warning(f"No {st.session_state.selected_detector} objects detected in current frame.")
                else:
                    danger_count = len(frame_alerts)
                    status_text.info(
                        f"Detected {len(labels)} object(s)"
                        f"{' – danger: ' + str(danger_count) if danger_count else ''}"
                    )
        except ConnectionError as error:
            status_text.error(
                f"Could not start Camera {camera_index}: {error} "
                "Check that the camera is connected, not in use by another app, "
                "and select Refresh camera list to try again."
            )
            if is_ppe_model and rfid_monitor is not None:
                rfid_monitor.status.matched = False
                st.session_state.rfid_session_started_at = None
                st.session_state.rfid_session_deadline = None
            if not is_ppe_model:
                st.session_state.fire_smoke_camera_running = False
        finally:
            if results_stream is not None:
                results_stream.close()
    elif not run:
        if requires_rfid:
            st.write("Webcam locked. Scan a matched RFID card first.")
        else:
            st.write("Camera is stopped. Click Start Camera in the sidebar to begin detection.")

