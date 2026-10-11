# Safety Detection

A Streamlit application for PPE kit or fire and smoke detection using the corresponding YOLO model.

![PPE Detection Demo](image.png)

# Dataset and annotations

![PPE Dataset](ds.avif)
## Prerequisites

- Python (>=3.8)
- `best.pt` for PPE kit detection.
- `best_smoke_fire.pt` for fire and smoke detection.

```bash
pip install -r requirements.txt
```

## Running the App

1. Place `best.pt` and `best_smoke_fire.pt` in the project directory.
2. Start the Streamlit server:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

3. Open our browser to `http://localhost:8501` (Streamlit will display the URL).

## Detection options

- Choose **PPE Kit** or **Fire and Smoke** to select the detection model.
- **Camera**: run live detection using a camera connected to the computer running Streamlit.
- **Video**: upload a video and download the annotated output.
- **Web**: access a camera connected to the device opening the app, through the browser. Click **Start**, allow browser camera access, then click **Select Device** to choose the external camera once; the browser remembers it as the default for Web detection.
- Adjust the confidence threshold from the sidebar.

## Alerts and Firebase

The **Alert Dashboard** records PPE-missing alerts and smoke/fire alerts from live camera, browser camera, and uploaded-video detection. Each new alert episode includes a snapshot. When Firebase is configured, the app writes the event and JPEG snapshot to the Realtime Database `ppe_alerts` node and sends a Firebase Cloud Messaging notification to the matching `ppe_safety_alerts` or `fire_safety_alerts` topic.

Install dependencies from `requirements.txt`, including `firebase-admin`. The app prefers `minniepro-e2b29-firebase-adminsdk-fbsvc-791c9195cb.json` when it is present, or you can set `FIREBASE_CREDENTIALS` to another service-account path. The database URL defaults to `https://minniepro-e2b29-default-rtdb.firebaseio.com`; override it with `FIREBASE_DATABASE_URL` if our Firebase Realtime Database uses a different URL. Without valid credentials, alerts still appear in the dashboard, but are not saved or pushed to Firebase.

## Notes

- In live mode, the camera must be connected to the computer running Streamlit. The app detects available OpenCV camera indexes and prefers index 1 when available; select another detected camera from the sidebar or refresh the list after connecting a device.
- In Web mode, use **Select Device** on the camera panel to choose among cameras detected by the browser. Camera permission may need to be granted before device names appear. Remote deployments need HTTPS for browser camera access.
- Helmet detections are retained only when the helmet box center falls within the top 22% of a detected person box. This spatial check suppresses helmets being held away from the head, but is an approximation because the model does not identify head pose or whether a helmet is being worn.
- The PPE model is `best.pt`; the fire and smoke model is `best_smoke_fire.pt`.
