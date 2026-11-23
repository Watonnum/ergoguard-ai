# ErgoGuard AI — Project Specification & Agent Prompt for Antigravity IDE

> **วิธีใช้งานไฟล์นี้**: เซฟไฟล์นี้เป็น `SPEC.md` ไว้ใน Root Folder ของโปรเจกต์ แล้วพิมพ์สั่ง Model ใน Antigravity IDE ว่า:
> *"Read `SPEC.md` carefully. Let's build the ErgoGuard AI project step-by-step starting from Phase 1 (project setup & dependencies) through Phase 4 (Streamlit UI). Verify each module before moving to the next."*

## 1. Project Overview
- **Project Name**: **ErgoGuard AI** (ระบบตรวจจับท่านั่งทำงานและป้องกันออฟฟิศซินโดรมอัจฉริยะ)
- **Course Context**: CSC472 Artificial Intelligence (Project Demo)
- **Target Environment**: macOS (Apple Silicon M3) / Python 3.10–3.12 (`venv`)
- **Core Objective**: Build a real-time webcam-based ergonomic posture monitoring web application that extracts upper-body joint angles, classifies posture using **K-Nearest Neighbors (K-NN)**, and triggers posture alerts and stretch recommendations using a **Rule-Based Expert System (Forward Chaining / IF-THEN)**.

## 2. Tech Stack & Dependencies
Use a clean, lightweight Python stack that runs reliably on macOS webcam without external paid APIs:
- **UI & Dashboard**: `streamlit`
- **Computer Vision & Pose**: `opencv-python`, `mediapipe`
- **Machine Learning (CSC472)**: `scikit-learn` (`KNeighborsClassifier`, `StandardScaler`), `numpy`, `pandas`
- **Visualization**: `plotly` or Streamlit native charts (`st.bar_chart` / `st.line_chart`)

## 3. Directory & Module Architecture
Scaffold the project with this exact modular structure:

```text
ergoguard-ai/
├── SPEC.md
├── requirements.txt
├── data/
│   └── posture_dataset.csv        # Seed dataset + user-calibrated samples
├── src/
│   ├── __init__.py
│   ├── pose_tracker.py            # Module 1: Webcam + MediaPipe Pose + Angle Math
│   ├── knn_classifier.py          # Module 2: K-NN Posture Classifier (Euclidean Distance)
│   ├── rule_engine.py             # Module 3: Rule-Based Expert System + 15s Time Window + Ergo Score
│   └── data_generator.py          # Helper: Generates initial labeled posture_dataset.csv
└── app.py                         # Module 4: Streamlit Real-Time Dashboard & Live Webcam Loop
```

## 4. Detailed Module Specifications

### Module 1: `src/pose_tracker.py` (Pose Estimation & Feature Extraction)
- **Responsibilities**:
  - Initialize MediaPipe Pose (`mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)`).
  - Detect upper-body landmarks from each BGR frame (mirrored horizontally for natural webcam UX).
  - Support both **side-profile** and **front/three-quarter** webcam angles (use the side with higher landmark `visibility`, i.e., `LEFT_EAR`/`LEFT_SHOULDER`/`LEFT_HIP` vs `RIGHT_EAR`/`RIGHT_SHOULDER`/`RIGHT_HIP`).
- **Key Metrics to Compute**:
  1. **Neck Inclination Angle ($\theta_{\text{neck}}$)**: Angle in degrees between the Ear–Shoulder vector and the vertical axis (0° = ear directly above shoulder; > 20° = forward head / tech neck).
  2. **Torso Inclination Angle ($\theta_{\text{torso}}$)**: Angle in degrees between the Hip–Shoulder vector and the vertical axis (or interior Ear–Shoulder–Hip angle; ~90°–100° upright vs < 80° slouching/leaning forward). Normalize clearly so the UI displays intuitive degrees: e.g., `neck_angle` (0°–60°, normal `< 20°`) and `torso_angle` (0°–180° where ~90°–100° is upright, or vertical tilt `0°–45°`). Keep the convention consistent across `data_generator.py`, `knn_classifier.py`, and `rule_engine.py`.
  3. **Shoulder Alignment Delta ($\Delta y_{\text{shoulder}}$)**: Normalized vertical difference between left and right shoulders to detect uneven leaning.
- **Visual Overlay**:
  - Draw clean cyan/green/red skeletal lines connecting Ear → Shoulder → Hip and landmark dots directly on the video frame.
  - Overlay live angle values (`Neck: XX deg`, `Torso: XX deg`) on the top-left of the frame.
- **Edge Case**: If landmarks are not visible (user stepped away), return `detected=False` cleanly without crashing.

### Module 2: `src/knn_classifier.py` & `src/data_generator.py` (K-NN Classifier)
- **Algorithm**: `KNeighborsClassifier(n_neighbors=5, metric='euclidean')` wrapped in a `Pipeline` with `StandardScaler()`.
- **Classes (3 Labels)**:
  - `Good Posture` (ท่านั่งปกติ)
  - `Forward Head` (อาการคอยื่น / Tech Neck)
  - `Slouching` (อาการนั่งหลังค่อม / โน้มตัว)
- **Seed Dataset (`data/posture_dataset.csv`)**:
  - Columns: `neck_angle, torso_angle, shoulder_tilt, label`
  - If `data/posture_dataset.csv` does not exist on startup, `data_generator.py` generates ~300 realistic synthetic samples (100 per class) with slight Gaussian noise around ergonomic thresholds:
    - `Good Posture`: `neck_angle` ~ 5°–18°, `torso_angle` ~ 88°–102°
    - `Forward Head`: `neck_angle` ~ 25°–48°, `torso_angle` ~ 82°–95°
    - `Slouching`: `neck_angle` ~ 20°–40°, `torso_angle` ~ 60°–78°
- **Methods**:
  - `train(csv_path)` -> fits scaler + K-NN model.
  - `predict(neck_angle, torso_angle, shoulder_tilt)` -> returns `(predicted_label, confidence_score, nearest_distances)`.
  - `add_calibration_sample(neck_angle, torso_angle, shoulder_tilt, label)` -> appends live user sample to CSV and retrains immediately (great for live demo calibration!).

### Module 3: `src/rule_engine.py` (Rule-Based Expert System & Temporal Smoothing)
- **Responsibilities**:
  - Implement a **15-second Sliding Time Window** (configurable in UI sidebar, e.g., 5s–30s for easy live demo testing!) so brief movements (grabbing a pen, glancing down) do not trigger false alarms.
  - Implement **Forward Chaining IF-THEN Rules** evaluating:
    1. `knn_label`
    2. `neck_angle`, `torso_angle`
    3. `bad_posture_duration_sec` (continuous seconds in non-good posture)
- **Explicit Rules to Implement**:
  - **Rule R1 (Good Posture)**: `IF knn_label == 'Good Posture' AND neck_angle < 20`: Status = `PASS` (ปกติ), Reset alert timer, No stretch needed.
  - **Rule R2 (Mild Drift Warning)**: `IF knn_label != 'Good Posture' AND bad_duration < threshold_sec`: Status = `MONITORING` (กำลังเฝ้าระวัง), Show countdown timer before alert.
  - **Rule R3 (Forward Head Alert)**: `IF (knn_label == 'Forward Head' OR neck_angle >= 30) AND bad_duration >= threshold_sec`: Status = `ALERT`, Trigger warning + Recommend **Chin Tuck & Upper Trapezius Stretch (30s)**.
  - **Rule R4 (Slouching Alert)**: `IF (knn_label == 'Slouching' OR torso_angle <= 75) AND bad_duration >= threshold_sec`: Status = `ALERT`, Trigger warning + Recommend **Thoracic Extension & Chest Opener (30s)**.
- **Ergo Score (0–100)**:
  - Compute a smooth real-time score based on current angle deviation + rolling session good-posture percentage (e.g., ~90–100 when upright, dropping toward ~40–50 during sustained Forward Head/Slouching).
- **Output Object**: Return a structured dictionary/dataclass containing `status` (`PASS` | `MONITORING` | `ALERT`), `ergo_score`, `triggered_rule_id`, `alert_message_th`, `stretch_recommendation_th`, and `bad_duration_sec`.

### Module 4: `app.py` (Streamlit Real-Time UI)
- **Theme & Layout**: Clean dark-slate aesthetic matching the ErgoGuard AI proposal.
- **Sidebar Controls**:
  - Camera Index selector (`0` default for MacBook FaceTime HD Camera)
  - Start / Stop Monitoring toggle
  - Alert Time Window slider (`5` to `30` seconds, default `15s` — setting it to `5s` makes live classroom demos super fast!)
  - `K` value slider for K-NN (`3`, `5`, `7`)
  - **Quick Calibration**: Buttons to capture current posture as `Good Posture` sample and retrain K-NN on the fly.
- **Main Area**:
  - **Left Column (60%)**: Live webcam feed with skeletal overlay + status pill (`PASS` green / `ALERT` red).
  - **Right Column (40%)**:
    - Big **Ergo Score** metric card (`XX / 100`)
    - Live **Neck Angle** & **Torso Angle** metrics with delta indicators
    - **K-NN Classification** & **Triggered Rule-Based Advice** card (shows exact Thai recommendation & stretch guide when alerted)
  - **Bottom Section**: Session Analytics (bar/line chart of posture distribution: Good vs. Forward Head vs. Slouching over time).

## 5. Build & Verification Plan for Antigravity Agent
Execute in order and verify each step:
1. **Phase 1 — Setup**: Create `requirements.txt`, virtual environment, install dependencies, and generate `data/posture_dataset.csv` via `src/data_generator.py`.
2. **Phase 2 — Core Logic & Unit Tests**: Implement `pose_tracker.py`, `knn_classifier.py`, and `rule_engine.py`. Run a quick Python test script verifying angle math, K-NN accuracy on the dataset, and Rule R1–R4 transitions (including the 15s time window).
3. **Phase 3 — Streamlit UI**: Implement `app.py` and run `streamlit run app.py` to verify there are zero import/runtime errors and camera permissions work cleanly on macOS.
4. **Phase 4 — Demo Polish**: Ensure all Thai labels render cleanly and add a "Demo Simulation Mode" toggle (optional fallback using test angles/video if a camera isn't attached during testing).
