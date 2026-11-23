"""
Automated Unit & Integration Tests for ErgoGuard AI (CSC472 Project).
Verifies Phase 1 (Dataset Generation) and Phase 2 (Pose Math, K-NN Classifier,
and Rule-Based Expert System R1-R4 transitions).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_generator import POSTURE_CLASSES, ensure_dataset_exists, generate_seed_dataframe
from src.knn_classifier import PostureKNNClassifier
from src.pose_tracker import (
    PoseTracker,
    calculate_neck_angle_2d,
    calculate_shoulder_tilt,
    calculate_torso_angle_2d,
    detect_is_side_profile,
)
from src.rule_engine import ErgonomicRuleEngine


def test_data_generator_creates_balanced_dataset(tmp_path: Path) -> None:
    csv_file = tmp_path / "posture_dataset.csv"
    saved_path = ensure_dataset_exists(csv_file, samples_per_class=100, force_regenerate=True)
    assert saved_path.exists()

    df = pd.read_csv(saved_path)
    assert len(df) == 300
    assert list(df.columns) == ["neck_angle", "torso_angle", "shoulder_tilt", "label"]
    for cls in POSTURE_CLASSES:
        assert (df["label"] == cls).sum() >= 60


def test_pose_angle_math_and_auto_fallback() -> None:
    # 1. Ear directly above shoulder -> ~0 deg neck inclination
    neck_upright = calculate_neck_angle_2d(ear_xy=(0.50, 0.30), shoulder_xy=(0.50, 0.50))
    assert 0.0 <= neck_upright <= 2.0

    # 2. Ear significantly forward of shoulder -> > 25 deg neck inclination (Forward Head)
    neck_forward = calculate_neck_angle_2d(ear_xy=(0.64, 0.32), shoulder_xy=(0.50, 0.50))
    assert neck_forward >= 30.0

    # 3. Shoulder directly above hip -> ~95 deg upright torso angle
    torso_upright = calculate_torso_angle_2d(shoulder_xy=(0.50, 0.45), hip_xy=(0.50, 0.80))
    assert 93.0 <= torso_upright <= 96.0

    # 4. Shoulder leaning forward relative to hip -> < 75 deg slouching torso angle
    torso_slouch = calculate_torso_angle_2d(shoulder_xy=(0.68, 0.50), hip_xy=(0.50, 0.80))
    assert torso_slouch <= 70.0

    # 5. Shoulder alignment delta
    tilt = calculate_shoulder_tilt(left_shoulder_y=0.50, right_shoulder_y=0.52, shoulder_span=0.30)
    assert 0.015 <= tilt <= 0.025

    # 6. Front-webcam auto-fallback when hips are occluded
    f_neck_good, f_torso_good = PoseTracker._compute_front_fallback_angles(
        ear_mid_y=0.32,
        sh_mid_y=0.52,
        shoulder_span=0.34,
        nose_z=-0.20,
        sh_mid_z=-0.05,
        hip_mid_xy=None,
        sh_mid_xy=None,
    )
    assert f_neck_good < 18.0
    assert f_torso_good >= 88.0

    f_neck_slouch, f_torso_slouch = PoseTracker._compute_front_fallback_angles(
        ear_mid_y=0.63,
        sh_mid_y=0.75,
        shoulder_span=0.34,
        nose_z=-0.55,
        sh_mid_z=-0.05,
        hip_mid_xy=None,
        sh_mid_xy=None,
    )
    assert f_neck_slouch >= 25.0
    assert f_torso_slouch <= 78.0


def test_pose_tracker_simulation_and_blank_frame() -> None:
    tracker = PoseTracker()
    assert tracker.backend in ("TASKS", "SOLUTIONS")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, metrics = tracker.process_frame(blank)
    assert annotated.shape == (480, 640, 3)
    assert metrics.detected is False

    sim_frame, sim_metrics = tracker.render_simulation_frame(
        neck_angle=12.0,
        torso_angle=94.0,
        shoulder_tilt=0.01,
        status="PASS",
    )
    assert sim_frame.shape == (460, 720, 3)
    assert sim_metrics.detected is True
    assert sim_metrics.neck_angle == 12.0
    tracker.close()


def test_knn_classifier_predictions_and_calibration(tmp_path: Path) -> None:
    csv_file = tmp_path / "posture_dataset.csv"
    clf = PostureKNNClassifier(csv_path=csv_file, n_neighbors=5)
    assert clf.cv_accuracy >= 0.92

    # Test Good Posture prediction
    label_good, conf_good, dists_good = clf.predict(11.0, 95.0, 0.01)
    assert label_good == "Good Posture"
    assert conf_good >= 0.8
    assert len(dists_good) == 5

    # Test Forward Head prediction
    label_fh, conf_fh, _ = clf.predict(38.0, 89.0, 0.02)
    assert label_fh == "Forward Head"
    assert conf_fh >= 0.8

    # Test Slouching prediction
    label_slouch, conf_slouch, _ = clf.predict(30.0, 68.0, 0.04)
    assert label_slouch == "Slouching"
    assert conf_slouch >= 0.8

    # Test Explainable AI neighbor details
    neighbors = clf.explain_neighbors(38.0, 89.0, 0.02)
    assert len(neighbors) == 5
    assert neighbors[0]["rank"] == 1
    assert "euclidean_distance" in neighbors[0]

    # Test live calibration sample addition
    new_count = clf.add_calibration_sample(14.0, 92.0, 0.015, "Good Posture", replicas=1)
    assert new_count == 301

    reset_count = clf.reset_to_seed_dataset()
    assert reset_count == 300


def test_rule_engine_r1_to_r4_transitions_and_15s_window() -> None:
    engine = ErgonomicRuleEngine(threshold_sec=15.0)

    # 1. Rule R1: Good Posture (< 20 deg neck) -> PASS
    res_r1 = engine.evaluate(
        knn_label="Good Posture",
        neck_angle=12.0,
        torso_angle=94.0,
        current_time=100.0,
    )
    assert res_r1.status == "PASS"
    assert res_r1.triggered_rule_id == "R1"
    assert res_r1.bad_duration_sec == 0.0
    assert res_r1.ergo_score >= 88.0

    # 2. Rule R2: Forward Head starts at t=105.0 (0s elapsed) and t=112.0 (7s < 15s) -> MONITORING
    engine.evaluate(
        knn_label="Forward Head",
        neck_angle=34.0,
        torso_angle=88.0,
        current_time=105.0,
    )
    res_r2 = engine.evaluate(
        knn_label="Forward Head",
        neck_angle=34.0,
        torso_angle=88.0,
        current_time=112.0,
    )
    assert res_r2.status == "MONITORING"
    assert res_r2.triggered_rule_id == "R2"
    assert res_r2.bad_duration_sec == pytest.approx(7.0, abs=0.1)
    assert res_r2.countdown_sec == pytest.approx(8.0, abs=0.1)

    # 3. Rule R3: Forward Head sustained at t=121.0 (16s >= 15s) -> ALERT + Chin Tuck
    res_r3 = engine.evaluate(
        knn_label="Forward Head",
        neck_angle=35.0,
        torso_angle=88.0,
        current_time=121.0,
    )
    assert res_r3.status == "ALERT"
    assert res_r3.triggered_rule_id == "R3"
    assert "Chin Tuck" in res_r3.stretch_recommendation_th
    assert res_r3.ergo_score <= 65.0

    # 4. Return to Good Posture at t=125.0 -> Timer resets, Rule R1 fires
    res_reset = engine.evaluate(
        knn_label="Good Posture",
        neck_angle=11.0,
        torso_angle=95.0,
        current_time=125.0,
    )
    assert res_reset.status == "PASS"
    assert res_reset.triggered_rule_id == "R1"
    assert res_reset.bad_duration_sec == 0.0

    # 5. Rule R4: Slouching sustained for >= 15s -> ALERT + Thoracic Extension
    engine.evaluate(
        knn_label="Slouching",
        neck_angle=28.0,
        torso_angle=68.0,
        current_time=200.0,
    )
    res_r4 = engine.evaluate(
        knn_label="Slouching",
        neck_angle=28.0,
        torso_angle=68.0,
        current_time=216.0,
    )
    assert res_r4.status == "ALERT"
    assert res_r4.triggered_rule_id == "R4"
    assert "Thoracic Extension" in res_r4.stretch_recommendation_th


def test_end_to_end_pipeline_and_app_syntax() -> None:
    """Verify full PoseTracker -> KNNClassifier -> RuleEngine pipeline and app.py syntax."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")
    compile(source, str(app_path), "exec")

    tracker = PoseTracker()
    clf = PostureKNNClassifier()
    engine = ErgonomicRuleEngine(threshold_sec=5.0)

    sim_frame, metrics = tracker.render_simulation_frame(36.0, 88.0, 0.02, status="ALERT")
    pred_label, conf, dists = clf.predict(metrics.neck_angle, metrics.torso_angle, metrics.shoulder_tilt)
    res = engine.evaluate(pred_label, metrics.neck_angle, metrics.torso_angle, override_bad_duration_sec=6.0)

    assert sim_frame.shape == (460, 720, 3)
    assert pred_label == "Forward Head"
    assert conf > 0.7
    assert len(dists) == 5
    assert res.status == "ALERT"
    assert res.triggered_rule_id == "R3"
    tracker.close()


def test_shoulder_tilt_does_not_hijack_good_posture_into_slouching() -> None:
    """Regression test: upright neck (11 deg) and torso (95 deg) must remain Good Posture even if shoulder_tilt is high."""
    clf = PostureKNNClassifier()
    label, conf, _ = clf.predict(11.0, 95.0, 0.185)
    assert label == "Good Posture"
    assert conf >= 0.8

    # Side-profile shoulder span (< 0.24) must damp perspective shoulder tilt below 0.035
    damped_tilt = calculate_shoulder_tilt(0.58, 0.654, shoulder_span=0.12)
    assert damped_tilt <= 0.030


def test_monitoring_ergo_score_stays_in_amber_band_and_combined_slouch_neck_warning() -> None:
    """Verify Ergo Score stays in 75-88 during MONITORING and drops < 60 only on ALERT."""
    engine = ErgonomicRuleEngine(threshold_sec=15.0)
    engine.evaluate("Slouching", neck_angle=35.0, torso_angle=70.0, current_time=10.0)
    res_mon = engine.evaluate("Slouching", neck_angle=35.0, torso_angle=70.0, current_time=17.8)
    assert res_mon.status == "MONITORING"
    assert 75.0 <= res_mon.ergo_score <= 88.0
    assert "นั่งหลังค่อม/โน้มตัว และก้มคอพร้อมกัน" in res_mon.alert_message_th

    res_alert = engine.evaluate("Slouching", neck_angle=35.0, torso_angle=70.0, current_time=26.0)
    assert res_alert.status == "ALERT"
    assert res_alert.triggered_rule_id == "R4"
    assert res_alert.ergo_score < 60.0
    assert "Slouching + Forward Head" in res_alert.alert_message_th


def test_nose_between_shoulders_and_personal_baseline_normalization() -> None:
    """Verify front-facing users never flip to side-profile due to narrow shoulders, and personal baseline normalizes body types."""
    # 1. Slender person facing front (shoulder span = 0.18, nose centered at 0.50) -> NOT side profile
    assert detect_is_side_profile(l_sh_x=0.59, r_sh_x=0.41, nose_x=0.50, l_sh_vis=0.9, r_sh_vis=0.9) is False

    # 2. Person turned sideways (shoulders stacked or nose outside shoulder span) -> IS side profile
    assert detect_is_side_profile(l_sh_x=0.52, r_sh_x=0.46, nose_x=0.42, l_sh_vis=0.9, r_sh_vis=0.5) is True

    # 3. Hysteresis prevents single-frame mode flicker
    tracker = PoseTracker()
    assert tracker.model_name == "pose_landmarker_heavy.task"
    assert tracker._resolve_view_mode_with_hysteresis(False) is False
    for _ in range(4):
        # 4 glitch frames (< 6 threshold) must NOT flip mode to side profile
        assert tracker._resolve_view_mode_with_hysteresis(True) is False

    # 4. Personal Baseline Normalization for different body types (e.g. heavier build / high camera)
    tracker.set_personal_baseline(raw_neck_angle=22.0, raw_torso_angle=85.0, raw_shoulder_tilt=0.03)
    norm_n, norm_t, norm_s = tracker._apply_personal_baseline(22.0, 85.0, 0.03)
    assert norm_n == pytest.approx(11.5, abs=0.1)
    assert norm_t == pytest.approx(94.5, abs=0.1)

    # When that same user bends neck +20 deg and slouches torso -22 deg from their baseline:
    slouch_n, slouch_t, _ = tracker._apply_personal_baseline(42.0, 63.0, 0.03)
    assert slouch_n == pytest.approx(31.5, abs=0.1)
    assert slouch_t == pytest.approx(72.5, abs=0.1)
    tracker.clear_personal_baseline()

    # 5. Standby frame renders cleanly without text clutter
    standby = tracker.render_camera_standby_frame()
    assert standby.shape == (540, 960, 3)

    # 6. Lateral head tilt ("เอียงคอ") in front view increases neck_angle >= 30 deg
    tilt_neck, _ = PoseTracker._compute_front_fallback_angles(
        ear_mid_y=0.34,
        sh_mid_y=0.54,
        shoulder_span=0.34,
        nose_z=-0.20,
        sh_mid_z=-0.05,
        hip_mid_xy=None,
        sh_mid_xy=None,
        ear_roll_deg=22.0,
        lateral_neck_deg=14.0,
    )
    assert tilt_neck >= 35.0







