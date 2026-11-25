"""
Module 1: src/pose_tracker.py
High-Accuracy Webcam Pose Estimation (MediaPipe PoseLandmarker Heavy),
Body-Size-Invariant Ergonomic Angle Math, Personal Baseline Normalization,
Nose-Between-Shoulders Auto View Detection (with Hysteresis), and Labeled
Anatomical Skeletal Overlay for ErgoGuard AI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any
import urllib.request
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None

HEAVY_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
LITE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEAVY_MODEL_PATH = DATA_DIR / "pose_landmarker_heavy.task"
LITE_MODEL_PATH = DATA_DIR / "pose_landmarker_lite.task"
BASELINE_JSON_PATH = DATA_DIR / "personal_baseline.json"
KAGGLE_REGRESSOR_PATH = DATA_DIR / "kaggle_spine_regressor.joblib"

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

# Standard MediaPipe BlazePose landmark indices
IDX_NOSE = 0
IDX_LEFT_EAR = 7
IDX_RIGHT_EAR = 8
IDX_LEFT_SHOULDER = 11
IDX_RIGHT_SHOULDER = 12
IDX_LEFT_HIP = 23
IDX_RIGHT_HIP = 24

# Ergonomic standard reference values (RULA/REBA upright neutral baseline)
STANDARD_UPRIGHT_NECK_DEG = 11.5
STANDARD_UPRIGHT_TORSO_DEG = 94.5

# Module-level singleton to prevent MediaPipe LiteRT C++ accelerator deadlocks on re-init
_SHARED_LANDMARKER: Any = None
_SHARED_SOLUTIONS_POSE: Any = None
_SHARED_BACKEND: str = "NONE"
_SHARED_MODEL_NAME: str = "NONE"


def _resolve_best_model_path(preferred_path: Path | None = None) -> Path:
    """Select pose_landmarker_heavy.task if available, falling back to lite."""
    if preferred_path is not None and preferred_path.exists() and preferred_path.stat().st_size > 1_000_000:
        return preferred_path
    if HEAVY_MODEL_PATH.exists() and HEAVY_MODEL_PATH.stat().st_size > 5_000_000:
        return HEAVY_MODEL_PATH
    if LITE_MODEL_PATH.exists() and LITE_MODEL_PATH.stat().st_size > 1_000_000:
        return LITE_MODEL_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LITE_MODEL_URL, str(LITE_MODEL_PATH))
    return LITE_MODEL_PATH


def _get_shared_mediapipe_backend(
    model_path: Path | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[str, Any, Any, str]:
    global _SHARED_LANDMARKER, _SHARED_SOLUTIONS_POSE, _SHARED_BACKEND, _SHARED_MODEL_NAME
    resolved_path = _resolve_best_model_path(model_path)

    # If already initialized with the same or better model, reuse singleton
    if _SHARED_BACKEND != "NONE" and _SHARED_MODEL_NAME == resolved_path.name:
        return _SHARED_BACKEND, _SHARED_SOLUTIONS_POSE, _SHARED_LANDMARKER, _SHARED_MODEL_NAME

    if mp is None:
        return "NONE", None, None, "NONE"

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        _SHARED_SOLUTIONS_POSE = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        _SHARED_BACKEND = "SOLUTIONS"
        _SHARED_MODEL_NAME = "solutions_pose_heavy"
        return _SHARED_BACKEND, _SHARED_SOLUTIONS_POSE, None, _SHARED_MODEL_NAME

    if hasattr(mp, "tasks"):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_python.BaseOptions(model_asset_path=str(resolved_path))
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            _SHARED_LANDMARKER = mp_vision.PoseLandmarker.create_from_options(options)
            _SHARED_BACKEND = "TASKS"
            _SHARED_MODEL_NAME = resolved_path.name
            return _SHARED_BACKEND, None, _SHARED_LANDMARKER, _SHARED_MODEL_NAME
        except Exception:
            _SHARED_BACKEND = "NONE"

    return "NONE", None, None, "NONE"


@dataclass
class PoseMetrics:
    """Extracted posture features and metadata for a single video frame."""

    detected: bool
    neck_angle: float = 0.0
    torso_angle: float = 90.0
    active_side: str = "NONE"       # 'LEFT' | 'RIGHT' | 'BILATERAL' | 'SIMULATED'
    view_mode: str = "NONE"         # 'SIDE_PROFILE' | 'FRONT_HYBRID' | 'FRONT_FALLBACK' | 'SIMULATED'
    visibility_score: float = 0.0
    raw_neck_angle: float = 0.0
    raw_torso_angle: float = 90.0
    baseline_calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def calculate_neck_angle_2d(ear_xy: tuple[float, float], shoulder_xy: tuple[float, float]) -> float:
    """
    Compute Neck Inclination Angle (theta_neck in degrees, 0° - 65°)
    between the Shoulder -> Ear vector and the upward vertical axis (0, -1).
    0° = ear directly above shoulder; >= 20° = forward head / tech neck.
    """
    dx = float(ear_xy[0] - shoulder_xy[0])
    dy = float(shoulder_xy[1] - ear_xy[1])  # upward in image coordinates
    dy_safe = max(1e-5, dy) if dy >= 0 else dy
    angle_rad = math.atan2(abs(dx), max(1e-5, abs(dy_safe)))
    angle_deg = math.degrees(angle_rad)
    return round(float(np.clip(angle_deg, 0.0, 65.0)), 2)


def calculate_torso_angle_2d(
    shoulder_xy: tuple[float, float],
    hip_xy: tuple[float, float],
) -> float:
    """
    Compute Torso Inclination Angle (theta_torso in degrees) normalized so that:
    ~90° - 100° = sitting upright (vertical spine)
    < 80° (e.g. 60° - 78°) = slouching / leaning forward.
    """
    dx = float(shoulder_xy[0] - hip_xy[0])
    dy = float(hip_xy[1] - shoulder_xy[1])  # positive when shoulder is above hip
    tilt_rad = math.atan2(abs(dx), max(1e-5, abs(dy)))
    tilt_deg = math.degrees(tilt_rad)
    torso_angle = 95.0 - tilt_deg
    return round(float(np.clip(torso_angle, 45.0, 110.0)), 2)


def detect_is_side_profile(
    l_sh_x: float,
    r_sh_x: float,
    nose_x: float,
    l_sh_vis: float = 1.0,
    r_sh_vis: float = 1.0,
) -> bool:
    """
    Determine whether the user is in Side-Profile vs. Front-View using
    Nose-Between-Shoulders horizontal geometry (immune to hair covering one ear!).

    Rules:
    1. Compute horizontal shoulder span `raw_span = abs(l_sh_x - r_sh_x)`.
       If `raw_span < 0.13`, the shoulders are stacked front-to-back -> SIDE_PROFILE.
    2. Check if the Nose (`nose_x`) lies horizontally inside the central 76% of the
       interval `[min(l_sh_x, r_sh_x), max(l_sh_x, r_sh_x)]`.
       - If YES (`0.12 <= rel_nose <= 0.88`) and both shoulders are reasonably visible
         (`min(l_sh_vis, r_sh_vis) >= 0.35`), the user is facing the camera -> FRONT_VIEW (`False`).
       - If NO (the nose is near or outside one shoulder edge), the user is turned
         sideways -> SIDE_PROFILE (`True`).
    """
    sh_min = min(float(l_sh_x), float(r_sh_x))
    sh_max = max(float(l_sh_x), float(r_sh_x))
    raw_span = sh_max - sh_min

    if raw_span < 0.13:
        return True

    rel_nose = (float(nose_x) - sh_min) / max(1e-5, raw_span)
    both_shoulders_present = min(float(l_sh_vis), float(r_sh_vis)) >= 0.35

    if both_shoulders_present and (0.12 <= rel_nose <= 0.88):
        return False

    return True


class PoseTracker:
    """
    Real-time MediaPipe Pose wrapper with:
    1. High-accuracy PoseLandmarker Heavy model (offline, zero cost).
    2. Nose-Between-Shoulders Auto View Detection + 6-frame Hysteresis buffer
       (never flickers into side-profile when facing the screen!).
    3. Body-Size-Invariant Shoulder-Width Normalization + Personal Baseline Calibration.
    4. Anatomical Labeled Skeleton Overlay (EAR, SHOULDER, C7, SPINE, VERT-REF)
       without off-screen hip hallucination.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        ema_alpha: float = 0.40,
        model_path: str | Path | None = None,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.ema_alpha = float(np.clip(ema_alpha, 0.1, 1.0))
        self.model_path = Path(model_path) if model_path is not None else None

        self._smoothed_neck: float | None = None
        self._smoothed_torso: float | None = None

        # Hysteresis state to prevent view_mode flickering
        self._current_is_side: bool | None = None
        self._mode_candidate_streak: int = 0
        self._hysteresis_frames: int = 6

        # Personal Baseline Normalization state (persisted to disk in data/personal_baseline.json)
        self.baseline_path: Path = BASELINE_JSON_PATH
        self.baseline_calibrated: bool = False
        self.baseline_neck_deg: float = STANDARD_UPRIGHT_NECK_DEG
        self.baseline_torso_deg: float = STANDARD_UPRIGHT_TORSO_DEG
        self._load_personal_baseline()

        self.pose = None
        self.landmarker = None
        self.backend: str = "NONE"
        self.model_name: str = "NONE"
        self.spine_regressor: Any = None
        self._init_mediapipe()
        self._load_spine_regressor()

    def _load_spine_regressor(self) -> None:
        """Load the 4-Spine-Keypoint Regressor trained on Kaggle melsmm/posture-keypoints-detection."""
        if joblib is not None and KAGGLE_REGRESSOR_PATH.exists():
            try:
                self.spine_regressor = joblib.load(KAGGLE_REGRESSOR_PATH)
            except Exception:
                self.spine_regressor = None

    def _init_mediapipe(self) -> None:
        self.backend, self.pose, self.landmarker, self.model_name = _get_shared_mediapipe_backend(
            self.model_path,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )

    def _load_personal_baseline(self) -> None:
        """Load persisted personal baseline from disk if available."""
        try:
            if self.baseline_path.exists():
                data = json.loads(self.baseline_path.read_text(encoding="utf-8"))
                if data.get("calibrated"):
                    self.baseline_neck_deg = float(data["baseline_neck_deg"])
                    self.baseline_torso_deg = float(data["baseline_torso_deg"])
                    self.baseline_calibrated = True
        except Exception:
            self.baseline_calibrated = False

    def set_personal_baseline(
        self,
        raw_neck_angle: float,
        raw_torso_angle: float,
        persist: bool = True,
    ) -> None:
        """
        Record the current user's upright anatomy as their personal 0-point reference
        and persist it to data/personal_baseline.json so it survives browser/server restarts.
        """
        self.baseline_neck_deg = float(raw_neck_angle)
        self.baseline_torso_deg = float(raw_torso_angle)
        self.baseline_calibrated = True
        self.reset_smoothing()
        if persist:
            try:
                self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
                self.baseline_path.write_text(
                    json.dumps(
                        {
                            "calibrated": True,
                            "baseline_neck_deg": round(self.baseline_neck_deg, 2),
                            "baseline_torso_deg": round(self.baseline_torso_deg, 2),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

    def clear_personal_baseline(self, persist: bool = True) -> None:
        """Reset personal baseline back to universal RULA/REBA ergonomic standard."""
        self.baseline_calibrated = False
        self.baseline_neck_deg = STANDARD_UPRIGHT_NECK_DEG
        self.baseline_torso_deg = STANDARD_UPRIGHT_TORSO_DEG
        self.reset_smoothing()
        if persist and self.baseline_path.exists():
            try:
                self.baseline_path.unlink()
            except Exception:
                pass

    def _apply_personal_baseline(
        self,
        raw_neck: float,
        raw_torso: float,
    ) -> tuple[float, float]:
        """Shift raw angles by the user's personal baseline delta if calibrated."""
        if not self.baseline_calibrated:
            return raw_neck, raw_torso

        adj_neck = STANDARD_UPRIGHT_NECK_DEG + (raw_neck - self.baseline_neck_deg)
        adj_torso = STANDARD_UPRIGHT_TORSO_DEG + (raw_torso - self.baseline_torso_deg)

        return (
            round(float(np.clip(adj_neck, 4.0, 60.0)), 2),
            round(float(np.clip(adj_torso, 50.0, 105.0)), 2),
        )

    def reset_smoothing(self) -> None:
        self._smoothed_neck = None
        self._smoothed_torso = None

    def _resolve_view_mode_with_hysteresis(self, candidate_is_side: bool) -> bool:
        """Require 6 consecutive frames before flipping between FRONT and SIDE_PROFILE."""
        if self._current_is_side is None:
            self._current_is_side = candidate_is_side
            self._mode_candidate_streak = 0
            return self._current_is_side

        if candidate_is_side == self._current_is_side:
            self._mode_candidate_streak = 0
            return self._current_is_side

        self._mode_candidate_streak += 1
        if self._mode_candidate_streak >= self._hysteresis_frames:
            self._current_is_side = candidate_is_side
            self._mode_candidate_streak = 0

        return self._current_is_side

    def _apply_ema(self, neck: float, torso: float) -> tuple[float, float]:
        if self._smoothed_neck is None:
            self._smoothed_neck = neck
            self._smoothed_torso = torso
        else:
            a = self.ema_alpha
            self._smoothed_neck = a * neck + (1.0 - a) * self._smoothed_neck
            self._smoothed_torso = a * torso + (1.0 - a) * self._smoothed_torso

        return (
            round(float(self._smoothed_neck), 2),
            round(float(self._smoothed_torso), 2),
        )

    def _extract_landmarks(self, rgb_frame: np.ndarray) -> Any | None:
        """Run inference using either Solutions API or Tasks Vision API."""
        if self.backend == "SOLUTIONS" and self.pose is not None:
            rgb_frame.flags.writeable = False
            results = self.pose.process(rgb_frame)
            rgb_frame.flags.writeable = True
            if results.pose_landmarks:
                return results.pose_landmarks.landmark
            return None

        if self.backend == "TASKS" and self.landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                return result.pose_landmarks[0]
            return None

        return None

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        mirror: bool = True,
        status: str = "PASS",
    ) -> tuple[np.ndarray, PoseMetrics]:
        """
        Process a BGR video frame, extract body-size-invariant posture metrics,
        and draw a clear labeled anatomical skeleton overlay.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            return empty, PoseMetrics(detected=False)

        # Upgrade to heavy model automatically if it finished downloading
        if self.model_name != HEAVY_MODEL_PATH.name and HEAVY_MODEL_PATH.exists() and HEAVY_MODEL_PATH.stat().st_size > 5_000_000:
            self._init_mediapipe()

        frame = cv2.flip(bgr_frame, 1) if mirror else bgr_frame.copy()
        h, w = frame.shape[:2]

        if self.backend == "NONE":
            self._draw_not_detected_banner(frame, "MediaPipe Pose unavailable — use Simulation Mode")
            return frame, PoseMetrics(detected=False)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm = self._extract_landmarks(rgb_frame)

        if not lm:
            self.reset_smoothing()
            self._draw_not_detected_banner(frame, "No Person Detected — Please step into camera view")
            return frame, PoseMetrics(detected=False)

        l_ear = lm[IDX_LEFT_EAR]
        r_ear = lm[IDX_RIGHT_EAR]
        l_sh = lm[IDX_LEFT_SHOULDER]
        r_sh = lm[IDX_RIGHT_SHOULDER]
        l_hip = lm[IDX_LEFT_HIP]
        r_hip = lm[IDX_RIGHT_HIP]
        nose = lm[IDX_NOSE]

        l_ear_vis = float(getattr(l_ear, "visibility", 1.0) or 0.0)
        r_ear_vis = float(getattr(r_ear, "visibility", 1.0) or 0.0)
        l_sh_vis = float(getattr(l_sh, "visibility", 1.0) or 0.0)
        r_sh_vis = float(getattr(r_sh, "visibility", 1.0) or 0.0)
        l_hip_vis = float(getattr(l_hip, "visibility", 1.0) or 0.0)
        r_hip_vis = float(getattr(r_hip, "visibility", 1.0) or 0.0)

        left_upper_vis = (l_ear_vis + l_sh_vis) / 2.0
        right_upper_vis = (r_ear_vis + r_sh_vis) / 2.0
        best_upper_vis = max(left_upper_vis, right_upper_vis)

        if best_upper_vis < 0.35:
            self.reset_smoothing()
            self._draw_not_detected_banner(frame, "Upper body landmarks occluded — Adjust webcam angle")
            return frame, PoseMetrics(detected=False)

        # Dominant side selection (for side profile)
        left_score = l_ear_vis + l_sh_vis + 0.3 * l_hip_vis - 0.5 * float(getattr(l_sh, "z", 0.0))
        right_score = r_ear_vis + r_sh_vis + 0.3 * r_hip_vis - 0.5 * float(getattr(r_sh, "z", 0.0))
        use_left = left_score >= right_score

        ear = l_ear if use_left else r_ear
        sh = l_sh if use_left else r_sh
        hip = l_hip if use_left else r_hip
        hip_vis = l_hip_vis if use_left else r_hip_vis

        shoulder_span = abs(l_sh.x - r_sh.x)

        # Robust Nose-Between-Shoulders view detection + Hysteresis filter
        candidate_is_side = detect_is_side_profile(
            l_sh_x=l_sh.x,
            r_sh_x=r_sh.x,
            nose_x=nose.x,
            l_sh_vis=l_sh_vis,
            r_sh_vis=r_sh_vis,
        )
        is_side_profile = self._resolve_view_mode_with_hysteresis(candidate_is_side)

        # Strictly require hips to be well inside the frame (y <= 0.88 and high confidence)
        # so off-screen desk sitting NEVER uses MediaPipe's hallucinated hip coordinates
        hip_truly_visible = (hip_vis >= 0.65) and (0.10 <= hip.y <= 0.88)

        if is_side_profile:
            view_mode = "SIDE_PROFILE"
            active_side = "LEFT" if use_left else "RIGHT"

            ear_px = (ear.x * w, ear.y * h)
            sh_px = (sh.x * w, sh.y * h)
            hip_px = (hip.x * w, hip.y * h)

            raw_neck = calculate_neck_angle_2d(ear_px, sh_px)

            if hip_truly_visible:
                raw_torso = calculate_torso_angle_2d(sh_px, hip_px)
            else:
                # Hips are occluded/below the desk in Side-Profile:
                # Do NOT use the hallucinated hip_px if it points toward the front/chest!
                # Check if MediaPipe's predicted hip is anatomically behind/under the shoulder
                # relative to the face direction (nose.x vs sh.x).
                face_dir = 1.0 if (nose.x >= sh.x) else -1.0
                # When leaning forward in side profile, the shoulder is shifted in `face_dir`
                # relative to the hip (`(sh.x - hip.x) * face_dir > 0`).
                hip_behind_or_aligned = ((sh.x - hip.x) * face_dir) >= -0.02

                neck_len_px = math.hypot(ear_px[0] - sh_px[0], ear_px[1] - sh_px[1])
                nose_drop_ratio = max(0.0, (nose.y - ear.y) * h / max(20.0, neck_len_px))
                upper_kyphosis_deg = max(0.0, raw_neck - 16.0) * 0.68 + (nose_drop_ratio * 18.0)

                if hip_behind_or_aligned and hip_vis >= 0.40:
                    torso_2d = calculate_torso_angle_2d(sh_px, hip_px)
                    est_torso = 0.55 * torso_2d + 0.45 * (95.0 - upper_kyphosis_deg)
                else:
                    # Pure upper-thoracic estimate when hip is completely occluded
                    est_torso = 95.0 - upper_kyphosis_deg

                raw_torso = round(float(np.clip(est_torso, 55.0, 102.0)), 2)

        elif hip_truly_visible:
            view_mode = "FRONT_HYBRID"
            active_side = "BILATERAL"
            aspect = float(w) / max(1.0, float(h))
            ear_mid_y = 0.5 * (l_ear.y + r_ear.y)
            sh_mid_x = 0.5 * (l_sh.x + r_sh.x) * aspect
            sh_mid_y = 0.5 * (l_sh.y + r_sh.y)
            hip_mid_x = 0.5 * (l_hip.x + r_hip.x) * aspect
            hip_mid_y = 0.5 * (l_hip.y + r_hip.y)

            # Compute lateral head roll (ear-to-ear tilt), lateral neck lean (head vs C7), and shoulder tilt angle
            ear_roll_deg = math.degrees(
                math.atan2(abs(l_ear.y - r_ear.y) * h, max(8.0, abs(l_ear.x - r_ear.x) * w))
            )
            sh_roll_deg = math.degrees(
                math.atan2(abs(l_sh.y - r_sh.y) * h, max(12.0, abs(l_sh.x - r_sh.x) * w))
            )
            head_center_x = 0.5 * nose.x + 0.25 * (l_ear.x + r_ear.x)
            c7_x = 0.5 * (l_sh.x + r_sh.x)
            lateral_neck_deg = math.degrees(
                math.atan2(abs(head_center_x - c7_x) * w, max(8.0, abs(sh_mid_y - ear_mid_y) * h))
            )

            raw_neck, raw_torso = self._compute_front_fallback_angles(
                ear_mid_y=ear_mid_y,
                sh_mid_y=sh_mid_y,
                shoulder_span=shoulder_span,
                nose_z=nose.z,
                sh_mid_z=0.5 * (l_sh.z + r_sh.z),
                hip_mid_xy=(hip_mid_x, hip_mid_y),
                sh_mid_xy=(sh_mid_x, sh_mid_y),
                nose_y=nose.y,
                nose_x=nose.x,
                sh_mid_x_norm=c7_x,
                ear_roll_deg=ear_roll_deg,
                lateral_neck_deg=lateral_neck_deg,
                sh_roll_deg=sh_roll_deg,
            )
        else:
            view_mode = "FRONT_FALLBACK"
            active_side = "BILATERAL"
            ear_mid_y = 0.5 * (l_ear.y + r_ear.y)
            sh_mid_y = 0.5 * (l_sh.y + r_sh.y)
            c7_x = 0.5 * (l_sh.x + r_sh.x)

            # Compute lateral head roll (ear-to-ear tilt), lateral neck lean (head vs C7), and shoulder tilt angle
            ear_roll_deg = math.degrees(
                math.atan2(abs(l_ear.y - r_ear.y) * h, max(8.0, abs(l_ear.x - r_ear.x) * w))
            )
            sh_roll_deg = math.degrees(
                math.atan2(abs(l_sh.y - r_sh.y) * h, max(12.0, abs(l_sh.x - r_sh.x) * w))
            )
            head_center_x = 0.5 * nose.x + 0.25 * (l_ear.x + r_ear.x)
            lateral_neck_deg = math.degrees(
                math.atan2(abs(head_center_x - c7_x) * w, max(8.0, abs(sh_mid_y - ear_mid_y) * h))
            )

            raw_neck, raw_torso = self._compute_front_fallback_angles(
                ear_mid_y=ear_mid_y,
                sh_mid_y=sh_mid_y,
                shoulder_span=max(0.16, shoulder_span),
                nose_z=nose.z,
                sh_mid_z=0.5 * (l_sh.z + r_sh.z),
                hip_mid_xy=None,
                sh_mid_xy=None,
                nose_y=nose.y,
                nose_x=nose.x,
                sh_mid_x_norm=c7_x,
                ear_roll_deg=ear_roll_deg,
                lateral_neck_deg=lateral_neck_deg,
                sh_roll_deg=sh_roll_deg,
            )

        # Apply Personal Baseline Normalization (if user calibrated their body shape)
        norm_neck, norm_torso = self._apply_personal_baseline(raw_neck, raw_torso)

        # Smooth with EMA
        neck_angle, torso_angle = self._apply_ema(norm_neck, norm_torso)

        metrics = PoseMetrics(
            detected=True,
            neck_angle=neck_angle,
            torso_angle=torso_angle,
            active_side=active_side,
            view_mode=view_mode,
            visibility_score=round(float(best_upper_vis), 2),
            raw_neck_angle=raw_neck,
            raw_torso_angle=raw_torso,
            baseline_calibrated=self.baseline_calibrated,
        )

        self._draw_skeleton_overlay(
            frame=frame,
            nose=nose,
            l_ear=l_ear,
            r_ear=r_ear,
            l_sh=l_sh,
            r_sh=r_sh,
            l_hip=l_hip,
            r_hip=r_hip,
            hip_truly_visible=hip_truly_visible,
            is_side_profile=is_side_profile,
            use_left=use_left,
            metrics=metrics,
            status=status,
        )

        return frame, metrics

    @staticmethod
    def _compute_front_fallback_angles(
        ear_mid_y: float,
        sh_mid_y: float,
        shoulder_span: float,
        nose_z: float,
        sh_mid_z: float,
        hip_mid_xy: tuple[float, float] | None,
        sh_mid_xy: tuple[float, float] | None,
        nose_y: float | None = None,
        nose_x: float | None = None,
        sh_mid_x_norm: float | None = None,
        ear_roll_deg: float = 0.0,
        lateral_neck_deg: float = 0.0,
        sh_roll_deg: float = 0.0,
    ) -> tuple[float, float]:
        """
        Scale-invariant front/three-quarter upper-body geometry normalized by
        bi-acromial shoulder width (`shoulder_span`), capturing forward head
        flexion, lateral head tilt ("เอียงคอซ้าย-ขวา"), AND shoulder/upper-torso
        slant ("ไหล่เอียง / ตัวเอียง"):
          - neck_angle: ~9°-17° upright, 25°-48° when head juts forward OR tilts sideways
          - torso_angle: ~90°-98° upright, 62°-76° when slouching OR tilting torso sideways
        """
        span = max(0.16, float(shoulder_span))
        neck_height = max(0.01, sh_mid_y - ear_mid_y)

        # 1. Scale-invariant vertical neck-to-shoulder ratio
        neck_ratio = neck_height / span

        # 2. Scale-invariant horizontal head-forward/side offset
        fhp_horiz_ratio = 0.0
        if nose_x is not None and sh_mid_x_norm is not None:
            fhp_horiz_ratio = max(0.0, (abs(float(nose_x) - float(sh_mid_x_norm)) / span) - 0.06)

        # 3. Relative Z-depth protrusion
        z_protrusion = max(0.0, float(sh_mid_z - nose_z))

        # 4. Head downward pitch (nose dropping toward chest relative to ears)
        pitch_drop = 0.0
        if nose_y is not None:
            pitch_drop = max(-0.08, (float(nose_y) - ear_mid_y) / max(0.06, neck_height))

        ratio_delta = 0.56 - neck_ratio
        z_delta = max(-0.05, z_protrusion - 0.24)
        forward_neck_angle = (
            12.0
            + (ratio_delta * 70.0)
            + (z_delta * 34.0)
            + (pitch_drop * 20.0)
            + (fhp_horiz_ratio * 42.0)
        )

        # 5. Lateral Head Roll / Neck Tilt ("เอียงคอซ้าย-ขวา") + Shoulder-Neck Skew
        roll_excess = max(0.0, float(ear_roll_deg) - 4.0)
        lat_excess = max(0.0, float(lateral_neck_deg) - 3.5)
        sh_skew_excess = max(0.0, float(sh_roll_deg) - 6.0)
        lateral_tilt_angle = 11.5 + (roll_excess * 1.25) + (lat_excess * 0.90) + (sh_skew_excess * 0.85)

        neck_angle = max(forward_neck_angle, lateral_tilt_angle)
        neck_angle = float(np.clip(neck_angle, 5.0, 55.0))

        # 6. Torso Inclination: include lateral shoulder/torso roll (sh_roll_deg)
        sh_roll_penalty = max(0.0, float(sh_roll_deg) - 4.5) * 1.45

        if hip_mid_xy is not None and sh_mid_xy is not None:
            torso_2d = calculate_torso_angle_2d(sh_mid_xy, hip_mid_xy)
            torso_height_ratio = abs(hip_mid_xy[1] - sh_mid_xy[1]) / span
            compression_penalty = max(0.0, 1.30 - torso_height_ratio) * 32.0
            torso_angle = torso_2d - compression_penalty - sh_roll_penalty
        else:
            # Scale-invariant slouch & lateral lean detection
            shoulder_drop = max(-0.04, sh_mid_y - 0.64)
            compression = max(-0.05, 0.48 - neck_ratio)
            torso_angle = 94.5 - (shoulder_drop * 75.0) - (compression * 78.0) - sh_roll_penalty

        torso_angle = float(np.clip(torso_angle, 55.0, 102.0))
        return round(neck_angle, 2), round(torso_angle, 2)

    @staticmethod
    def _draw_spine_text(
        frame: np.ndarray,
        text: str,
        pt: tuple[int, int],
        offset: tuple[int, int] = (14, 14),
    ) -> None:
        """Draw crisp white text with subtle dark outline like the Kaggle Spine Keypoints reference."""
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
        tx = int(np.clip(pt[0] + offset[0], 8, max(8, w - tw - 8)))
        ty = int(np.clip(pt[1] + offset[1], th + 8, max(th + 8, h - 8)))
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (250, 250, 250), 1, cv2.LINE_AA)

    def _draw_skeleton_overlay(
        self,
        frame: np.ndarray,
        nose: Any,
        l_ear: Any,
        r_ear: Any,
        l_sh: Any,
        r_sh: Any,
        l_hip: Any,
        r_hip: Any,
        hip_truly_visible: bool,
        is_side_profile: bool,
        use_left: bool,
        metrics: PoseMetrics,
        status: str,
    ) -> None:
        """
        Render the 4 Anatomical Spine Keypoints (CERVICAL SPINE, THORACIC SPINE,
        LUMBAR SPINE, SACRAL SPINE) and the Yellow Spine Bounding Box Frame
        matching the Kaggle `melsmm/posture-keypoints-detection` visualization.
        """
        h, w = frame.shape[:2]

        def to_px(pt: Any) -> tuple[int, int]:
            return int(np.clip(pt.x * w, 0, w - 1)), int(np.clip(pt.y * h, 0, h - 1))

        if is_side_profile:
            ear_obj = l_ear if use_left else r_ear
            sh_obj = l_sh if use_left else r_sh
            hip_obj = l_hip if use_left else r_hip

            ear_pt = to_px(ear_obj)
            sh_pt = to_px(sh_obj)
            face_dir = -1.0 if (nose.x * w < sh_pt[0]) else 1.0
            neck_len = max(35.0, math.hypot(ear_pt[0] - sh_pt[0], ear_pt[1] - sh_pt[1]))

            # 1. CERVICAL SPINE (at ear / upper cervical neck)
            pt_cervical = ear_pt

            # 2. THORACIC SPINE (shifted slightly toward posterior upper back behind shoulder)
            thoracic_x = int(np.clip(sh_pt[0] - face_dir * (0.22 * neck_len), 12, w - 12))
            thoracic_y = int(np.clip(sh_pt[1] + (0.08 * neck_len), 12, h - 25))
            pt_thoracic = (thoracic_x, thoracic_y)

            # 3. LUMBAR SPINE & 4. SACRAL SPINE
            if hip_truly_visible:
                hip_pt = to_px(hip_obj)
                sacral_x = int(np.clip(hip_pt[0] - face_dir * (0.18 * neck_len), 12, w - 12))
                sacral_y = int(np.clip(hip_pt[1], pt_thoracic[1] + 40, h - 14))
                pt_sacral = (sacral_x, sacral_y)

                lumbar_x = int(
                    np.clip(
                        0.45 * pt_thoracic[0] + 0.55 * pt_sacral[0] - face_dir * (0.18 * neck_len),
                        12,
                        w - 12,
                    )
                )
                lumbar_y = int(np.clip(0.45 * pt_thoracic[1] + 0.55 * pt_sacral[1], pt_thoracic[1] + 20, h - 18))
                pt_lumbar = (lumbar_x, lumbar_y)
            else:
                tilt_rad = math.radians(max(0.0, 95.0 - metrics.torso_angle))
                avail_h = max(70, h - pt_thoracic[1] - 20)
                lumbar_x = int(
                    np.clip(
                        pt_thoracic[0] - face_dir * (0.58 * avail_h * math.sin(tilt_rad) + 0.16 * neck_len),
                        12,
                        w - 12,
                    )
                )
                lumbar_y = int(np.clip(pt_thoracic[1] + 0.55 * avail_h * math.cos(tilt_rad), 12, h - 22))
                pt_lumbar = (lumbar_x, lumbar_y)

                sacral_x = int(
                    np.clip(
                        pt_lumbar[0] - face_dir * (0.25 * avail_h * math.sin(tilt_rad) - 0.10 * neck_len),
                        12,
                        w - 12,
                    )
                )
                sacral_y = int(np.clip(pt_thoracic[1] + 0.94 * avail_h, pt_lumbar[1] + 15, h - 14))
                pt_sacral = (sacral_x, sacral_y)

        else:
            # Front / Three-Quarter View: 4 Anatomical Spine Keypoints following neck & shoulder tilt
            l_sh_px = to_px(l_sh)
            r_sh_px = to_px(r_sh)
            l_ear_px = to_px(l_ear)
            r_ear_px = to_px(r_ear)
            nose_px = to_px(nose)

            c7_mid = ((l_sh_px[0] + r_sh_px[0]) // 2, (l_sh_px[1] + r_sh_px[1]) // 2)
            ear_mid_x = (l_ear_px[0] + r_ear_px[0]) // 2
            ear_mid_y = (l_ear_px[1] + r_ear_px[1]) // 2

            # 1. CERVICAL SPINE (Upper cervical spine at ear/occiput level, never on the nose tip)
            cervical_x = int(0.75 * ear_mid_x + 0.25 * nose_px[0])
            cervical_y = int(ear_mid_y + 0.22 * max(10, c7_mid[1] - ear_mid_y))
            pt_cervical = (int(np.clip(cervical_x, 12, w - 12)), int(np.clip(cervical_y, 12, h - 20)))

            # 2. THORACIC SPINE (C7/T2 thoracic midpoint)
            pt_thoracic = c7_mid

            # Signed shoulder tilt angle (perpendicular determines thoracic-lumbar spine slant)
            # In mirrored frame, left and right shoulders define the clavicle bar
            sh_leftmost, sh_rightmost = (l_sh_px, r_sh_px) if l_sh_px[0] <= r_sh_px[0] else (r_sh_px, l_sh_px)
            sh_dx = max(12.0, float(sh_rightmost[0] - sh_leftmost[0]))
            sh_dy = float(sh_rightmost[1] - sh_leftmost[1])
            spine_tilt_rad = math.atan2(sh_dy, sh_dx)

            if hip_truly_visible:
                l_hip_px = to_px(l_hip)
                r_hip_px = to_px(r_hip)
                pt_sacral = ((l_hip_px[0] + r_hip_px[0]) // 2, (l_hip_px[1] + r_hip_px[1]) // 2)
                pt_lumbar = (
                    (pt_thoracic[0] + pt_sacral[0]) // 2,
                    (pt_thoracic[1] + pt_sacral[1]) // 2,
                )
            else:
                avail_h = min(170, max(65, h - pt_thoracic[1] - 18))
                # Spine projects perpendicular to the shoulder line
                lumbar_dx = int(-0.55 * avail_h * math.sin(spine_tilt_rad))
                lumbar_dy = int(0.55 * avail_h * math.cos(spine_tilt_rad))
                pt_lumbar = (
                    int(np.clip(pt_thoracic[0] + lumbar_dx, 12, w - 12)),
                    int(np.clip(pt_thoracic[1] + lumbar_dy, pt_thoracic[1] + 15, h - 22)),
                )

                sacral_dx = int(-0.95 * avail_h * math.sin(spine_tilt_rad))
                sacral_dy = int(0.95 * avail_h * math.cos(spine_tilt_rad))
                pt_sacral = (
                    int(np.clip(pt_thoracic[0] + sacral_dx, 12, w - 12)),
                    int(np.clip(pt_thoracic[1] + sacral_dy, pt_lumbar[1] + 12, h - 12)),
                )

        # -------------------------------------------------------------------
        # Draw Kaggle Posture Keypoints Style Overlay (Image 1 Exact Style)
        # -------------------------------------------------------------------
        spine_pts = [pt_cervical, pt_thoracic, pt_lumbar, pt_sacral]
        xs = [p[0] for p in spine_pts]
        ys = [p[1] for p in spine_pts]

        box_color = (0, 235, 255) if status == "PASS" else ((0, 180, 255) if status == "MONITORING" else (60, 75, 255))
        yellow_bgr = (0, 238, 255)
        dark_outline = (15, 15, 15)

        bx1 = int(np.clip(min(xs) - 34, 10, w - 20))
        by1 = int(np.clip(min(ys) - 18, 22, h - 20))
        bx2 = int(np.clip(max(xs) + 34, bx1 + 30, w - 10))
        by2 = int(np.clip(max(ys) + 16, by1 + 30, h - 10))
        bmx = (bx1 + bx2) // 2
        bmy = (by1 + by2) // 2

        # 1. Outer Spine Bounding Frame (Yellow rectangle + 8 yellow perimeter dots)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2, cv2.LINE_AA)
        anchor_pts = [
            (bx1, by1), (bmx, by1), (bx2, by1),
            (bx1, bmy),             (bx2, bmy),
            (bx1, by2), (bmx, by2), (bx2, by2),
        ]
        for apt in anchor_pts:
            cv2.circle(frame, apt, 6, yellow_bgr, -1, cv2.LINE_AA)
            cv2.circle(frame, apt, 7, dark_outline, 1, cv2.LINE_AA)

        # Top white reference point above cervical box
        top_ref_pt = (bmx, max(10, by1 - 20))
        cv2.circle(frame, top_ref_pt, 6, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(frame, top_ref_pt, 7, dark_outline, 1, cv2.LINE_AA)

        # 2. Connecting Spine Curve (CERVICAL -> THORACIC -> LUMBAR -> SACRAL)
        for i in range(len(spine_pts) - 1):
            cv2.line(frame, spine_pts[i], spine_pts[i + 1], yellow_bgr, 2, cv2.LINE_AA)

        # 3. 4 Anatomical Spine Keypoint Dots & Labels
        keypoint_specs = [
            (pt_cervical, (36, 28, 237), "CERVICAL SPINE", (14, 16)),   # Red
            (pt_thoracic, (76, 177, 34), "THORACIC SPINE", (14, 16)),   # Green
            (pt_lumbar, (232, 62, 0), "LUMBAR SPINE", (14, 16)),        # Blue
            (pt_sacral, (0, 242, 255), "SACRAL SPINE", (14, 14)),       # Yellow
        ]
        for pt, kcolor, klabel, koff in keypoint_specs:
            cv2.circle(frame, pt, 7, kcolor, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 8, dark_outline, 2, cv2.LINE_AA)
            self._draw_spine_text(frame, klabel, pt, offset=koff)

    @staticmethod
    def _draw_hud_box(frame: np.ndarray, metrics: PoseMetrics, status: str) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (14, 14), (345, 134), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        if status == "ALERT":
            badge_color = (68, 68, 239)
        elif status == "MONITORING":
            badge_color = (11, 158, 245)
        else:
            badge_color = (129, 185, 16)

        cv2.rectangle(frame, (14, 14), (20, 134), badge_color, -1)
        cv2.putText(
            frame,
            f"Neck:  {metrics.neck_angle:5.1f} deg",
            (30, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (248, 250, 252),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Torso: {metrics.torso_angle:5.1f} deg",
            (30, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (248, 250, 252),
            2,
            cv2.LINE_AA,
        )
        cal_tag = " [BASELINE ON]" if metrics.baseline_calibrated else ""
        cv2.putText(
            frame,
            f"Mode:  {metrics.view_mode}{cal_tag}",
            (30, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )

        h, w = frame.shape[:2]
        pill_w = 155
        cv2.rectangle(frame, (w - pill_w - 16, 16), (w - 16, 56), badge_color, -1)
        cv2.putText(
            frame,
            f"STATUS: {status}",
            (w - pill_w - 4, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_not_detected_banner(frame: np.ndarray, message: str) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, h // 2 - 40), (w - 20, h // 2 + 40), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.putText(
            frame,
            message,
            (40, h // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (56, 189, 248),
            2,
            cv2.LINE_AA,
        )

    def render_simulation_frame(
        self,
        neck_angle: float,
        torso_angle: float,
        status: str = "PASS",
        width: int = 720,
        height: int = 460,
    ) -> tuple[np.ndarray, PoseMetrics]:
        """
        Generate a synthetic dark-slate skeletal visualization frame for
        Demo Simulation Mode when testing without a physical webcam.
        """
        frame = np.full((height, width, 3), (26, 20, 15), dtype=np.uint8)

        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (38, 30, 24), 1)
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (38, 30, 24), 1)

        if status == "ALERT":
            color = (68, 68, 240)
        elif status == "MONITORING":
            color = (11, 158, 245)
        else:
            color = (129, 185, 16)

        hip_pt = (int(width * 0.45), int(height * 0.80))
        torso_len = 155.0
        neck_len = 95.0

        torso_forward_rad = math.radians(95.0 - float(torso_angle))
        sh_x = int(hip_pt[0] + torso_len * math.sin(torso_forward_rad))
        sh_y = int(hip_pt[1] - torso_len * math.cos(torso_forward_rad))
        sh_pt = (sh_x, sh_y)

        neck_forward_rad = math.radians(float(neck_angle))
        ear_x = int(sh_pt[0] + neck_len * math.sin(neck_forward_rad))
        ear_y = int(sh_pt[1] - neck_len * math.cos(neck_forward_rad))
        ear_pt = (ear_x, ear_y)

        cv2.line(frame, (hip_pt[0] - 25, hip_pt[1] + 10), (hip_pt[0] - 25, hip_pt[1] - 165), (80, 90, 105), 4)
        cv2.line(frame, (hip_pt[0] - 25, hip_pt[1] + 10), (hip_pt[0] + 65, hip_pt[1] + 10), (80, 90, 105), 4)

        cv2.line(frame, hip_pt, (hip_pt[0], hip_pt[1] - 180), (110, 120, 135), 1, cv2.LINE_AA)
        cv2.line(frame, sh_pt, (sh_pt[0], sh_pt[1] - 115), (110, 120, 135), 1, cv2.LINE_AA)

        cv2.line(frame, hip_pt, sh_pt, color, 6, cv2.LINE_AA)
        cv2.line(frame, sh_pt, ear_pt, color, 6, cv2.LINE_AA)

        cv2.circle(frame, ear_pt, 22, color, 2, cv2.LINE_AA)
        for pt in (hip_pt, sh_pt, ear_pt):
            cv2.circle(frame, pt, 8, (238, 211, 34), -1, cv2.LINE_AA)

        metrics = PoseMetrics(
            detected=True,
            neck_angle=round(float(neck_angle), 2),
            torso_angle=round(float(torso_angle), 2),
            active_side="SIMULATED",
            view_mode="SIMULATED",
            visibility_score=1.0,
            raw_neck_angle=round(float(neck_angle), 2),
            raw_torso_angle=round(float(torso_angle), 2),
            baseline_calibrated=self.baseline_calibrated,
        )
        return frame, metrics

    def render_camera_standby_frame(
        self,
        width: int = 960,
        height: int = 540,
    ) -> np.ndarray:
        """
        Render a clean Google Meet-style dark standby frame when the webcam is turned off.
        """
        frame = np.full((height, width, 3), (26, 22, 18), dtype=np.uint8)
        cx, cy = width // 2, height // 2

        # Subtle avatar/camera circle in the center
        cv2.circle(frame, (cx, cy), 54, (45, 38, 30), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 54, (75, 65, 52), 2, cv2.LINE_AA)

        # Minimal camera body + lens polygon inside the circle
        cv2.rectangle(frame, (cx - 22, cy - 14), (cx + 10, cy + 14), (160, 150, 140), 2, cv2.LINE_AA)
        pts = np.array([[cx + 10, cy - 6], [cx + 24, cy - 13], [cx + 24, cy + 13], [cx + 10, cy + 6]], np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(160, 150, 140), thickness=2, lineType=cv2.LINE_AA)
        # Slash line indicating camera is currently off
        cv2.line(frame, (cx - 26, cy - 22), (cx + 26, cy + 22), (110, 110, 235), 2, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        """Reset smoothing state without destroying the shared process-level MediaPipe accelerator."""
        self.reset_smoothing()
