"""
Module 3: src/rule_engine.py
Rule-Based Expert System (Forward Chaining / IF-THEN) with Sliding Time Window
and Real-Time Ergo Score (0-100) computation for ErgoGuard AI (CSC472).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any


@dataclass
class RuleEvaluationResult:
    """
    Structured output of the Rule-Based Expert System.
    Supports both attribute access (res.status) and dict-style access (res['status']).
    """

    status: str  # 'PASS' | 'MONITORING' | 'ALERT'
    ergo_score: float
    triggered_rule_id: str  # 'R1' | 'R2' | 'R3' | 'R4'
    alert_message_th: str
    stretch_recommendation_th: str
    bad_duration_sec: float
    countdown_sec: float = 0.0
    threshold_sec: float = 15.0
    session_good_ratio: float = 1.0
    rule_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return asdict(self).keys()


class ErgonomicRuleEngine:
    """
    Forward-Chaining Rule-Based Expert System for Ergonomic Posture Monitoring.

    Explicit Rules:
    - Rule R1 (Good Posture):
        IF knn_label == 'Good Posture' AND neck_angle < 20
        THEN Status = 'PASS', Reset alert timer, No stretch needed.
    - Rule R2 (Mild Drift Warning):
        IF (not R1) AND bad_duration < threshold_sec
        THEN Status = 'MONITORING', Show countdown timer before alert.
    - Rule R3 (Forward Head Alert):
        IF (knn_label == 'Forward Head' OR neck_angle >= 30) AND bad_duration >= threshold_sec
        THEN Status = 'ALERT', Recommend Chin Tuck & Upper Trapezius Stretch (30s).
    - Rule R4 (Slouching Alert):
        IF (knn_label == 'Slouching' OR torso_angle <= 75) AND bad_duration >= threshold_sec
        THEN Status = 'ALERT', Recommend Thoracic Extension & Chest Opener (30s).
    """

    def __init__(self, threshold_sec: float = 15.0) -> None:
        self.threshold_sec: float = float(threshold_sec)
        self.bad_posture_start_time: float | None = None
        self.bad_duration_sec: float = 0.0
        self.last_eval_time: float | None = None

        # Session analytics counters
        self.total_evaluations: int = 0
        self.good_evaluations: int = 0
        self.class_counts: dict[str, int] = {
            "Good Posture": 0,
            "Forward Head": 0,
            "Slouching": 0,
        }
        self._smoothed_ergo_score: float = 96.0

    def set_threshold(self, threshold_sec: float) -> None:
        """Update configurable sliding time window threshold (e.g. 5s - 30s)."""
        self.threshold_sec = max(1.0, float(threshold_sec))

    def reset(self) -> None:
        """Reset timers and session statistics."""
        self.bad_posture_start_time = None
        self.bad_duration_sec = 0.0
        self.last_eval_time = None
        self.total_evaluations = 0
        self.good_evaluations = 0
        self.class_counts = {
            "Good Posture": 0,
            "Forward Head": 0,
            "Slouching": 0,
        }
        self._smoothed_ergo_score = 96.0

    def evaluate(
        self,
        knn_label: str,
        neck_angle: float,
        torso_angle: float,
        current_time: float | None = None,
        override_bad_duration_sec: float | None = None,
    ) -> RuleEvaluationResult:
        """
        Run Forward-Chaining inference over current facts:
          Fact 1: knn_label
          Fact 2: neck_angle, torso_angle
          Fact 3: bad_posture_duration_sec
        """
        now = float(current_time) if current_time is not None else time.monotonic()
        neck_val = float(neck_angle)
        torso_val = float(torso_angle)

        # Update session counts
        self.total_evaluations += 1
        if knn_label in self.class_counts:
            self.class_counts[knn_label] += 1

        # Fact check for Rule R1 (Good Posture condition: neck < 20° AND torso > 80°)
        is_good_posture = (
            (knn_label == "Good Posture")
            and (neck_val < 20.0)
            and (torso_val > 80.0)
        )

        if is_good_posture:
            self.good_evaluations += 1
            self.bad_posture_start_time = None
            self.bad_duration_sec = 0.0
        else:
            if override_bad_duration_sec is not None:
                self.bad_duration_sec = float(override_bad_duration_sec)
            else:
                if self.bad_posture_start_time is None:
                    self.bad_posture_start_time = now
                    self.bad_duration_sec = 0.0
                else:
                    self.bad_duration_sec = max(0.0, now - self.bad_posture_start_time)

        self.last_eval_time = now
        bad_dur = round(self.bad_duration_sec, 1)
        countdown = max(0.0, round(self.threshold_sec - bad_dur, 1))

        # Evaluate predicates for Forward Chaining Rules R1 - R4
        r1_cond = is_good_posture
        r2_cond = (not is_good_posture) and (self.bad_duration_sec < self.threshold_sec)
        r3_cond = (
            (not is_good_posture)
            and (self.bad_duration_sec >= self.threshold_sec)
            and (knn_label == "Forward Head" or neck_val >= 30.0)
        )
        r4_cond = (
            (not is_good_posture)
            and (self.bad_duration_sec >= self.threshold_sec)
            and (knn_label == "Slouching" or torso_val <= 75.0)
        )

        # Decide which rule fires (Forward Chaining priority resolution)
        if r1_cond:
            triggered_rule_id = "R1"
            status = "PASS"
            alert_message_th = "ท่านั่งอยู่ในเกณฑ์มาตรฐานตามหลักการยศาสตร์ (Good Ergonomic Posture)"
            stretch_recommendation_th = "ยังไม่ต้องยืดเหยียดกล้ามเนื้อ — รักษาแนวศีรษะและหลังให้ตรงต่อไป"
        elif r2_cond:
            triggered_rule_id = "R2"
            status = "MONITORING"
            is_slouch = knn_label == "Slouching" or torso_val <= 78.0
            is_neck_forward = knn_label == "Forward Head" or neck_val >= 25.0
            if is_slouch and is_neck_forward:
                posture_desc = "นั่งหลังค่อม/โน้มตัว และก้มคอพร้อมกัน"
            elif is_slouch:
                posture_desc = "นั่งหลังค่อม/โน้มตัว"
            else:
                posture_desc = "คอยื่นไปด้านหน้า"

            alert_message_th = (
                f"กำลังเฝ้าระวัง ({posture_desc}) — หากค้างท่านี้ครบ {int(self.threshold_sec)} วินาที ระบบจะแจ้งเตือน"
            )
            stretch_recommendation_th = (
                f"ปรับท่านั่งให้ตรงภายใน {countdown:.1f} วินาที เพื่อรีเซ็ตตัวจับเวลาและรักษาคะแนน Ergo Score"
            )
        else:
            # Duration has reached or exceeded threshold_sec -> ALERT!
            # Determine between R4 (Slouching) and R3 (Forward Head)
            if knn_label == "Slouching" or torso_val <= 78.0:
                triggered_rule_id = "R4"
                r4_cond = True
                status = "ALERT"
                if neck_val >= 25.0:
                    alert_message_th = (
                        f"แจ้งเตือน! ตรวจพบอาการนั่งหลังค่อม/โน้มตัวร่วมกับก้มคอ (Slouching + Forward Head) ต่อเนื่อง {bad_dur:.1f} วินาที"
                    )
                    stretch_recommendation_th = (
                        "แนะนำท่ายืดเหยียด: Thoracic Extension & Chest Opener ร่วมกับ Chin Tuck (30 วินาที) — "
                        "ประสานมือด้านหลังศีรษะ แอ่นอกเปิดไหล่ไปด้านหลังช้าๆ และเก็บคางเข้าหาลำคอ ค้างไว้ 10-15 วินาที ทำซ้ำ 2-3 รอบ"
                    )
                else:
                    alert_message_th = (
                        f"แจ้งเตือน! ตรวจพบอาการนั่งหลังค่อม/โน้มตัว (Slouching) ต่อเนื่อง {bad_dur:.1f} วินาที"
                    )
                    stretch_recommendation_th = (
                        "แนะนำท่ายืดเหยียด: Thoracic Extension & Chest Opener (30 วินาที) — "
                        "ประสานมือด้านหลังศีรษะ แอ่นอกเปิดไหล่ไปด้านหลังช้าๆ ค้างไว้ 10-15 วินาที ทำซ้ำ 2-3 รอบ"
                    )
            else:
                triggered_rule_id = "R3"
                r3_cond = True
                status = "ALERT"
                alert_message_th = (
                    f"แจ้งเตือน! ตรวจพบอาการคอยื่น (Forward Head / Tech Neck) ต่อเนื่อง {bad_dur:.1f} วินาที"
                )
                stretch_recommendation_th = (
                    "แนะนำท่ายืดเหยียด: Chin Tuck & Upper Trapezius Stretch (30 วินาที) — "
                    "เก็บคางเข้าหาลำคอช้าๆ ให้แนวหูตรงกับไหล่ ค้างไว้ 5 วินาที แล้วเอียงศีรษะยืดกล้ามเนื้อบ่าข้างละ 15 วินาที"
                )

        # Build human-readable rule trace
        rule_trace = [
            {
                "rule_id": "R1",
                "name": "ท่านั่งปกติ",
                "condition": "นั่งหลังตรง และก้มคอไม่เกิน 20°",
                "matched": bool(triggered_rule_id == "R1"),
                "effect": "ผ่านเกณฑ์ (รีเซ็ตเวลาสะสม)",
            },
            {
                "rule_id": "R2",
                "name": "เฝ้าระวัง",
                "condition": f"เริ่มก้มคอหรือหลังค่อม ({bad_dur:.1f}s / {self.threshold_sec:.0f}s)",
                "matched": bool(triggered_rule_id == "R2"),
                "effect": f"นับถอยหลังอีก {countdown:.1f} วินาทีก่อนเตือน",
            },
            {
                "rule_id": "R3",
                "name": "เตือนอาการคอยื่น",
                "condition": f"ก้ม/ยื่นคอเกินเกณฑ์ต่อเนื่องครบ {self.threshold_sec:.0f} วินาที",
                "matched": bool(triggered_rule_id == "R3"),
                "effect": "แจ้งเตือน + แนะนำท่าเก็บคาง (Chin Tuck)",
            },
            {
                "rule_id": "R4",
                "name": "เตือนอาการหลังค่อม",
                "condition": f"นั่งหลังค่อม/โน้มตัวต่อเนื่องครบ {self.threshold_sec:.0f} วินาที",
                "matched": bool(triggered_rule_id == "R4"),
                "effect": "แจ้งเตือน + แนะนำท่าแอ่นอกยืดหลัง (Chest Opener)",
            },
        ]

        # Compute real-time Ergo Score (0-100)
        session_good_ratio = self.good_evaluations / max(1, self.total_evaluations)
        ergo_score = self._compute_ergo_score(
            neck_angle=neck_val,
            torso_angle=torso_val,
            bad_duration_sec=self.bad_duration_sec,
            session_good_ratio=session_good_ratio,
            status=status,
        )

        return RuleEvaluationResult(
            status=status,
            ergo_score=round(ergo_score, 1),
            triggered_rule_id=triggered_rule_id,
            alert_message_th=alert_message_th,
            stretch_recommendation_th=stretch_recommendation_th,
            bad_duration_sec=bad_dur,
            countdown_sec=countdown,
            threshold_sec=self.threshold_sec,
            session_good_ratio=round(session_good_ratio, 3),
            rule_trace=rule_trace,
        )

    def _compute_ergo_score(
        self,
        neck_angle: float,
        torso_angle: float,
        bad_duration_sec: float,
        session_good_ratio: float,
        status: str,
    ) -> float:
        """
        Compute smooth real-time Ergo Score (0-100) from neck_angle, torso_angle,
        bad_duration_sec, and session_good_ratio:
        - ~88-100 when upright in Good Posture (PASS)
        - Gently glides in the yellow warning band (75-87.5) during MONITORING (< threshold_sec)
        - Drops into the red alert band (32-58) only once sustained >= threshold_sec (ALERT)
        """
        neck_excess = max(0.0, neck_angle - 14.0)
        torso_deficit = max(0.0, 90.0 - torso_angle)
        window_ratio = min(1.0, max(0.0, bad_duration_sec / max(1.0, self.threshold_sec)))

        if status == "PASS":
            angle_minor_penalty = (neck_excess * 0.6) + (torso_deficit * 0.5)
            raw_score = 97.5 - angle_minor_penalty + (session_good_ratio - 0.5) * 4.0
            target_score = float(min(100.0, max(88.0, raw_score)))
            self._smoothed_ergo_score = 0.55 * target_score + 0.45 * self._smoothed_ergo_score
            return float(min(100.0, max(88.0, self._smoothed_ergo_score)))

        if status == "MONITORING":
            severity_penalty = min(4.5, (neck_excess * 0.08) + (torso_deficit * 0.08))
            time_glide_penalty = window_ratio * 8.5
            target_score = 88.0 - severity_penalty - time_glide_penalty
            target_score = float(min(87.5, max(75.0, target_score)))
            self._smoothed_ergo_score = 0.60 * target_score + 0.40 * self._smoothed_ergo_score
            return float(min(87.5, max(75.0, self._smoothed_ergo_score)))

        # status == "ALERT" (sustained bad posture >= threshold_sec)
        angle_penalty = (neck_excess * 0.85) + (torso_deficit * 0.85)
        overtime_sec = max(0.0, bad_duration_sec - self.threshold_sec)
        overtime_penalty = min(14.0, overtime_sec * 0.6)
        target_score = 62.0 - min(18.0, angle_penalty * 0.45) - overtime_penalty
        target_score = float(min(58.0, max(32.0, target_score)))
        self._smoothed_ergo_score = 0.65 * target_score + 0.35 * self._smoothed_ergo_score
        return float(min(58.0, max(32.0, self._smoothed_ergo_score)))

