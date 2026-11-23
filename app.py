"""
Module 4: app.py
ErgoGuard AI — Google Meet-Style Minimal Posture Monitor (CSC472)
Features:
- Full-focus centered camera stage (no sidebar, no walls of text)
- Top 1-line compact telemetry bar (Status, Score, Neck, Torso)
- Bottom Google Meet-style control dock:
    [เปิด/ปิดกล้อง] [ตั้งท่านั่งตรง] [เสียงเตือน] [ตั้งค่า] [สถิติ & กฎ]
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import DEFAULT_CSV_PATH, ensure_dataset_exists
from src.knn_classifier import PostureKNNClassifier
from src.pose_tracker import PoseMetrics, PoseTracker
from src.rule_engine import ErgonomicRuleEngine, RuleEvaluationResult


# ---------------------------------------------------------------------------
# Page Config & Google Meet-Style Minimal Dark CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ErgoGuard AI",
    page_icon="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/video.svg",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MEET_STYLE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600;700&family=IBM+Plex+Sans+Thai:wght@400;500;600&display=swap');

    :root {
        --bg-canvas: #111318;
        --bg-surface: #1C1F26;
        --bg-dock: #1E222B;
        --border-subtle: rgba(255, 255, 255, 0.09);
        --text-main: #F8FAFC;
        --text-sub: #94A3B8;
        --pass: #22C55E;
        --warn: #F59E0B;
        --alert: #EF4444;
        --font-ui: 'IBM Plex Sans Thai', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'Fira Code', monospace;
    }

    html, body, [class*="css"] {
        font-family: var(--font-ui) !important;
        background-color: var(--bg-canvas) !important;
        color: var(--text-main) !important;
    }

    /* Hide Sidebar & Streamlit Chrome completely */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    header[data-testid="stHeader"] {
        display: none !important;
    }

    .block-container {
        max-width: 920px !important;
        padding-top: 1.1rem !important;
        padding-bottom: 1.5rem !important;
    }

    /* Top Minimal Bar (Google Meet top bar style) */
    .meet-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 10px 18px;
        margin-bottom: 12px;
    }

    .meet-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
        font-size: 1.02rem;
        color: var(--text-main);
        white-space: nowrap;
    }

    .meet-metrics {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: nowrap;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 500;
        white-space: nowrap;
        background: #111318;
        border: 1px solid var(--border-subtle);
        color: var(--text-sub);
    }

    .pill strong {
        font-family: var(--font-mono);
        color: var(--text-main);
        font-weight: 600;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 13px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    .st-pass {
        background: rgba(34, 197, 94, 0.14);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .st-pass .dot { background: #22C55E; }

    .st-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .st-warn .dot { background: #F59E0B; }

    .st-alert {
        background: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
    }
    .st-alert .dot { background: #EF4444; }

    /* Video Viewport Rounding */
    [data-testid="stImage"] img {
        border-radius: 16px !important;
        border: 1px solid var(--border-subtle) !important;
        background: #161A22;
    }

    /* Compact 1-Line Toast Bar under Camera */
    .meet-toast {
        border-radius: 12px;
        padding: 10px 16px;
        font-size: 0.88rem;
        margin: 8px 0 12px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }

    .toast-neutral {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        color: var(--text-sub);
    }

    .toast-warn {
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #FDE68A;
    }

    .toast-alert {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.45);
        color: #FECACA;
    }

    /* Bottom Dock Buttons (Google Meet Pill Buttons) */
    div.stButton > button,
    div[data-testid="stPopover"] > div > button {
        border-radius: 999px !important;
        height: 44px !important;
        font-weight: 500 !important;
        font-size: 0.86rem !important;
        border: 1px solid var(--border-subtle) !important;
        transition: all 150ms ease !important;
    }
</style>
"""
st.markdown(MEET_STYLE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    ensure_dataset_exists(DEFAULT_CSV_PATH)

    if "knn_classifier" not in st.session_state:
        st.session_state.knn_classifier = PostureKNNClassifier(csv_path=DEFAULT_CSV_PATH, n_neighbors=5)

    if "rule_engine" not in st.session_state:
        st.session_state.rule_engine = ErgonomicRuleEngine(threshold_sec=15.0)

    if "pose_tracker" not in st.session_state:
        st.session_state.pose_tracker = PoseTracker(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = PoseMetrics(
            detected=True,
            neck_angle=11.5,
            torso_angle=94.5,
            shoulder_tilt=0.010,
            active_side="NONE",
            view_mode="STANDBY",
            visibility_score=1.0,
        )

    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    if "sound_on" not in st.session_state:
        st.session_state.sound_on = True

    if "camera_index" not in st.session_state:
        st.session_state.camera_index = 0

    if "alert_window_sec" not in st.session_state:
        st.session_state.alert_window_sec = 15

    if "last_sound_time" not in st.session_state:
        st.session_state.last_sound_time = 0.0

    if "toast_feedback" not in st.session_state:
        st.session_state.toast_feedback = ""


init_session_state()


def trigger_macos_alert_sound() -> None:
    """Play a non-blocking macOS system alert sound with an 8-second cooldown."""
    now = time.monotonic()
    if now - st.session_state.last_sound_time < 8.0:
        return
    st.session_state.last_sound_time = now
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage Placeholders (1. Top Bar -> 2. Camera Viewport -> 3. Toast -> 4. Dock)
# ---------------------------------------------------------------------------
top_bar_spot = st.empty()
video_spot = st.empty()
toast_spot = st.empty()

# ---------------------------------------------------------------------------
# Bottom Floating Control Dock (Google Meet Style)
# ---------------------------------------------------------------------------
dock_c1, dock_c2, dock_c3, dock_c4, dock_c5 = st.columns([1.25, 1.25, 1.1, 1.05, 1.15], gap="small")

with dock_c1:
    cam_btn_label = "⏹ ปิดกล้อง" if st.session_state.camera_on else "📹 เปิดกล้อง"
    cam_btn_type = "primary" if not st.session_state.camera_on else "secondary"
    if st.button(cam_btn_label, width="stretch", type=cam_btn_type):
        st.session_state.camera_on = not st.session_state.camera_on
        st.session_state.toast_feedback = ""
        st.rerun()

with dock_c2:
    if st.button("🎯 ตั้งท่านั่งตรง", width="stretch", help="นั่งหลังตรงแล้วกด 1 ครั้งเพื่อจำสัดส่วนร่างกายของคุณ"):
        cur_m: PoseMetrics = st.session_state.last_metrics
        if cur_m.view_mode == "STANDBY":
            st.session_state.toast_feedback = "กรุณากด '📹 เปิดกล้อง' ให้ระบบเห็นท่านั่งของคุณก่อนตั้งค่า"
        else:
            raw_n = getattr(cur_m, "raw_neck_angle", cur_m.neck_angle) or cur_m.neck_angle
            raw_t = getattr(cur_m, "raw_torso_angle", cur_m.torso_angle) or cur_m.torso_angle
            raw_s = getattr(cur_m, "raw_shoulder_tilt", cur_m.shoulder_tilt) or cur_m.shoulder_tilt
            st.session_state.pose_tracker.set_personal_baseline(raw_n, raw_t, raw_s)
            st.session_state.rule_engine.reset()
            st.session_state.toast_feedback = "บันทึกท่านั่งมาตรฐานตามสรีระของคุณเรียบร้อยแล้ว (จำถาวร)"

with dock_c3:
    snd_label = "🔊 เสียง: เปิด" if st.session_state.sound_on else "🔇 เสียง: ปิด"
    if st.button(snd_label, width="stretch"):
        st.session_state.sound_on = not st.session_state.sound_on
        st.rerun()

with dock_c4:
    with st.popover("⚙️ ตั้งค่า", width="stretch"):
        st.markdown("**ตั้งค่าการแจ้งเตือน & AI**")
        win_sec = st.slider(
            "เวลาแจ้งเตือนเมื่อนั่งผิดท่า (วินาที)",
            min_value=5,
            max_value=30,
            value=int(st.session_state.alert_window_sec),
            step=1,
        )
        st.session_state.alert_window_sec = win_sec
        st.session_state.rule_engine.set_threshold(float(win_sec))

        k_val = st.select_slider(
            "ความละเอียดการเทียบท่า K-NN (K)",
            options=[3, 5, 7],
            value=int(st.session_state.knn_classifier.n_neighbors),
        )
        st.session_state.knn_classifier.set_k(int(k_val))

        cam_idx = st.selectbox(
            "เลือกกล้อง (Camera)",
            options=[0, 1],
            index=int(st.session_state.camera_index),
        )
        st.session_state.camera_index = int(cam_idx)

        st.divider()
        st.caption("สอนท่าเพิ่มเข้าชุดข้อมูล (บันทึกถาวรลงไฟล์ CSV)")
        cur_m = st.session_state.last_metrics
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if st.button("+ ท่าปกติ", width="stretch"):
                if cur_m.view_mode == "STANDBY":
                    st.session_state.toast_feedback = "กรุณาเปิดกล้องให้จับภาพก่อนกดสอนท่า"
                else:
                    st.session_state.knn_classifier.add_calibration_sample(
                        cur_m.neck_angle, cur_m.torso_angle, cur_m.shoulder_tilt, "Good Posture", replicas=8
                    )
                    st.session_state.toast_feedback = f"บันทึก 'ท่าปกติ' (คอ {cur_m.neck_angle:.0f}°, หลัง {cur_m.torso_angle:.0f}°) ถาวรแล้ว"
            if st.button("+ ท่าหลังค่อม", width="stretch"):
                if cur_m.view_mode == "STANDBY":
                    st.session_state.toast_feedback = "กรุณาเปิดกล้องให้จับภาพก่อนกดสอนท่า"
                else:
                    st.session_state.knn_classifier.add_calibration_sample(
                        cur_m.neck_angle, cur_m.torso_angle, cur_m.shoulder_tilt, "Slouching", replicas=8
                    )
                    st.session_state.toast_feedback = f"บันทึก 'ท่าหลังค่อม' (คอ {cur_m.neck_angle:.0f}°, หลัง {cur_m.torso_angle:.0f}°) ถาวรแล้ว"
        with p_col2:
            if st.button("+ ท่าคอยื่น", width="stretch"):
                if cur_m.view_mode == "STANDBY":
                    st.session_state.toast_feedback = "กรุณาเปิดกล้องให้จับภาพก่อนกดสอนท่า"
                else:
                    st.session_state.knn_classifier.add_calibration_sample(
                        cur_m.neck_angle, cur_m.torso_angle, cur_m.shoulder_tilt, "Forward Head", replicas=8
                    )
                    st.session_state.toast_feedback = f"บันทึก 'ท่าคอยื่น/เอียงคอ' (คอ {cur_m.neck_angle:.0f}°, หลัง {cur_m.torso_angle:.0f}°) ถาวรแล้ว"
            if st.button("รีเซ็ตข้อมูล", width="stretch"):
                st.session_state.knn_classifier.reset_to_seed_dataset()
                st.session_state.pose_tracker.clear_personal_baseline()
                st.session_state.rule_engine.reset()
                st.session_state.toast_feedback = "รีเซ็ตข้อมูลและค่าสรีระกลับเป็นค่าเริ่มต้นแล้ว"

with dock_c5:
    stats_popover = st.popover("📊 สถิติ & กฎ", width="stretch")


# ---------------------------------------------------------------------------
# Helper Functions to Render Top Bar, Toast, and Stats Popover
# ---------------------------------------------------------------------------
def render_stage_ui(
    rgb_frame: np.ndarray,
    metrics: PoseMetrics,
    rule_res: RuleEvaluationResult,
    is_live: bool,
) -> None:
    status = rule_res.status
    if not is_live:
        st_cls = "st-pass"
        st_text = "พร้อมใช้งาน"
    elif status == "PASS":
        st_cls = "st-pass"
        st_text = "ท่านั่งปกติ"
    elif status == "MONITORING":
        st_cls = "st-warn"
        st_text = f"เฝ้าระวัง ({rule_res.countdown_sec:.0f}s)"
    else:
        st_cls = "st-alert"
        st_text = "แจ้งเตือนปรับท่านั่ง"

    with top_bar_spot.container():
        st.markdown(
            f"""
            <div class="meet-topbar">
                <div class="meet-brand">
                    <span style="color:#22C55E;">●</span>
                    <span>ErgoGuard AI</span>
                </div>
                <div class="meet-metrics">
                    <span class="status-pill {st_cls}"><span class="dot"></span>{st_text}</span>
                    <span class="pill">คะแนน <strong>{rule_res.ergo_score:.0f}</strong></span>
                    <span class="pill">คอ <strong>{metrics.neck_angle:.0f}°</strong></span>
                    <span class="pill">หลัง <strong>{metrics.torso_angle:.0f}°</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    video_spot.image(rgb_frame, channels="RGB", width="stretch")

    # Compact 1-line Toast bar below video
    if st.session_state.toast_feedback:
        toast_cls = "toast-neutral"
        toast_msg = f"✓ {st.session_state.toast_feedback}"
    elif not is_live:
        toast_cls = "toast-neutral"
        toast_msg = "กดปุ่ม '📹 เปิดกล้อง' ด้านล่างเพื่อเริ่มตรวจจับท่านั่งทำงาน"
    elif status == "PASS":
        cal_note = " (ปรับเทียบสรีระของคุณแล้ว)" if metrics.baseline_calibrated else ""
        toast_cls = "toast-neutral"
        toast_msg = f"ท่านั่งอยู่ในเกณฑ์ดี{cal_note} • สัดส่วนการนั่งถูกท่า {rule_res.session_good_ratio * 100:.0f}%"
    elif status == "MONITORING":
        toast_cls = "toast-warn"
        toast_msg = f"{rule_res.alert_message_th} (เหลือเวลา {rule_res.countdown_sec:.1f} วินาที)"
    else:
        toast_cls = "toast-alert"
        toast_msg = f"{rule_res.stretch_recommendation_th}"

    with toast_spot.container():
        st.markdown(
            f'<div class="meet-toast {toast_cls}"><span>{toast_msg}</span></div>',
            unsafe_allow_html=True,
        )


def render_stats_popover_content(
    metrics: PoseMetrics,
    neighbors_info: list[dict[str, Any]],
    rule_res: RuleEvaluationResult,
) -> None:
    with stats_popover:
        st.markdown("**1. กฎการตัดสินใจของระบบ (Rule-Based Expert System)**")
        rules_df = pd.DataFrame(rule_res.rule_trace)
        rules_df["สถานะ"] = rules_df["matched"].map(lambda x: "🟢 ทำงานอยู่" if x else "⚪ รอเงื่อนไข")
        rules_display = rules_df.rename(
            columns={
                "rule_id": "กฎ",
                "name": "หมวดหมู่",
                "condition": "เงื่อนไข (ภาษาคน)",
                "effect": "ผลลัพธ์",
            }
        )[["กฎ", "สถานะ", "เงื่อนไข (ภาษาคน)", "ผลลัพธ์"]]
        st.dataframe(rules_display, width="stretch", hide_index=True)

        st.markdown("**2. ตำแหน่งท่านั่งของคุณเทียบกับฐานข้อมูล K-NN**")
        df_all = st.session_state.knn_classifier.dataset_df
        color_map = {
            "Good Posture": "#22C55E",
            "Forward Head": "#F59E0B",
            "Slouching": "#EF4444",
        }
        fig_scatter = px.scatter(
            df_all,
            x="neck_angle",
            y="torso_angle",
            color="label",
            color_discrete_map=color_map,
            opacity=0.65,
            labels={
                "neck_angle": "มุมก้มคอ (°)",
                "torso_angle": "มุมลำตัว (°)",
                "label": "กลุ่มท่านั่ง",
            },
        )
        fig_scatter.add_vline(x=20.0, line_dash="dot", line_color="#64748B")
        fig_scatter.add_hline(y=78.0, line_dash="dot", line_color="#64748B")
        fig_scatter.add_trace(
            go.Scatter(
                x=[metrics.neck_angle],
                y=[metrics.torso_angle],
                mode="markers",
                marker=dict(size=14, symbol="diamond", color="#38BDF8", line=dict(width=2, color="#FFFFFF")),
                name="ตำแหน่งของคุณ",
            )
        )
        # Place legend cleanly below the chart so it NEVER overlaps title or points
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1C1F26",
            plot_bgcolor="#111318",
            height=290,
            margin=dict(l=15, r=15, t=20, b=65),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                title_text="",
            ),
        )
        st.plotly_chart(fig_scatter, width="stretch")

        st.markdown("**3. ข้อมูลอ้างอิงที่ใกล้เคียงที่สุด 5 ลำดับ (K-NN)**")
        neighbors_df = pd.DataFrame(neighbors_info).rename(
            columns={
                "rank": "ลำดับ",
                "label": "ท่านั่ง",
                "neck_angle": "มุมคอ (°)",
                "torso_angle": "มุมหลัง (°)",
                "euclidean_distance": "ระยะห่าง (Distance)",
            }
        )[["ลำดับ", "ท่านั่ง", "มุมคอ (°)", "มุมหลัง (°)", "ระยะห่าง (Distance)"]]
        st.dataframe(neighbors_df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Main Execution: Standby View vs. Live Webcam Stream
# ---------------------------------------------------------------------------
last_m: PoseMetrics = st.session_state.last_metrics
pred_label, conf, dists = st.session_state.knn_classifier.predict(
    last_m.neck_angle, last_m.torso_angle, last_m.shoulder_tilt
)
neighbors_info = st.session_state.knn_classifier.explain_neighbors(
    last_m.neck_angle, last_m.torso_angle, last_m.shoulder_tilt
)
init_rule_res = st.session_state.rule_engine.evaluate(
    knn_label=pred_label,
    neck_angle=last_m.neck_angle,
    torso_angle=last_m.torso_angle,
    shoulder_tilt=last_m.shoulder_tilt,
    override_bad_duration_sec=0.0,
)
render_stats_popover_content(last_m, neighbors_info, init_rule_res)

if not st.session_state.camera_on:
    standby_bgr = st.session_state.pose_tracker.render_camera_standby_frame()
    standby_rgb = cv2.cvtColor(standby_bgr, cv2.COLOR_BGR2RGB)
    render_stage_ui(
        rgb_frame=standby_rgb,
        metrics=last_m,
        rule_res=init_rule_res,
        is_live=False,
    )
else:
    cap = cv2.VideoCapture(int(st.session_state.camera_index))
    if not cap.isOpened():
        st.session_state.camera_on = False
        st.error("ไม่สามารถเปิดกล้องเว็บแคมได้ กรุณาตรวจสอบสิทธิ์การใช้กล้องบน macOS")
    else:
        try:
            while st.session_state.camera_on and cap.isOpened():
                ok, bgr_frame = cap.read()
                if not ok:
                    break

                prev_status = (
                    "ALERT"
                    if st.session_state.rule_engine.bad_duration_sec >= st.session_state.alert_window_sec
                    else (
                        "MONITORING"
                        if st.session_state.rule_engine.bad_duration_sec > 0
                        else "PASS"
                    )
                )

                annotated_bgr, live_metrics = st.session_state.pose_tracker.process_frame(
                    bgr_frame,
                    mirror=True,
                    status=prev_status,
                )

                if live_metrics.detected:
                    st.session_state.last_metrics = live_metrics
                    pred_label, conf, dists = st.session_state.knn_classifier.predict(
                        live_metrics.neck_angle,
                        live_metrics.torso_angle,
                        live_metrics.shoulder_tilt,
                    )
                    rule_res = st.session_state.rule_engine.evaluate(
                        knn_label=pred_label,
                        neck_angle=live_metrics.neck_angle,
                        torso_angle=live_metrics.torso_angle,
                        shoulder_tilt=live_metrics.shoulder_tilt,
                    )

                    if rule_res.status == "ALERT" and st.session_state.sound_on:
                        trigger_macos_alert_sound()

                    rgb_display = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    render_stage_ui(
                        rgb_frame=rgb_display,
                        metrics=live_metrics,
                        rule_res=rule_res,
                        is_live=True,
                    )
                else:
                    rgb_display = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    video_spot.image(rgb_display, channels="RGB", width="stretch")

                time.sleep(0.03)
        finally:
            cap.release()
