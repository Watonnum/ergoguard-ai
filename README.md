# ErgoGuard AI — ระบบตรวจจับท่านั่งทำงานและป้องกันออฟฟิศซินโดรมอัจฉริยะ

เว็บแอปพลิเคชันตรวจจับและวิเคราะห์ท่านั่งทำงานแบบเรียลไทม์ผ่านกล้องเว็บแคม สำหรับรายวิชา **CSC472 Artificial Intelligence** พัฒนาด้วยสถาปัตยกรรมที่ผสมผสาน **Computer Vision (MediaPipe BlazePose Heavy)**, **Machine Learning (K-Nearest Neighbors & Random Forest Spine Keypoints Regressor)** และ **Rule-Based Expert System (Forward Chaining)** ในอินเทอร์เฟซแบบมินิมอลสไตล์ Google Meet

---

## Architecture & Tech Stack

| Layer / Component | Technology / Library | บทบาทหน้าที่ในระบบ |
| :--- | :--- | :--- |
| **Frontend & UI** | `Streamlit` + Custom CSS (`IBM Plex Sans Thai`, `Fira Code`) | หน้าจอแสดงผลแบบ Google Meet-Style Minimal Stage (จอกล้องตรงกลาง + แถบสถานะด้านบน + แถบปุ่มควบคุมด้านล่าง + Popover ตั้งค่าและสถิติ) |
| **Computer Vision** | `OpenCV` + `MediaPipe PoseLandmarker Heavy` (`pose_landmarker_heavy.task`) | จับภาพจากกล้องเว็บแคม สกัดพิกัดข้อต่อร่างกายส่วนบน 33 จุด พร้อมระบบ **Nose-Between-Shoulders Auto View Detection** แยกแยะมุมหน้าตรงและด้านข้างอัตโนมัติ |
| **Spine Keypoints Model** | `Scikit-Learn` (`RandomForestRegressor`) + Kaggle `melsmm/posture-keypoints-detection` | วาดและติดตามจุดกระดูกสันหลัง 4 ส่วน (`CERVICAL`, `THORACIC`, `LUMBAR`, `SACRAL SPINE`) พร้อมกรอบ Bounding Box อ้างอิงจากชุดข้อมูลภาพจริง 300 ภาพ |
| **Posture Classifier (ML)** | `Scikit-Learn` (`KNeighborsClassifier` + `StandardScaler` + `ErgonomicFeatureWeighter`) | จำแนกท่านั่งด้วยระยะห่าง **Euclidean Distance** ออกเป็น 3 คลาส: `Good Posture`, `Forward Head`, `Slouching` |
| **Expert System (AI)** | Custom **Forward-Chaining Rule Engine** (`src/rule_engine.py`) | ประเมินกฎ `IF-THEN` (`R1`–`R4`) ร่วมกับหน้าต่างเวลา **Sliding Time Window (5–30 วินาที)** เพื่อกรองการขยับตัวชั่วคราว คำนวณคะแนน **Ergo Score (0–100)** และแนะนำท่ายืดเหยียด |
| **Data & Visualization** | `Pandas`, `NumPy`, `Plotly` | จัดการชุดข้อมูล `data/posture_dataset.csv` และแสดงกราฟ Feature Space รวมถึงตาราง Explainable AI (XAI) |
| **Testing** | `Pytest` | ชุดทดสอบอัตโนมัติ 9 รายการครอบคลุม Angle Math, Auto-Fallback, K-NN, Rule Engine R1–R4 และ Pipeline (`tests/test_ergoguard.py`) |

---

## โครงสร้างโปรเจกต์ (Project Structure)

```text
ergoGuard_AI/
├── app.py                              # หน้าเว็บหลัก Streamlit (Google Meet-Style UI)
├── requirements.txt                    # รายการไลบรารี Python ที่ต้องติดตั้ง
├── SPEC.md                             # เอกสารข้อกำหนดของโปรเจกต์
├── data/
│   ├── posture_dataset.csv             # ชุดข้อมูลท่านั่ง 300 ตัวอย่างจาก Kaggle + ข้อมูลที่ผู้ใช้สอนเพิ่ม
│   ├── pose_landmarker_heavy.task      # โมเดล MediaPipe BlazePose Heavy (Offline)
│   └── kaggle_spine_regressor.joblib   # โมเดลทำนายจุดกระดูกสันหลัง 4 จุด
├── src/
│   ├── pose_tracker.py                 # Module 1: Webcam + MediaPipe + คำนวณมุม + วาด Spine 4 จุด
│   ├── knn_classifier.py               # Module 2: K-NN Classifier + Live Calibration
│   ├── rule_engine.py                  # Module 3: Forward-Chaining Rules (R1–R4) + Ergo Score
│   └── data_generator.py               # ตัวสกัดและสร้างชุดข้อมูล posture_dataset.csv
└── tests/
    └── test_ergoguard.py               # ชุดทดสอบ Pytest
```

---

## วิธีติดตั้งและเริ่มต้นใช้งาน (Installation & Setup)

> **ข้อกำหนดระบบ**: macOS / Windows / Linux ที่มีกล้องเว็บแคม และ **Python 3.10 – 3.13**

1. **โคลนโปรเจกต์จาก GitHub**
   ```bash
   git clone https://github.com/Watonnum/ergoguard-ai.git
   cd ergoguard-ai
   ```

2. **สร้าง Virtual Environment และติดตั้ง Dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # สำหรับ Windows ใช้: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **รันทดสอบระบบ (ไม่บังคับ)**
   ```bash
   pytest tests/test_ergoguard.py -v
   ```

4. **เปิดใช้งานเว็บแอปพลิเคชัน**
   ```bash
   streamlit run app.py
   ```
   จากนั้นเปิดเบราว์เซอร์ไปที่ **`http://localhost:8501`** (และกดอนุญาตสิทธิ์การใช้กล้องเว็บแคมเมื่อเบราว์เซอร์หรือ macOS ถาม)

---

## วิธีใช้งานหน้าเว็บ (User Guide)

แถบปุ่มควบคุมด้านล่างหน้าจอออกแบบให้ใช้งานง่ายใน 1 คลิก:

1. **`📹 เปิดกล้อง` / `⏹ ปิดกล้อง`**: เริ่มหรือหยุดการตรวจจับท่านั่งแบบเรียลไทม์
2. **`🎯 ตั้งท่านั่งตรง`**: นั่งหลังตรงตามธรรมชาติของสรีระคุณแล้วกดปุ่มนี้ 1 ครั้ง ระบบจะบันทึกสัดส่วนร่างกายของคุณลงไฟล์ `data/personal_baseline.json` แบบถาวร เพื่อให้รองรับความแตกต่างของสรีระและมุมกล้อง
3. **`🔊 เสียง: เปิด` / `🔇 เสียง: ปิด`**: เปิด/ปิดเสียงแจ้งเตือนเมื่อนั่งผิดท่าต่อเนื่องเกินเวลาที่กำหนด
4. **`⚙️ ตั้งค่า`**:
   - ปรับ **เวลาแจ้งเตือน (5–30 วินาที)** (แนะนำตั้งเป็น `5 วินาที` เมื่อสาธิตหน้าชั้นเรียน)
   - ปรับความละเอียด **K-NN (`K = 3, 5, 7`)**
   - สอนท่าเพิ่มเข้าชุดข้อมูลถาวร (`+ ท่าปกติ`, `+ ท่าคอยื่น`, `+ ท่าหลังค่อม`) หรือกด `รีเซ็ตข้อมูล` กลับสู่ค่าเริ่มต้น
5. **`📊 สถิติ & กฎ`**: เปิดดูตารางการทำงานของกฎ `R1–R4` (ภาษาไทยเข้าใจง่าย), กราฟการกระจายตัวของข้อมูล K-NN และตำแหน่งท่านั่งปัจจุบันของคุณ

---

## หมายเหตุเกี่ยวกับการ Deploy (Cloud / Vercel)

โปรเจกต์นี้พัฒนาด้วย **Streamlit + OpenCV (`cv2.VideoCapture`) + MediaPipe** ซึ่งต้องใช้ Persistent WebSocket Server และเข้าถึงกล้องเว็บแคมของเครื่องผู้ใช้งานโดยตรง จึง **ไม่สามารถ Deploy ขึ้น Vercel** (ซึ่งรองรับเฉพาะ Static Site และ Stateless HTTP Serverless Functions ที่มีขนาด Bundle จำกัดไม่เกิน 250 MB) ได้โดยตรง แนะนำให้รันแบบ Local (`streamlit run app.py`) หรือปรับใช้ **Streamlit Community Cloud** ร่วมกับ `streamlit-webrtc` หากต้องการใช้งานผ่านอินเทอร์เน็ต
