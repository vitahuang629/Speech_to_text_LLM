# Voice Record Assistant | 語音紀錄助理

## Purpose | 專案目標
This project is designed to assist **consultants**, **physical therapists**, and **physicians** in efficiently recording conversations with patients. It automatically extracts and structures key patient profiles and clinical insights during dialogues.

此專案旨在協助**諮詢師**、**物理治療師**與**醫師**，在與病人對話過程中有效紀錄對話內容，並自動整理病患的個人輪廓與重點資訊。

---

## Methodology | 技術方法

### 1. Data Preparation & Model Training | 資料準備與模型訓練
* **Audio Pre-processing:** Audio recordings are segmented into 30-second clips, manually corrected, and aligned with transcripts.
* **Model Fine-tuning:** These audio-text pairs are used to fine-tune **Azure Speech-to-Text**, incorporating domain-specific medical vocabulary to enhance recognition accuracy.

音檔經由 30 秒切割與人工校正後，作為訓練語音辨識模型（Azure Speech）的標記資料。同時收集醫療專業術語，以顯著提升特定領域的辨識準確度。

### 2. Summarization with LLM | 大語言模型摘要
* **Inference Pipeline:** Transcribed text is securely processed via a backend hosted on **AWS EC2**.
* **Model:** We utilize **OpenAI** models to first **refine raw transcripts** and then generate structured, concise summaries, ensuring key medical insights are captured.

完成語音轉文字後，內容將透過架設於 **AWS EC2** 的後端程式傳送至 **OpenAI** 模型，先進行文本潤飾再生成摘要，自動提取對話重點並產出結構化的病患資訊。

---

## Demo Screenshot | 系統展示
![系統架構圖](static/workflow.png)

---

## My Role | 我的貢獻
**I was responsible for the end-to-end backend architecture, including:**
**本人負責後端系統架構設計與全流程串接，包含：**

* **Real-time Infrastructure:** Implementing **WebSocket** servers for stable, low-latency audio streaming.
    * *實作低延遲即時語音串流的 WebSocket 通訊。*
* **Service Integration:** Integrating **Azure Speech-to-Text** API for high-fidelity transcription.
    * *整合 Azure 語音轉文字服務。*
* **Cloud Deployment:** Deploying the core engine on **AWS EC2** and orchestrating **OpenAI API** for efficient summarization.
    * *於 AWS 部署核心引擎並調度 OpenAI API 進行摘要處理。*
* **Frontend Prototyping:** Building a web-based demo interface to visualize the real-time transcription and summary pipeline.
    * *開發前端展示介面以視覺化呈現辨識與摘要流程。*

---

## Project Structure | 專案結構
* `aws_main.py`: Core logic on AWS EC2, ensuring session persistence and data integrity. (AWS 核心程式，負責處理語音並維持連線穩定)
* `main.py`: Main entry point managing multiple WebSocket instances. (專案入口點，調度多個服務實例)
* `consult_ws.py` & `doct_ws.py`: Dedicated WebSocket services for Consultants and Doctors. (諮詢師與醫師專用的 WebSocket 服務)
* `index.html`: Interactive frontend for simulation and testing. (前端測試頁面)

---

## Applications | 應用場景
* **Medical Consultation:** Automating clinical record-keeping. (醫療諮詢紀錄自動化)
* **Physiotherapy:** Summarizing session progress and patient feedback. (物理治療進度與回饋摘要)
* **Aesthetics Clinics:** Structuring customer beauty profiles and treatment history. (醫美診所客製化輪廓與療程紀錄)

---

## Author
**Vita Huang** Backend & AI Integration Specialist  
[GitHub @vitahuang629](https://github.com/vitahuang629)