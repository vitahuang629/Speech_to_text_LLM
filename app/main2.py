import json
import azure.cognitiveservices.speech as speechsdk
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import requests
import os
import time
from datetime import datetime
from collections import defaultdict
from asyncio import Lock
import re
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, timezone
from fastapi.responses import JSONResponse
from openai import OpenAI
from starlette import websockets

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
speech_key = os.environ.get("SPEECH_KEY")
speech_region = os.environ.get("SPEECH_REGION")
endpoint_id = os.environ.get("ENDPOINT_ID")

import os

app = FastAPI()
# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], #允許所有來源，如要限制請換成 ["http://localhost:3000"] 等
    allow_credentials = False, #跨域憑證
    allow_methods = ["*"], #允許所有HTTP 方法
    allow_headers = ["*"], #允許所有HTTP 標頭
)
app.mount("/static", StaticFiles(directory="static"), name="static") ############################html

# 用於儲存最終辨識結果的全局變數
final_results = defaultdict(str) #避免打架
summary_results = defaultdict(str)
result_lock = Lock()
active_sessions = {} #儲存 session_id: websocket 7/3
last_access_time = {} #key: session_id, value:datetime

clean_interval = 480 #秒
data_ttl = 1200 #存活時間(20分鐘)


class OpenAILLM:
    def __init__(self, model, api_key, temperature=0.7, messages=None):
        self.model = model
        self.temperature = temperature
        self.messages = messages or []
        self.client = OpenAI(api_key=api_key)

    def get_summary(self, text, system_prompt):
        messages = self.messages.copy()
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API請求錯誤: {e}")
            return f"獲取摘要時出錯: {str(e)}"


# 初始化LLM
llm = OpenAILLM(
    # model="gpt-3.5-turbo",
    model="gpt-4o-mini",
    # api_key=os.getenv("OPENAI_API_KEY"),
    api_key = OPENAI_API_KEY,
    temperature=0.7,
    messages=[{"role": "system", "content": "你是一位專業的醫美諮詢紀錄分析員"}]
)

summarize_prompt = (
    """
你是對話潤飾助手。

任務：
1. 將逐字稿中因語音辨識錯誤或口語化、口吃、重複、斷裂造成的非通順語句進行語句修正與語意整理，使對話內容清晰、自然、流暢。
2. 保留逐句對話結構與原始順序。
3. 不刪掉任何對話內容，只修正語句通順。
4. 修正後的語句要自然、完整，可讀性高，但不要加入摘要、評論或額外訊息。
5. 保留口語化特色，但避免不必要的重複、斷句不完整、錯誤用詞。
6. 適當將不完整或語意模糊的句子補成完整句。

輸出格式：
---
【潤飾過的逐字稿】:
 xxx
---
⚠️ 僅輸出潤飾後的逐字稿，不要加任何分析或分類。
    """
)

# 系統提示
system_prompt = (
    """
你是對話分析助手，請根據以下規則，客觀整理逐字稿內容。

任務：
1. 將對話內容依照指定分類，整理為條列重點摘要，幫助後續使用者快速了解談話重點。

🗣 本內容為一段逐字對話紀錄，內容包含兩個角色的發言。  
請根據實際回覆內容，抽取對話中明確表達的**實際行為、經驗、需求或觀察**，重新整理為條列重點。

⚠️ 注意：
1. 只根據逐字稿內容，產出客觀、清楚、無腦補的紀錄摘要。
2. 每個分類下的內容必須是經過重組的客觀事實描述，**可以包含逐字稿中的細節**，但必須用完整、通順的語句表達。
3. **用詞準確、語氣中立**，避免醫學化名詞與情緒化語氣。
4. 必須一次輸出 **全部 8 個分類**，**不可遺漏**，包括【減重】、【睡眠】、【疼痛】、【私密處】、【臉部】、【再生醫療】、【其他】。
5. 如對話中未提及該分類，該分類內容請留空。
6.【其他】必須獨立列出，統整整段對話的重點，並簡述與患者相關的主要內容。

【輸出格式範例】
---
【減重】:
- xxx
【睡眠】:
- xxx
【疼痛】:
- xxx
【私密處】:
- xxx
【臉部】:
- xxx
【再生醫療】:
- xxx
【其他】:
- xxx
---

【分類規則（僅供參考，禁止輸出）】
以下分類定義是提供你整理資訊時使用，請勿將這些文字、說明或分類定義一併輸出。

- 減重：提到體重、瘦身、減肥、飲食。
- 睡眠：提到睡眠、失眠、安眠藥。
- 疼痛：提到疼痛、痠痛、姿勢不良等不適。
- 私密處：提到私密處、頻尿、親密關係、私秘保養。
- 臉部：提到臉、皺紋、保養、法令紋、皮膚。
- 再生醫療：提到注射、TRT、靜脈雷射、NMN。
- 其他：必須有，總結整體談話重點。

  ❗禁止捏造、推測或擴寫逐字稿中未提及的資訊，例如：「應該是壓力大導致失眠」這類判斷性語句不可出現。所有資訊皆須直接來自逐字稿內容，且不可有臆測、揣測或合理推論。

"""
)

doctor_system_prompt = ("""
你是專業的醫療對話分析助手，負責整理醫師與客人之間的逐字稿。  
請根據提供的逐字稿產出**結構化摘要**，不要編造逐字稿中不存在的資訊。
                        
請用以下格式輸出:
                        
                        1.客人主訴
                        2.診斷
                        3.建議

 ❗禁止捏造、推測或擴寫逐字稿中未提及的資訊。
""")

@app.get("/")   #如果不使用前端的話
async def index():
    return HTMLResponse(open("static/index.html", "r", encoding="utf-8").read())

@app.get("/")
async def index():
    return {"message": "Backend is running."}


MAX_CONNECTIONS = 10
active_connections = set()


@app.websocket("/ws/consultant")
async def websocket_endpoint(websocket: WebSocket):

    # 安全發送函數
    async def safe_send(ws: WebSocket, data):
        """在 WebSocket 關閉時安全發送訊息"""
        try:
            await ws.send_text(data)
        except RuntimeError as e:
            print(f"⚠️ websocket 已關閉，無法送訊息: {e}")
        except Exception as e:
            print(f"⚠️ 發送訊息時發生其他錯誤: {e}")

    print(f"📡 新連線來自：{websocket.client.host}:{websocket.client.port}")
    
    if len(active_connections) >= MAX_CONNECTIONS:
        await websocket.close(code=1008, reason="Connection limit exceeded")
        print(f"🔌 連線被拒絕：已達人數上限 {MAX_CONNECTIONS}")
        return
    await websocket.accept()
    await websocket.send_text("🔊 新連線來了")
    active_connections.add(websocket) # <--- 加入集合
    await websocket.send_text("🔊 已連線語音辨識 WebSocket")
###################################################################7/3
    import uuid
    session_id = str(uuid.uuid4()) #為這次對話建立唯一ID
    active_sessions[session_id] = websocket #綁定 7/3

    final_transcripts = []         #所有語音辨識結果(逐字稿)
    segment_transcripts = []       #當前10分鐘的分段結果
    all_segment_summaries = []     #每段的LLM摘要
    all_refined_transcripts = []   #每段的潤飾摘要


    segment_start_time = time.time() #計算每段開始時間
    segment_duration_limit = 480  # 8 分鐘
    last_audio_time = time.time() #最後收到音訊的時間

    loop = asyncio.get_event_loop()

    # Speech config
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "zh-TW"
    speech_config.endpoint_id = endpoint_id
    speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationStrategy, "Semantic")

    stream = speechsdk.audio.PushAudioInputStream()
    audio_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # 回傳 session_id 給前端 7/3
    await websocket.send_json({"type": "session_id", "session_id": session_id}) 

    # 分段摘要函數也需要加強驗證
    async def process_segment_for_summary(transcripts_to_summarize):
        if not transcripts_to_summarize or not any(t.strip() for t in transcripts_to_summarize):
            return ""
        
        segment_text = "\n".join(transcripts_to_summarize)
        if not segment_text:
            return ""
        try:
            first_summary = llm.get_summary(segment_text, summarize_prompt)    #潤飾逐字稿
            summary = llm.get_summary(first_summary, system_prompt) #0724      #分類摘要
            
            # 驗證摘要格式是否正確
            required_categories = ["減重", "睡眠", "疼痛", "私密處", "臉部", "再生醫療", "其他"]  #ˇ0807拿掉摘要
            for category in required_categories:
                if f"【{category}】:" not in summary:
                    print(f"⚠️ 警告：摘要中缺少 【{category}】: 分類")
            
            print("🧾 LLM回傳潤飾摘要原文：", first_summary)
            return {
                "refined_transcript": first_summary,
                "categorized_summary": summary
            }
            
        except Exception as e:
            print(f"獲取分段LLM摘要時出錯: {e}")
            return f"獲取分段摘要時出錯: {str(e)}"

    # def merge_summaries_by_category(summaries: list[str]) -> str: 改成json 7/9
    def merge_summaries_by_category(summaries: list[str]) -> dict:
        """改進版的分段合併函數"""
        categories = [
            "減重", "睡眠", "疼痛", "私密處", "臉部", "再生醫療", "其他"
        ]
        merged = defaultdict(list)
        
        # print('summaries', summaries)
        
        # regex：從【分類】開始，直到下一個【分類】或結尾
        pattern = rf"(?:【)?({'|'.join(categories)})(?:】)?[:：]\s*([\s\S]*?)(?=(?:【)?(?:{'|'.join(categories)})(?:】)?[:：]|$)"

        for summary in summaries:
            if not summary or not isinstance(summary, str):
                continue  # 跳過不是字串的內容

            matches = re.findall(pattern, summary, re.DOTALL)

            for category, content in matches:
                content = content.strip()
                lines = [line.strip() for line in content.split("\n") if line.strip()]
                for line in lines:
                    if not any(keyword in line for keyword in ["未提及", "無相關", "無", "未", "x"]):
                        if not line.startswith("- "):
                            line = "- " + line
                        if line not in merged[category]:
                            merged[category].append(line)


        # 生成最終結果
        # result = []
        result_dict = {}
        for category in categories:
            if merged.get(category):    #0721
                combined = "\n".join(merged[category])    #0721
            else:    #0721
                combined = ""  # 這裡改為空字串     #0721
            result_dict[category] = combined


        return result_dict
    
    # 分段摘要定時器 (背景定時任務)
    async def segment_timer_task():
        nonlocal segment_start_time, segment_transcripts, all_segment_summaries, all_refined_transcripts
        try:
            print('start segment_timer_task')
            # 一開始就潤稿一次（如果有錄音）
            if segment_transcripts:
                result = await process_segment_for_summary(segment_transcripts)
                all_segment_summaries.append(result["categorized_summary"])
                all_refined_transcripts.append(result["refined_transcript"])
                segment_transcripts.clear()
            # 進入定時檢查迴圈 (錄音時間超過限制)
            while True:
                await asyncio.sleep(10)
                now = time.time()
                if now - segment_start_time >= segment_duration_limit:
                    if segment_transcripts:
                        #result = await loop.run_in_executor(None, lambda: llm.get_summary("\n".join(segment_transcripts), summarize_prompt))
                        result = await process_segment_for_summary(segment_transcripts)
                        all_segment_summaries.append(result["categorized_summary"])
                        all_refined_transcripts.append(result["refined_transcript"])
                        await safe_send(websocket, json.dumps({
                            "type": "segment_summary",
                            "summary": result["categorized_summary"]
                        }))
                        segment_transcripts.clear()
                    segment_start_time = now

        except asyncio.CancelledError:
            print("⏹ 分段任務已取消")

    #######################################加入靜音偵測

    async def check_silence():
        """背景任務：檢查是否超過10秒沒收到音訊"""
        nonlocal last_audio_time, segment_transcripts, all_segment_summaries, all_refined_transcripts, segment_start_time

        try:
            print('start check silence')
            while True:
                await asyncio.sleep(1)  # 每秒檢查一次
                if time.time() - last_audio_time > 10:  # 超過 10 秒沒收到音訊
                    print("⚠️ 超過 10 秒未收到音訊")
                    await websocket.send_json({
                        "type": "error",
                        "message": "未檢測到語音，請確認麥克風是否有聲音"
                    })
                    if segment_transcripts and websocket.client_state == websockets.CONNECTED:  # ✅ flush 當前累積的文字
                        try: 
                            result = await process_segment_for_summary(segment_transcripts)
                            all_segment_summaries.append(result["categorized_summary"])
                            all_refined_transcripts.append(result["refined_transcript"])
                            await safe_send(websocket, json.dumps({
                                "type": "segment_summary",
                                "summary": result["categorized_summary"]
                            }))
                            segment_transcripts.clear()
                            #flush後重置時間
                            # segment_start_time = time.time()
                        except Exception as e:
                            print(f"⚠️ 靜音 flush 出錯: {e}")
                    
                    # 發送提示訊息給前端
                    try: 
                        await websocket.send_json({
                        "type": "error",
                        "message": "未檢測到語音，請確認麥克風是否有聲音"
                    })
                    except Exception as e:
                        print(f"⚠️ 發送靜音提示失敗: {e}")

        except asyncio.CancelledError:
            print("⏹ 靜音檢查任務已取消")

        # 接收辨識結果
    def recognized_callback(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text
            final_transcripts.append(text)
            segment_transcripts.append(text)
            # asyncio.run_coroutine_threadsafe(websocket.send_text(text), loop)
            asyncio.run_coroutine_threadsafe(
                safe_send(websocket, text),
                loop
            )

    recognizer.recognized.connect(recognized_callback)
    recognizer.start_continuous_recognition()

    await websocket.send_text(json.dumps({
        "type": "session_id",
        "session_id": session_id
    }))


    timer_task = asyncio.create_task(segment_timer_task())
    silence_task = asyncio.create_task(check_silence())

    try:
        while True:
            msg = await websocket.receive()   #0723
            # print("📩 raw message:", msg, flush=True)
            if isinstance(msg, dict):
                if msg.get("type") == "websocket.disconnect":
                    print('sssssssssssssss websocket disconnect')
                    break           

                # 處理文字訊息 (包括 "stop" 指令)
                if msg.get("type") == "websocket.receive":
                    if "text" in msg:
                        text_data = msg["text"].strip()
                        print(f"📝 Received text message: {text_data}")

                        # 檢查是否為 "stop" 指令
                        if text_data == "stop":
                            print("🛑 Received 'stop' command")
                            break

                        # 如果是 JSON 指令，可以這樣解析：
                        try:
                            json_data = json.loads(text_data)
                            if json_data.get("command") == "stop":
                                print("🛑 Received JSON stop command")
                                break
                        except json.JSONDecodeError:
                            pass  # 非 JSON 訊息，忽略

                    # 處理二進位數據 (音訊)
                    elif "bytes" in msg:
                        if msg["bytes"]:
                            stream.write(msg["bytes"])
                            last_audio_time = time.time()
                    else: 
                        print("⚠️ 收到空音訊，未檢測到聲音")
                        # 這邊你可以選擇傳回前端錯誤訊息
                        await websocket.send_json({
                            "type": "error",
                            "message": "未檢測到語音，請確認麥克風是否有聲音"
                        })

            else:
                print(f"⚠️ 收到非 dict 的訊息: {msg}")


        recognizer.stop_continuous_recognition()
        stream.close()
        timer_task.cancel()
        silence_task.cancel()

        if final_transcripts: #有轉出文字才做後續
            # ✨ 最後一段補摘要
            if segment_transcripts:
                result = await process_segment_for_summary(segment_transcripts)
                all_segment_summaries.append(result["categorized_summary"])  #只存分類摘要
                all_refined_transcripts.append(result["refined_transcript"])
                segment_transcripts.clear()

            # 組合最終摘要與逐字稿
            #final_combined_summary = "\n\n".join(all_segment_summaries)
            final_refined_transcript = "\n".join(all_refined_transcripts).strip()
            final_combined_summary = merge_summaries_by_category(all_segment_summaries) #6/17 同類別合併
            patterns_to_remove = [
            '- 患者提到臉的皺紋、保養、法令紋。',
            '- 患者提到體重、瘦身需求。',
            '- 患者提到體重、瘦身、減肥、飲食。',
            '- 患者提到私秘保養。',
            '- 患者提到注射、TRT、靜脈雷射、NMN。',
            '- 患者提到注射治療、TRT、靜脈雷射、NMN等。',
            '- 患者提到注射、TRT、靜脈雷射和NMN。'
            '- 患者提到私密處保養。',
            '- 患者提到臉、皺紋、保養、法令紋和皮膚。',
            '- 患者提到睡眠、失眠、安眠藥。',
            '- 患者提到私密處、頻尿、親密關係、私秘保養。',
            '- 患者提到頻尿。',
            '- 患者提到失眠、安眠藥。',
            '- 患者提到睡眠、失眠和安眠藥。'
            '- 患者提到頻尿、親密關係、私秘保養。',
            '- 患者提到臉、皺紋、保養、法令紋、皮膚。',
            '- 患者提到臉、皺紋、保養等。'
            ]

            # 遍歷字典的每個 key
            for key, value in final_combined_summary.items():
                for pattern in patterns_to_remove:
                    if pattern in value:
                        value = value.replace(pattern, "")
                # 清理多餘換行 & 空白
                final_combined_summary[key] = "\n".join(
                    [line for line in value.splitlines() if line.strip()]
                ).strip()
            summary_results[session_id] = final_combined_summary
            final_text = "\n".join(final_transcripts)
            final_results[session_id] = final_text

            print("📝 最終諮詢師合併摘要：", final_combined_summary)
            print("📝 最終諮詢師完整逐字稿：", final_text)
            print("📝 最終諮詢師完整潤飾稿：", final_refined_transcript)
            print("📝 session_id：", session_id) #7/3
            
            # 在這裡傳送最終摘要完成通知 7/23
            await websocket.send_text(json.dumps({
                "type": "final_summary_ready",
                "session_id": session_id
            }))
            #傳送合併潤飾搞
            await websocket.send_text(json.dumps({
                "type": "final_refined_transcript",
                "session_id": session_id,
                "refined_transcript": final_refined_transcript
            }))
            # 最終完整逐字稿
            await websocket.send_text(json.dumps({
                "type": "final_combined_text",
                "session_id": session_id,
                "summary": final_text
            }))
            # 最終摘要資料也送過去
            await websocket.send_text(json.dumps({
                "type": "final_combined_summary",
                "session_id": session_id,
                "summary": final_combined_summary
            }))
            return final_combined_summary



    except Exception as e:
        print(f"處理語音資料時出錯: {e}")
        recognizer.stop_continuous_recognition()
        stream.close()
        timer_task.cancel()
        try:
            await timer_task
        except asyncio.CancelledError:
            pass
    
    finally: # <--- 使用 finally 確保無論如何都會執行
        # 確保資源被清理
        if 'timer_task' in locals() and not timer_task.done():
            timer_task.cancel()
            try:
                await timer_task
            except asyncio.CancelledError:
                pass
        
        if websocket in active_connections:
            active_connections.remove(websocket)
        
        active_sessions.pop(session_id, None)
        print(f"🔌 諮詢師連線已關閉 (Session: {session_id})，目前連線數: {len(active_connections)}")


