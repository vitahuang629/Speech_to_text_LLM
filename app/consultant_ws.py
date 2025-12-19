import json
import azure.cognitiveservices.speech as speechsdk
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, APIRouter
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
from openai import AsyncOpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
speech_key = os.environ.get("SPEECH_KEY")
speech_region = os.environ.get("SPEECH_REGION")
endpoint_id = os.environ.get("ENDPOINT_ID")

router = APIRouter()

MAX_CONNECTIONS = 10
active_connections = set()

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
        # self.client = OpenAI(api_key=api_key)
        self.client = AsyncOpenAI(api_key=api_key)

    async def get_summary(self, text, system_prompt):
        messages = self.messages.copy()
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        try:
            response = await self.client.chat.completions.create(
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
                你是一個「逐字稿整理助手」。
                任務是：整理診所人員與客人的對話逐字稿，讓其他診所人員能快速掌握這次對談狀況。

                請嚴格遵守以下規範：

                1. 診所與客人對話的逐字稿是你唯一能使用的資料來源，若逐字稿沒有提到，禁止杜撰、推測、延伸。
                2. 必須使用繁體中文。
                3. 必須保留逐字稿中的個人化資訊（藥物名稱、症狀、療程、擔心的點、個性描述、身體狀態、興趣）。
                4. 條列式呈現，內容需具體、明確、保留細節。
                5. 語氣口語化但專業，讓國中生可理解。
                6. 禁止給出結論、建議、判斷，只能重述與整理既有事實。

輸出格式：
---
【潤飾過的逐字稿】:
 xxx
---
⚠️ 僅輸出潤飾後的逐字稿，不要加任何分析或分類。
    """
)

summarize_user_prompt = """
以下是對話逐字稿，請依 system prompt 規範整理：{segment_text}
根據以上資料請重新理解後，用你自己的方式重新寫一次，讓內容變得更清楚、更好懂，最好是國中生也能輕鬆看懂，若資料中有個人化資訊（藥名、病症、興趣、身體狀態），務必保留"""


# 系統提示
system_prompt = (
    """
你是一個「逐條分類助手」。
任務是：根據使用者提供的條列式內容，將內容分類到指定六種類型，並整理總結到「其他」區塊。

分類規則
分類類別與定義：
減重：任何關於瘦身、減肥、飲食控制、運動等話題
睡眠：任何關於睡眠、壓力、身心狀況、自律神經、內分泌等話題
疼痛：任何關於物理治療、身體部位疼痛等話題
私密處：任何關於性生活、私密處、親密關係等話題
臉部：任何關於臉部醫美、外表、皮膚狀況、醫美療程等話題
再生醫療：任何針劑、注射、紅光、粒線體、身體健康改善等體內治療話題
其他：整段對話的總整理，保留能對客人資訊更深入了解的內容，必須客觀、事實描述，不給建議

整理規範：
嚴格保留原文資訊，包括所有個人化資訊（藥物、療程、症狀、擔心的點等）
不能杜撰、推測、給建議或判斷
條列式呈現，清楚、具體
若某類別無相關內容，仍需顯示類別名稱，但內容留空

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

若無相關內容，則只顯示該類別名稱，內容留空
"""
)

system_user_prompt = """請依照 system prompt 規範，將以下條列式內容分類整理：
            {first_summary}"""      



@router.websocket("/ws/consultant")
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

    # 1. 延長靜音超時時間 (單位 ms)：避免使用者思考太久被切斷 (預設可能比較短)
    # 設定為 20 秒 (20000ms)，如果20秒沒聲音才視為斷句
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "20000")
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "20000")
    
    # 2. 如果你的 Azure 資源允許，可以嘗試降低Profanity(髒話)過濾等級，
    # 有時候含糊的聲音會被誤判為不雅字眼而被過濾掉，導致沒有輸出
    speech_config.set_profanity(speechsdk.ProfanityOption.Raw)

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
            first_summary = await llm.get_summary(text = summarize_user_prompt.format(segment_text = segment_text), system_prompt= summarize_prompt)    #潤飾逐字稿
            summary = await llm.get_summary(text = system_user_prompt.format(first_summary = first_summary), system_prompt = system_prompt) #0724      #分類摘要
            
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
                            segment_start_time = time.time()
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


    # 定義一個重啟函數，避免重複寫
    def restart_azure_recognizer(reason_msg):
        # 只有在 WebSocket 還活著的時候才重啟
        if websocket.client_state == websockets.CONNECTED:
            print(f"🔄 Azure 停止 ({reason_msg})，正在嘗試重啟...")
            try:
                # 先停再開，確保乾淨重啟
                recognizer.stop_continuous_recognition()
                recognizer.start_continuous_recognition()
                print("✅ Azure 重啟成功")
            except Exception as e:
                print(f"❌ Azure 重啟失敗: {e}")
        else:
            print("🛑 WebSocket 已斷線，Azure 正常停止")

    def on_session_stopped(evt):
        restart_azure_recognizer("Session Stopped")

    def on_canceled(evt):
        print(f"⚠️ Azure Canceled Details: {evt.result.reason}")
        if evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            print(f"❌ Canceled Reason: {cancellation_details.reason}")
            print(f"❌ Error Details: {cancellation_details.error_details}")
            
            # 如果是因為 Error 或 EndOfStream，嘗試重啟
            restart_azure_recognizer(f"Canceled: {cancellation_details.reason}")

    # 綁定事件
    recognizer.session_stopped.connect(on_session_stopped)
    recognizer.canceled.connect(on_canceled)




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
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("❓ 無法辨識 (NoMatch): 使用者可能含糊不清或噪音過大")
            # 這裡可以選擇要不要通知前端，通常後端紀錄就好

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
                    print(f"🔌 前端已斷線 (Code: {msg.get('code')})")
                    break           

                # 處理文字訊息 (包括 "stop" 指令)
                if msg.get("type") == "websocket.receive":
                    if "text" in msg:
                        text_data = msg["text"].strip()
                        print(f"📝 Received text message: {text_data}")

                        # 檢查是否為 "stop" 指令
                        if text_data == "stop":
                            print("🛑 Received 'stop' command")
                            print(f"🕒 Stop command received at: {time.time()}")
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
                        #  保活用 silence frame（100ms）
                        stream.write(b"\x00" * 3200)
                        # 這邊你可以選擇傳回前端錯誤訊息
                        await websocket.send_json({
                            "type": "error",
                            "message": "未檢測到語音，請確認麥克風是否有聲音"
                        })

            else:
                print(f"⚠️ 收到非 dict 的訊息: {msg}")
                break  #1204

    
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
            print("📝 中斷了跑出session_id：", session_id) #7/3
            
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
            # 接下來是處理異常，確保捕捉到前端中斷
    
    except WebSocketDisconnect:
        # 捕捉前端 close() 引發的中斷
        print(f"🔌 WebSocketDisconnect (Frontend Close) detected.")
        # 🚨 修正：印出中斷時的時間
        print(f"🕒 WebSocket disconnected at: {time.time()}")
        pass # 讓流程繼續到下方的 LLM 處理區塊

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
