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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
speech_key = os.environ.get("SPEECH_KEY")
speech_region = os.environ.get("SPEECH_REGION")
endpoint_id = os.environ.get("ENDPOINT_ID")


router = APIRouter()

DOCT_MAX_CONNECTIONS = 10
doct_active_connections = set()

# 用於儲存最終辨識結果的全局變數
doct_final_results = defaultdict(str) #避免打架
doct_summary_results = defaultdict(str)
doct_result_lock = Lock()
doct_active_sessions = {} #儲存 session_id: websocket 7/3
doct_last_access_time = {} #key: session_id, value:datetime

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

doctor_system_prompt = ("""
你是專業的醫療對話分析助手，負責整理「醫師與客人之間的逐字稿」。
請務必依照逐字稿內容進行分析，不得捏造、推測、擴寫 任何未提及資訊。

請閱讀逐字稿後，依以下定義輸出「結構化摘要」：

【分類定義】
---
客人主訴：
指客人主動提出、描述的問題、不適、症狀、需求、煩惱或想改善的部位。
（例：痘痘、斑、鬆弛、法令紋、臉凹、毛孔粗大、想變亮、想改善輪廓等）

診斷：
指醫師根據觀察、觸診、問診所提出的判斷、觀點、解釋或分析。
（例：皮膚狀況、肌膚鬆弛程度、骨架、脂肪分布、膠原蛋白流失等）

建議：
指醫師提出的治療選項、改善方法、施打建議、療程名稱、保養建議。
（例：可以考慮音波、電波、玻尿酸、皮秒、保養建議）

【分類規則】
---
僅能根據逐字稿中的資訊分類，不可推測或硬補內容。

如果內容無法清楚歸類 → 請統一歸在「診斷」。

若逐字稿沒有提到某一類別，則該欄留空即可。
                        
【請以以下格式輸出】（照格式即可）

【輸出格式範例】
---
【客人主訴】:
- xxx
【診斷】:
- xxx
【建議】:
- xxx


""")

@router.websocket("/ws/doctor")
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
    
    if len(doct_active_connections) >= DOCT_MAX_CONNECTIONS:
        await websocket.close(code=1008, reason="Connection limit exceeded")
        print(f"🔌 連線被拒絕：已達人數上限 {DOCT_MAX_CONNECTIONS}")
        return
    await websocket.accept()
    await websocket.send_text("🔊 新連線來了")
    doct_active_connections.add(websocket) # <--- 加入集合
    await websocket.send_text("🔊 已連線語音辨識 WebSocket")
###################################################################7/3
    import uuid
    doct_session_id = str(uuid.uuid4()) #為這次對話建立唯一ID
    doct_active_sessions[doct_session_id] = websocket #綁定 7/3

    doct_final_transcripts = []         #所有語音辨識結果(逐字稿)
    doct_segment_transcripts = []       #當前10分鐘的分段結果
    doct_all_segment_summaries = []     #每段的LLM摘要
    doct_all_refined_transcripts = []   #每段的潤飾摘要


    doct_segment_start_time = time.time() #計算每段開始時間
    doct_segment_duration_limit = 480  # 8 分鐘
    doct_last_audio_time = time.time() #最後收到音訊的時間

    doct_loop = asyncio.get_event_loop()

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
    await websocket.send_json({"type": "session_id", "session_id": doct_session_id}) 

    # 分段摘要函數也需要加強驗證
    async def process_segment_for_summary(transcripts_to_summarize):
        if not transcripts_to_summarize or not any(t.strip() for t in transcripts_to_summarize):
            return ""
        
        segment_text = "\n".join(transcripts_to_summarize)
        if not segment_text:
            return ""
        try:
            first_summary = llm.get_summary(segment_text, summarize_prompt)    #潤飾逐字稿
            summary = llm.get_summary(first_summary, doctor_system_prompt) #0724      #分類摘要
            
            # 驗證摘要格式是否正確
            required_categories = ["客人主訴", "診斷", "建議"]  #ˇ0807拿掉摘要
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
        
    def merge_summaries_by_category(summaries: list[str]) -> dict:
        """改進版的分段合併函數"""
        categories = [
            "客人主訴", "診斷", "建議"
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
        nonlocal doct_segment_start_time, doct_segment_transcripts, doct_all_segment_summaries, doct_all_refined_transcripts
        try:
            print('start segment_timer_task')
            # 一開始就潤稿一次（如果有錄音）
            if doct_segment_transcripts:
                result = await process_segment_for_summary(doct_segment_transcripts)
                doct_all_segment_summaries.append(result["categorized_summary"])
                doct_all_refined_transcripts.append(result["refined_transcript"])
                doct_segment_transcripts.clear()
            # 進入定時檢查迴圈 (錄音時間超過限制)
            while True:
                await asyncio.sleep(10)
                now = time.time()
                if now - doct_segment_start_time >= doct_segment_duration_limit:
                    if doct_segment_transcripts:
                        #result = await loop.run_in_executor(None, lambda: llm.get_summary("\n".join(segment_transcripts), summarize_prompt))
                        result = await process_segment_for_summary(doct_segment_transcripts)
                        doct_all_segment_summaries.append(result["categorized_summary"])
                        doct_all_refined_transcripts.append(result["refined_transcript"])
                        await safe_send(websocket, json.dumps({
                            "type": "segment_summary",
                            "summary": result["categorized_summary"]
                        }))
                        doct_segment_transcripts.clear()
                    doct_segment_start_time = now

        except asyncio.CancelledError:
            print("⏹ 分段任務已取消")

    #######################################加入靜音偵測

    async def check_silence():
        """背景任務：檢查是否超過10秒沒收到音訊"""
        nonlocal doct_last_audio_time, doct_segment_transcripts, doct_all_segment_summaries, doct_all_refined_transcripts, doct_segment_start_time

        try:
            print('start check silence')
            while True:
                await asyncio.sleep(1)  # 每秒檢查一次
                if time.time() - doct_last_audio_time > 10:  # 超過 10 秒沒收到音訊
                    print("⚠️ 超過 10 秒未收到音訊")
                    await websocket.send_json({
                        "type": "error",
                        "message": "未檢測到語音，請確認麥克風是否有聲音"
                    })
                    if doct_segment_transcripts and websocket.client_state == websockets.CONNECTED:  # ✅ flush 當前累積的文字
                        try: 
                            result = await process_segment_for_summary(doct_segment_transcripts)
                            doct_all_segment_summaries.append(result["categorized_summary"])
                            doct_all_refined_transcripts.append(result["refined_transcript"])
                            await safe_send(websocket, json.dumps({
                                "type": "segment_summary",
                                "summary": result["categorized_summary"]
                            }))
                            doct_segment_transcripts.clear()
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
            doct_final_transcripts.append(text)
            doct_segment_transcripts.append(text)
            # asyncio.run_coroutine_threadsafe(websocket.send_text(text), loop)
            asyncio.run_coroutine_threadsafe(
                safe_send(websocket, text),
                doct_loop
            )

    recognizer.recognized.connect(recognized_callback)
    recognizer.start_continuous_recognition()

    await websocket.send_text(json.dumps({
        "type": "session_id",
        "session_id": doct_session_id
    }))


    timer_task = asyncio.create_task(segment_timer_task())
    silence_task = asyncio.create_task(check_silence())

    try:
        while True:
            msg = await websocket.receive()   #0723
            # print("📩 raw message:", msg, flush=True)
            # try:
            #     msg = await asyncio.wait_for(websocket.receive(), timeout=1)
            # except asyncio.TimeoutError:
            #     # 每 1 秒檢查一次連線狀態
            #     continue
            if isinstance(msg, dict):
                # if msg.get("type") == "websocket.disconnect":
                #     print('sssssssssssssss websocket disconnect')
                #     break           

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

        if doct_final_transcripts: #有轉出文字才做後續
            # ✨ 最後一段補摘要
            if doct_segment_transcripts:
                result = await process_segment_for_summary(doct_segment_transcripts)
                doct_all_segment_summaries.append(result["categorized_summary"])  #只存分類摘要
                doct_all_refined_transcripts.append(result["refined_transcript"])
                doct_segment_transcripts.clear()

            # 組合最終摘要與逐字稿
            #final_combined_summary = "\n\n".join(all_segment_summaries)
            doct_final_refined_transcript = "\n".join(doct_all_refined_transcripts).strip()
            doct_final_combined_summary = merge_summaries_by_category(doct_all_segment_summaries)
            doct_summary_results[doct_session_id] = doct_final_combined_summary
            doct_final_text = "\n".join(doct_final_transcripts)
            doct_final_results[doct_session_id] = doct_final_text

            # print("📝 最終合併摘要：", final_combined_summary)
            print("📝 最終醫生完整逐字稿：", doct_final_text)
            print("📝 最終醫生完整潤飾稿：", doct_final_refined_transcript)
            print("📝 中斷了跑出session_id：", doct_session_id) #7/3
            
            # 在這裡傳送最終摘要完成通知 7/23
            await websocket.send_text(json.dumps({
                "type": "final_summary_ready",
                "session_id": doct_session_id
            }))
            #傳送合併潤飾搞
            await websocket.send_text(json.dumps({
                "type": "final_refined_transcript",
                "session_id": doct_session_id,
                "refined_transcript": doct_final_refined_transcript
            }))
            # 最終完整逐字稿
            await websocket.send_text(json.dumps({
                "type": "final_combined_text",
                "session_id": doct_session_id,
                "summary": doct_final_text
            }))
            # 最終摘要資料也送過去
            await websocket.send_text(json.dumps({
                "type": "final_combined_summary",
                "session_id": doct_session_id,
                "summary": doct_final_combined_summary
            }))
            return doct_final_refined_transcript
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
        
        if websocket in doct_active_connections:
            doct_active_connections.remove(websocket)
        
        doct_active_sessions.pop(doct_session_id, None)
        print(f"🔌 醫生連線已關閉 (Session: {doct_session_id})，目前連線數: {len(doct_active_connections)}")



