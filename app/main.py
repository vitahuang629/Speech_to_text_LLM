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

# with open("app/config.json") as f:
#     config = json.load(f)
# speech_key = config["AZURE_SPEECH_KEY"]
# speech_region = config["AZURE_SERVICE_REGION"]
# endpoint_id = config["ENDPOINT_ID"]
# ollama_url = config["OLLAMA_URL"]

speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SERVICE_REGION")
endpoint_id = os.getenv("ENDPOINT_ID")
ollama_url = os.getenv("OLLAMA_URL")

# print("AZURE_SPEECH_KEY:", speech_key)
# print("AZURE_SERVICE_REGION:", speech_region)
# print("ENDPOINT_ID:", endpoint_id)
# print("OLLAMA_URL:", ollama_url)


app = FastAPI()
# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], #允許所有來源，如要限制請換成 ["http://localhost:3000"] 等
    allow_credentials = False, #跨域憑證
    allow_methods = ["*"], #允許所有HTTP 方法
    allow_headers = ["*"], #允許所有HTTP 標頭
)
# app.mount("/static", StaticFiles(directory="static"), name="static") ############################html

# 用於儲存最終辨識結果的全局變數
final_results = defaultdict(str) #避免打架
summary_results = defaultdict(str)
result_lock = Lock()
active_sessions = {} #儲存 session_id: websocket 7/3
last_access_time = {} #key: session_id, value:datetime

clean_interval = 60 #秒
data_ttl = 1200 #存活時間(20分鐘)

class RemoteLLM:
    def __init__(self, model, ollama_url, temperature=0.7, messages=None):
        self.model = model
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.messages = messages or []
    
    def get_summary(self, text, system_prompt):
        """向LLM發送請求並獲取摘要"""
        # 複製初始訊息
        messages = self.messages.copy()
        
        # 添加系統提示和用戶輸入
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        
        # 準備請求數據
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False
        }
        
        # 發送請求到Ollama服務
        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=data)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "無法獲取摘要")
        except requests.exceptions.Timeout:
            print("❌ LLM 請求超時")
            return "摘要請求超時，請稍後再試"
        except Exception as e:
            print(f"LLM請求錯誤: {e}")
            return f"獲取摘要時出錯: {str(e)}"


# 初始化LLM
llm = RemoteLLM(
    model="ycchen/breeze-7b-instruct-v1_0",
    # model = "kenneth85/llama-3-taiwan",
    # model = "jcai/llama3-taide-lx-8b-chat-alpha1:q6_k",
    ollama_url=ollama_url,
    temperature=0.7,
    messages=[{"role": "system", "content": "你是一位專業的醫美諮詢紀錄分析員"}]
)

# 系統提示
system_prompt = (
    """
你是對話分析助手，請根據以下規則，客觀整理逐字稿內容。

任務：
1. 將逐字稿中因語音辨識錯誤產生的非通順語句，**進行語句修正與語意整理**，使對話內容更通順、清晰。
2. 接著將對話內容依照指定分類，整理為條列重點摘要，幫助後續使用者快速了解談話重點。

🗣 本內容為一段逐字對話紀錄，內容包含不同角色的發言。  
請根據實際回覆內容，抽取對話中明確表達的**實際行為、經驗、需求或觀察**，重新整理為條列重點。

⚠️ 注意：
1. 只根據逐字稿內容，產出客觀、清楚、無腦補的紀錄摘要。
2. 每個分類下的內容必須為「重組過的摘要語句」，**不得直接複製逐字稿原句**或抄寫分類名稱。
3. **用詞準確、語氣中立**，避免醫學化名詞與情緒化語氣。
4. 必須一次輸出 **全部 8 個分類**，**不可遺漏**，包括【減重】、【睡眠】、【疼痛】、【私密處】、【臉部】、【再生醫療】、【其他】、【摘要】。
5. 如對話中未提及該分類，該分類內容請留空。
6. 【摘要】分類必須獨立產出，內容須統整對話的整體重點。

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
【摘要】:
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
- 其他：服務滿意度、療程效果、人員滿意度。
- 摘要：必須有，總結整體談話重點。

  ❗禁止捏造、推測或擴寫逐字稿中未提及的資訊，例如：「應該是壓力大導致失眠」這類判斷性語句不可出現。所有資訊皆須直接來自逐字稿內容，且不可有臆測、揣測或合理推論。

"""
)

# @app.get("/")   #如果不使用前端的話
# async def index():
#     return HTMLResponse(open("static/index.html", "r", encoding="utf-8").read())

@app.get("/")
async def index():
    return {"message": "Backend is running."}


@app.get("/api/final-result/{session_id}")
async def get_final_result(session_id: str):
    """
    REST API 端點，用於獲取最終辨識結果和 LLM 摘要
    """
    async with result_lock: # <--- 增加這行鎖
        now = datetime.now(timezone.utc)
        # print("finalllllllllllll", final_results)
        # print('summmmmmmmmmmmm', summary_results)
        if session_id in final_results and session_id in summary_results:
            # print('yeeeeeeeeeeeesssssss')
            last_access_time[session_id] = now
            # final_text = final_results.pop(session_id)  #7/9
            # final_summary = summary_results.pop(session_id) #7/9
            return {
                "type": "final_result",
                "text": final_results[session_id],
                "summary": summary_results.get(session_id, "正在生成摘要...") #7/9
                #"summary": summary_results.get(session_id, {})
                #"summary": summary_results[session_id]
            }
            # return {
            #         "type": "final_result",
            #         "text": final_text,
            #         "summary": final_summary
            #     }
        elif session_id in summary_results or session_id in summary_results:
            # 也許只是還沒把 final_results 塞好
            return {
                "type": "pending",
                "message": "結果尚未完全生成，請稍後再試"
            }

        else:
            return {
                "type": "error",
                "message": "找不到指定的辨識結果"
            }


MAX_CONNECTIONS = 10
active_connections = set()


@app.websocket("/ws/asr")
async def websocket_endpoint(websocket: WebSocket):
    print(f"📡 新連線來自：{websocket.client.host}:{websocket.client.port}")
    if len(active_connections) >= MAX_CONNECTIONS:
        await websocket.close(code=1008, reason="Connection limit exceeded")
        print(f"🔌 連線被拒絕：已達人數上限 {MAX_CONNECTIONS}")
        return
    await websocket.accept()
    active_connections.add(websocket) # <--- 加入集合
    await websocket.send_text("🔊 已連線語音辨識 WebSocket")
###################################################################7/3
    import uuid
    session_id = str(uuid.uuid4()) #為這次對話建立唯一ID
    active_sessions[session_id] = websocket #綁定 7/3

    final_transcripts = []         #所有語音辨識結果(逐字稿)
    segment_transcripts = []       #當前10分鐘的分段結果
    all_segment_summaries = []     #每段的LLM摘要

    segment_start_time = time.time()
    segment_duration_limit = 480  # 8 分鐘

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
            summary = llm.get_summary(segment_text, system_prompt) #0724
            
            # 驗證摘要格式是否正確
            required_categories = ["減重", "睡眠", "疼痛", "私密處", "臉部", "再生醫療", "其他", "摘要"]
            for category in required_categories:
                if f"【{category}】:" not in summary:
                    print(f"⚠️ 警告：摘要中缺少 【{category}】: 分類")
            
            # 檢查是否有不當的格式
            if "摘要：" in summary and "【摘要】:" not in summary:
                print("⚠️ 警告：發現格式錯誤，包含「摘要：」而非「【摘要】:」")
            
            #print(f"📋 分段摘要生成：{summary[:100]}...")
            print("🧾 LLM回傳摘要原文：", summary)
            return summary
            
        except Exception as e:
            print(f"獲取分段LLM摘要時出錯: {e}")
            return f"獲取分段摘要時出錯: {str(e)}"

    # def merge_summaries_by_category(summaries: list[str]) -> str: 改成json 7/9
    def merge_summaries_by_category(summaries: list[str]) -> dict:
        """改進版的分段合併函數"""
        categories = [
            "減重", "睡眠", "疼痛", "私密處", "臉部", "再生醫療", "其他", "摘要"
        ]
        merged = defaultdict(list)

        for summary in summaries:
            if not summary:
                continue
            
            summary = summary.strip()
            
            for category in categories:
                try:
                    # 更精確的正則表達式，只匹配【分類】: 格式
                    # 匹配從【分類】: 開始，到下一個【分類】: 或字串結尾的內容
                    # pattern = rf"【{category}】[:：]\s*((?:(?!\n【).|\n)*?)(?=\n【|$)"#6/30 避免沒有拼起來
                    pattern = rf"(?:【)?{category}(?:】)?[:：]\s*((?:(?!\n(?:【)?(?:{'|'.join(categories)})(?:】)?[:：]).|\n)*?)(?=\n(?:【)?(?:{'|'.join(categories)})(?:】)?[:：]|$)" #8/1更新不論有沒有括號

                    matches = re.findall(pattern, summary, re.DOTALL)
                    
                    for match in matches:
                        content = match.strip()
                        
                        # 將內容按行分割，並過濾掉空行和只有空白的行
                        lines = [line.strip() for line in content.split('\n') if line.strip()]
                        if lines:
                            for line in lines:
                                if not any(keyword in line for keyword in ["未提及", "無相關", "無", "未", "x"]):
                                    if not line.startswith("- "):
                                        line = "- " + line
                                    if line not in merged[category]:
                                        merged[category].append(line)
                                
                except Exception as e:
                    print(f"⚠️ 分析 {category} 時出錯: {e}")
                    continue

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
        # return "\n".join(result)
    
    # 分段摘要定時器 (背景定時任務)
    async def segment_timer_task():
        nonlocal segment_start_time, segment_transcripts, all_segment_summaries
        try:
            while True:
                await asyncio.sleep(10)
                now = time.time()
                if now - segment_start_time >= segment_duration_limit:
                    if segment_transcripts:
                        summary = await process_segment_for_summary(segment_transcripts)
                        all_segment_summaries.append(summary)

                        # 🔁 即時回傳摘要給前端（可選）
                        await websocket.send_text(json.dumps({
                            "type": "segment_summary",
                            "summary": summary
                        }))

                        segment_transcripts.clear()
                    segment_start_time = now
        except asyncio.CancelledError:
            print("⏹ 分段任務已取消")

    # 接收辨識結果
    def recognized_callback(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text
            final_transcripts.append(text)
            segment_transcripts.append(text)
            asyncio.run_coroutine_threadsafe(websocket.send_text(text), loop)

    recognizer.recognized.connect(recognized_callback)
    recognizer.start_continuous_recognition()

    await websocket.send_text(json.dumps({
        "type": "session_id",
        "session_id": session_id
    }))

    timer_task = asyncio.create_task(segment_timer_task())

    try:
        while True:
            # data = await websocket.receive_bytes() #原先接收音訊0723
            # stream.write(data) #0723

            msg = await websocket.receive()   #0723
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
                    # print("🔊 Writing audio data to stream...")
                    stream.write(msg["bytes"])

        recognizer.stop_continuous_recognition()
        stream.close()
        timer_task.cancel()

        if final_transcripts: #有轉出文字才做後續
            # ✨ 最後一段補摘要
            if segment_transcripts:
                summary = await process_segment_for_summary(segment_transcripts)
                all_segment_summaries.append(summary)
                segment_transcripts.clear()

            # 組合最終摘要與逐字稿
            #final_combined_summary = "\n\n".join(all_segment_summaries)
            final_combined_summary = merge_summaries_by_category(all_segment_summaries) #6/17 同類別合併
            summary_results[session_id] = final_combined_summary
            final_text = "\n".join(final_transcripts)
            final_results[session_id] = final_text

            print("📝 最終合併摘要：", final_combined_summary)
            print("📝 最終完整逐字稿：", final_text)
            print("📝 session_id：", session_id) #7/3
            
            # 在這裡傳送最終摘要完成通知 7/23
            await websocket.send_text(json.dumps({
                "type": "final_summary_ready",
                "session_id": session_id
            }))
            # 逐字稿
            await websocket.send_text(json.dumps({
                "type": "final_combined_text",
                "session_id": session_id,
                "summary": final_text
            }))
            # 再把摘要資料也送過去
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
    
    finally: # <--- 使用 finally 確保無論如何都會執行
        # 確保資源被清理
        if 'timer_task' in locals() and not timer_task.done():
            timer_task.cancel()
        
        if websocket in active_connections:
            active_connections.remove(websocket)
        
        active_sessions.pop(session_id, None)
        print(f"🔌 連線已關閉 (Session: {session_id})，目前連線數: {len(active_connections)}")




#你是一位專業的諮詢紀錄分析員，撰寫的摘要將提供給醫美診所的諮詢師使用，用於建立客戶的個人輪廓。請務必：
#本診所專注於以下四大項目：睡眠治療、體雕項目、臉部拉提保養、性功能提升。請僅將此作為理解語境的背景資訊，摘要中不得揣測客戶需求是否與此相關，除非逐字稿中明確提及。
#你的產出將直接作為診所系統中的「客戶紀錄」，請以專業、清晰、實用為原則。
#【摘要】：請務必產出，不可省略。請統整客戶在對話中明確提及的背景資訊，例如：年齡、職業、生活作息、壓力來源、飲食、運動等，幫助諮詢師快速掌握客戶樣貌。

# 你是對話分析助手，請客觀整理逐字稿內容。

# 任務：將對話內容按指定分類整理，幫助後續使用者快速了解談話重點。

# 1. 只根據逐字稿內容，產出客觀、清楚、無腦補的紀錄摘要。
# 2. 用詞準確，避免使用逐字稿中未出現的診斷名詞、病名或醫學判斷。
# 3. 若有模糊或錯誤用詞（如聽寫錯誤、片語不清），請註記為「語意不明」或「合理修正」並加註原文。
# 4. 產出格式應包含：
#    - 【減重】: 提到體重、瘦身、和減肥相關的片段。如果沒有提到相關內容，則回覆: "無相關內容"。
#    - 【睡眠】: 提到睡眠、安眠藥相關的片段。如果沒有提到相關內容，則回覆: "無相關內容"。
#    - 【疼痛】: 提到疼痛、痠痛相關的片段。如果沒有提到相關內容，則回覆: "無相關內容"。
#    - 【私密處】: 提到私密處相關的片段。如果沒有提到相關內容，則回覆: "無相關內容"。
#    - 【臉部】: 提到臉部、皺紋相關的片段。如果沒有提到相關內容，則回覆: "無相關內容"。
#    - 【再生醫療】: 提到再生醫療相關的片段。如果沒有提到相關內容，則回覆: "無相關內容"。
#    - 【其他】: 只要無法被歸納為上面的片段。
#    - 【摘要】：請務必產出，不可省略。請統整客戶在對話中的內容。

#   ❗禁止捏造、推測或擴寫逐字稿中未提及的資訊，例如：「應該是壓力大導致失眠」這類判斷性語句不可出現。所有資訊皆須直接來自逐字稿內容，且不可有臆測、揣測或合理推論。

# 分段摘要函數
    # async def process_segment_for_summary(transcripts_to_summarize):
    #     if not transcripts_to_summarize:
    #         return ""
    #     segment_text = "\n".join(transcripts_to_summarize)
    #     try:
    #         summary = llm.get_summary(segment_text, system_prompt)
    #         print(f"📋 分段摘要生成：{summary[:50]}...")
    #         return summary
    #     except Exception as e:
    #         print(f"獲取分段LLM摘要時出錯: {e}")
    #         return f"獲取分段摘要時出錯: {str(e)}"




    # def merge_summaries_by_category(summaries: list[str]) -> str:    #6/17 加入分段合併
    #     categories = [
    #         "減重", "睡眠", "疼痛", "私密處", "臉部", "再生醫療", "其他", "摘要"
    #     ]
    #     merged = defaultdict(list)
    #     for summary in summaries:
    #         if not summary:
    #             continue
    #         for category in categories:
    #             try:
    #                 # ✅ 改進 regex：避免 group 擋住內容、加容錯
    #                 pattern = f"【{category}】:\\s*((?:.|\n)*?)(?=(\n【|$))"
    #                 matches = re.findall(pattern, summary, re.DOTALL)
    #                 for match in matches:
    #                     content = match[0].strip()
    #                     if content and content != "無相關內容":
    #                         merged[category].append(content)
    #             except Exception as e:
    #                 print(f"⚠️ 分析 {category} 時出錯: {e}")
    #                 continue

    #     result = []
    #     for category in categories:
    #         if merged.get(category):
    #             combined = " / ".join(merged[category])
    #         else:
    #             combined = "無相關內容"
    #         result.append(f"【{category}】: {combined}")
        
    #     return "\n".join(result)

    # @app.get("/api/final-result/{session_id}")
# async def get_final_result(session_id: str):
#     """
#     REST API 端點，用於獲取最終辨識結果和LLM摘要
#     """
#     now = datetime.now(timezone.utc)
#     if session_id in final_results:
#         # result = final_results[session_id]
#         # summary = summary_results.get(session_id, "正在生成摘要...")
#         last_access_time[session_id] = now #更新最後存取時間
        
#         # # 獲取後刪除，避免佔用記憶體
#         # if session_id in final_results:
#         #     del final_results[session_id]
#         # if session_id in summary_results:
#         #     del summary_results[session_id]
            
#         return {
#             "type": "final_result", 
#             "text": final_results[session_id],
#             "summary": summary_results.get(session_id, "正在生成摘要...")
#         }
#     else:
#         return {"type": "error", "message": "找不到指定的辨識結果"}

    # except WebSocketDisconnect:
    #     async with result_lock:
    #         recognizer.stop_continuous_recognition()
    #         stream.close()
    #         timer_task.cancel()

    #         if final_transcripts: #有轉出文字才做後續
    #             # ✨ 最後一段補摘要
    #             if segment_transcripts:
    #                 summary = await process_segment_for_summary(segment_transcripts)
    #                 all_segment_summaries.append(summary)
    #                 segment_transcripts.clear()

    #             # 組合最終摘要與逐字稿
    #             #final_combined_summary = "\n\n".join(all_segment_summaries)
    #             final_combined_summary = merge_summaries_by_category(all_segment_summaries) #6/17 同類別合併
    #             summary_results[session_id] = final_combined_summary
    #             final_text = "\n".join(final_transcripts)
    #             final_results[session_id] = final_text

    #             print("📝 最終合併摘要：", final_combined_summary)
    #             print("📝 最終完整逐字稿：", final_text)
    #             print("📝 session_id：", session_id) #7/3

    #             # 在這裡傳送最終摘要完成通知 7/23
    #             await websocket.send_text(json.dumps({
    #                 "type": "final_summary_ready",
    #                 "session_id": session_id
    #             }))
    #             return final_combined_summary
    #         else:
    #             print(f"❌ 沒有辨識到任何文字，跳過 LLM 生成 (Session: {session_id})")

