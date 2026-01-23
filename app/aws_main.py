import boto3
import requests
import json
import os
import urllib.parse
import asyncio  # 新增 asyncio 用於非同步等待
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel 
import re


load_dotenv()

# --- 設定 AWS S3 ---
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_REGION = os.environ.get("AWS_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
AZURE_SPEECH_KEY = os.environ.get("SPEECH_KEY")
AZURE_REGION = os.environ.get("SPEECH_REGION")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

router = APIRouter()

# --- 定義輸入資料格式 ---
class AudioRequest(BaseModel):
    url: str  # 前端只需要傳一個 S3 的 url 進來

# --- 設定AI整理逐字稿---
class OpenAILLM:
    def __init__(self, model, api_key, temperature=0.7, messages=None):
        self.model = model
        self.temperature = temperature
        self.messages = messages or []
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
    model="gpt-4o-mini",
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
針劑：任何針劑、注射、紅光、粒線體、身體健康改善等體內治療話題
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
【針劑】:
- xxx
【其他】:
- xxx
---

若無相關內容，則只顯示該類別名稱，內容留空
"""
)

system_user_prompt = """請依照 system prompt 規範，將以下條列式內容分類整理：
            {first_summary}"""      


def get_s3_presigned_url(object_key, expiration=7200):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': object_key},
            ExpiresIn=expiration
        )
        print(f"[AWS] 生成簽章 URL 成功")
        return url
    except Exception as e:
        print(f"[AWS] 生成失敗: {str(e)}")
        return None

def trigger_azure_transcription(audio_url):
    api_url = f"https://{AZURE_REGION}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "contentUrls": [audio_url],
        "locale": "zh-TW",
        "displayName": "Batch_Transcription_Job",
        "properties": {
            "diarizationEnabled": False,
            "punctuationMode": "DictatedAndAutomatic"
        }
    }
    print("[Azure] 發送請求中...")
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 201:
            job_info = response.json()
            print(f"[Azure] 成功！Job ID: {job_info['self']}")
            return job_info['self']
        else:
            print(f"[Azure] 失敗: {response.text}")
            return None
    except Exception as e:
        print(f"[Azure] 請求異常: {e}")
        return None
    

def extract_key_from_url(full_url):
    """
    從完整 S3 URL 解析出正確的 Key，並處理中文解碼
    """
    # 1. 移除 https://...amazon.com/ 部分
    # 解析出路徑部分: /ASR/2025...%E9%...webm
    parsed = urllib.parse.urlparse(full_url)
    path = parsed.path 
    
    # 2. 去除開頭的斜線 (如果有)
    if path.startswith('/'):
        path = path[1:]
        
    # 3. 將 URL 編碼轉回中文 (例如 %E9%BB%83 -> 黃)
    decoded_key = urllib.parse.unquote(path)
    
    return decoded_key

def parse_summary_to_db_structure(text):
    """
    將 GPT 生成的字串解析為前端需要的資料庫格式
    """
    # 1. 定義標準空骨架 (ID 1~7 對應你的分類，8~9 留空)
    data_structure = [
        {"id": 1, "value": "", "name": "減重"},
        {"id": 2, "value": "", "name": "睡眠"},
        {"id": 3, "value": "", "name": "疼痛"},
        {"id": 4, "value": "", "name": "私密處"},
        {"id": 5, "value": "", "name": "臉部"},
        {"id": 6, "value": "", "name": "再生醫療"},
        {"id": 7, "value": "", "name": "其他"},
        {"id": 8, "value": "", "name": "諮詢師治療計畫"},  # 保持為空
        {"id": 9, "value": "", "name": "客戶治療滿意度"}   # 保持為空
    ]

    # 2. 針對前 7 個類別進行內容提取
    for item in data_structure:
        # 只處理 ID 1~7
        if item["id"] > 7:
            continue
            
        name = item["name"]
        
        # 設定搜尋關鍵字 (如果要相容 GPT 有時輸出 '針劑' 的情況)
        search_key = name
        if name == "再生醫療":
             # 讓程式同時找 "【再生醫療】" 或 "【針劑】"
             pattern_str = r"【(再生醫療|針劑)】:\s*(.*?)(?=\n\n【|\n【|\n---|$)"
        else:
             pattern_str = f"【{name}】:\s*(.*?)(?=\n\n【|\n【|\n---|$)"
        
        # 使用正規表示法抓取內容
        match = re.search(pattern_str, text, re.DOTALL)
        
        if match:
            content = match.group(2).strip() if name == "再生醫療" else match.group(1).strip()
            if content == "-" or content == "":
                item["value"] = ""
            else:
                item["value"] = content

    return data_structure

# --- 核心邏輯修改為 Async ---

async def check_and_download_result(job_url):
    """
    輪詢 (Polling) 檢查轉錄狀態，成功後下載文字
    注意：這裡改用了 async 和 await asyncio.sleep
    """
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY
    }

    print(f"[Azure] 開始監控工作狀態: {job_url}")

    while True:
        # 這裡使用 requests (同步) 呼叫是可以的，若併發量大可改用 httpx
        response = requests.get(job_url, headers=headers)
        
        if response.status_code != 200:
            print(f"無法取得狀態: {response.text}")
            return None

        status_data = response.json()
        status = status_data["status"]
        print(f"目前狀態: {status} ... (等待 5 秒)")

        if status == "Succeeded":
            print("[Azure] 轉錄完成！正在下載結果...")
            files_url = status_data["links"]["files"]
            files_response = requests.get(files_url, headers=headers)
            files_data = files_response.json()

            for file_info in files_data["values"]:
                if file_info["kind"] == "Transcription":
                    result_url = file_info["links"]["contentUrl"]
                    result_content = requests.get(result_url).content
                    result_json = json.loads(result_content)

                    full_text = ""
                    for combined in result_json["combinedRecognizedPhrases"]:
                        full_text += combined["display"] + " "
                    return full_text
            break

        elif status == "Failed":
            error_msg = status_data['properties'].get('error', 'Unknown Error')
            print(f"[Azure] 轉錄失敗: {error_msg}")
            return None

        # 關鍵修改：使用非同步等待，避免卡住伺服器
        await asyncio.sleep(5)


# --- 定義 API 路由 ---
# 這裡不需要 meeting_id，直接接收一個 URL 即可

@router.post("/process-audio")
async def process_audio_endpoint(request: AudioRequest):
    """
    接收 S3 URL -> 觸發 Azure 轉錄 -> 等待結果 -> GPT 整理 -> 回傳 JSON
    """
    raw_url = request.url
    print(f"收到處理請求，URL: {raw_url}")

    # 1. 解析 Key
    s3_key = extract_key_from_url(raw_url)
    
    # 2. 取得 AWS 簽章連結
    s3_sas_url = get_s3_presigned_url(s3_key)
    
    if not s3_sas_url:
        raise HTTPException(status_code=400, detail="無法生成 AWS 簽章 URL")

    # 3. 送出轉錄工作
    job_url = trigger_azure_transcription(s3_sas_url)
    
    if not job_url:
        raise HTTPException(status_code=500, detail="Azure 轉錄請求失敗")

    # 4. 等待並取得文字
    print("--- 等待轉錄結果中 (這可能需要幾分鐘) ---")
    
    # 這裡會等待直到轉錄完成
    text = await check_and_download_result(job_url)

    if not text:
        raise HTTPException(status_code=500, detail="轉錄失敗或無法取得文字")

    print("\n========= 最終逐字稿 =========")
    print(text[:100] + "...") # 只印出前100字避免 log 太長
    print("=============================")

    # 5. LLM 處理
    # 潤飾逐字稿
    first_summary = await llm.get_summary(
        text=summarize_user_prompt.format(segment_text=text), 
        system_prompt=summarize_prompt
    )
    
    # 分類摘要
    combined_summary = await llm.get_summary(
        text=system_user_prompt.format(first_summary=first_summary), 
        system_prompt=system_prompt
    )

    combined_summary = combined_summary.replace("【針劑】", "【再生醫療】")
    formatted_result = parse_summary_to_db_structure(combined_summary)
    print(formatted_result)
    # 6. 回傳結果 (JSON)
    # 不再使用 send_text (WebSocket)，而是直接 return
    return {
        "text": text,
        "ai_text":first_summary,
        "final_combined_summary": formatted_result
    }
    
        

# if __name__ == "__main__":
#     # 您的測試網址
#     raw_url = "https://hopkins-main.s3.ap-northeast-1.amazonaws.com/ASR/20251229145634_2025-12-29_14-56-33_%E9%BB%83%E9%9B%AF%E4%BC%B6.webm"
    
#     # 1. 解析 Key
#     s3_key = extract_key_from_url(raw_url)
    
#     # 2. 取得 AWS 簽章連結
#     s3_sas_url = get_s3_presigned_url(s3_key)
    
#     if s3_sas_url:
#         # 3. 送出轉錄工作
#         job_url = trigger_azure_transcription(s3_sas_url)
        
#         if job_url:
#             # 4. (新增步驟) 等待並取得文字
#             print("--- 等待轉錄結果中 (這可能需要幾分鐘) ---")
#             text = check_and_download_result(job_url)
            
#             print("\n========= 最終逐字稿 =========")
#             print(text)
#             print("=============================")