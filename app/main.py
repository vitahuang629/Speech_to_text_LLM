from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  
from fastapi.responses import HTMLResponse    
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import app.doct_ws as doct_ws
import app.consultant_ws as consultant_ws
import app.aws_main as aws_main

# 重要：加入 root_path 讓 /docs 在子路徑下也能正常顯示
app = FastAPI(root_path="/clinics-speech")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], #允許所有來源，如要限制請換成 ["http://localhost:3000"] 等
    allow_credentials = False, #跨域憑證
    allow_methods = ["*"], #允許所有HTTP 方法
    allow_headers = ["*"], #允許所有HTTP 標頭
)

# 2. 將醫師和諮詢師的路由掛載到應用程式中
# prefix 參數非常重要，它確保路由的完整路徑是 /ws/doctor 和 /ws/consultant
app.include_router(doct_ws.router)
app.include_router(consultant_ws.router)

# 3. 【新增】掛載 AWS 相關路由
# 建議加上 tags=["AWS"]，這樣在 /docs 文件頁面上會分類顯示，比較整齊
app.include_router(aws_main.router, tags=["AWS Audio"])


# # 靜態檔案路由 (保持不變)
# app.mount("/static", StaticFiles(directory="static"), name="static")


# @app.get("/")   #如果不使用前端的話
# async def index():
#     return HTMLResponse(open("static/index.html", "r", encoding="utf-8").read())

@app.get("/")
async def index():
    return {"message": "Clinics Speech Backend is running."}