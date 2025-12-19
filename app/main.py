from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from fastapi.middleware.cors import CORSMiddleware


# 1. 從 routers 資料夾匯入您定義的路由物件
import app.doct_ws as doct_ws
import app.consultant_ws as consultant_ws

# 初始化 FastAPI 應用
app = FastAPI()

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


# 靜態檔案路由 (保持不變)
# app.mount("/static", StaticFiles(directory="static"), name="static")


# @app.get("/")   #如果不使用前端的話
# async def index():
#     return HTMLResponse(open("static/index.html", "r", encoding="utf-8").read())

@app.get("/")
async def index():
    return {"message": "Backend is running."}