# Clinics-Text-Api/Dockerfile
# 使用官方 Python 基礎映像
FROM python:3.11.0-slim

# 設定工作目錄
WORKDIR /app

#複製其他專案檔案
COPY . /app/

#安裝poetry
RUN pip install poetry==1.8.5

# 複製 Poetry 設定檔
COPY pyproject.toml poetry.lock* /app/

# 安裝依賴
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# 開放 port
EXPOSE 8000

# 啟動指令
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
