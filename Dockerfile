# Dockerfile
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # 한국 로케일/타임존(Optional)
    TZ=Asia/Seoul

# 필수 패키지 설치 (빌드도구 + curl + gpg + 글꼴 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gnupg \
    ca-certificates \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

# Google Chrome 설치
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /etc/apt/keyrings/google-linux-signing-keyring.gpg \
    && chmod a+r /etc/apt/keyrings/google-linux-signing-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/google-linux-signing-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
       > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /gptbitcoin

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 소스 복사
COPY . /gptbitcoin

# 스크린샷 디렉토리 미리 생성(볼륨 마운트 대상)
RUN mkdir -p /gptbitcoin/screenshots

# 컨테이너에서 기본 실행(필요시 compose에서 override 가능)
CMD ["python", "autotrade.py"]