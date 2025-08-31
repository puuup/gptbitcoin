# Dockerfile
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Selenium이 찾을 수 있도록 경로 지정
    CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER=/usr/bin/chromedriver \
    # 한국 로케일/타임존(Optional)
    TZ=Asia/Seoul

# 필수 시스템 패키지 + Chromium/ChromeDriver 설치
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      chromium chromium-driver \
      ca-certificates curl wget git \
      fonts-liberation \
      libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 \
      libdrm2 libgbm1 libgtk-3-0 libnss3 \
      libx11-6 libxcomposite1 libxdamage1 libxext6 libxfixes3 \
      libxkbcommon0 libxrandr2 xdg-utils \
      # Pillow가 종종 요구하는 라이브러리
      libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /gptbitcoin

RUN pip install --upgrade pip && pip install -r /gptbitcoin/requirements.txt

# 앱 소스 복사
COPY . /gptbitcoin

# 스크린샷 디렉토리 미리 생성(볼륨 마운트 대상)
RUN mkdir -p /gptbitcoin/screenshots

# 컨테이너에서 기본 실행(필요시 compose에서 override 가능)
CMD ["python", "autotrade.py"]