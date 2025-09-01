# 📈 GPT 기반 비트코인 자동매매 프로젝트

## 프로젝트 개요

이 프로젝트는 **비트코인 자동매매 시스템**으로,
기술적 지표 분석과 외부 뉴스·심리 지표를 종합해 매수/매도/보유 여부를 판단합니다.

분석에는 OpenAI GPT 모델을 활용했으며, Upbit API를 통해 실제 주문까지 자동으로 실행됩니다.
모든 결과는 SQLite DB에 기록되며, **Streamlit 대시보드**에서 매매 기록과 포트폴리오 추이를 한눈에 확인할 수 있습니다.

단순히 자동 매매를 넘어서, 과거 의사결정을 평가하고 교훈을 학습하는 **Reflection 기능**을 넣어, 시스템이 점차 개선될 수 있도록 설계했습니다.

---

## 주요 기능

* **자동 매매 의사결정**

  * RSI, MACD, 볼린저 밴드 등 다양한 기술적 지표
  * 공포·탐욕 지수(Fear & Greed Index)
  * 뉴스/유튜브 감성 분석
* **실시간 주문**

  * Upbit API와 연동하여 매수/매도 실행
  * 최소 주문 금액 검증
* **데이터 저장 및 분석**

  * 거래 기록, 포트폴리오 스냅샷, 회고 데이터 저장
* **대시보드 시각화**

  * Streamlit 기반 실시간 모니터링
  * 매매 기록/포트폴리오/Reflection 결과 확인 가능

---

## 프로젝트 구조

```
.
├── autotrade.py          # 자동매매 실행 로직
├── streamlit-app.py      # 실시간 대시보드
├── requirements.txt      # 패키지 의존성
├── Dockerfile            # 컨테이너 실행 환경 정의
├── docker-compose.yml    # 서비스 실행 정의
└── trading_history.db    # SQLite DB (실행 중 자동 생성)
```

---

## 실행 방법

### 1) 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다:

```env
UPBIT_ACCESS_KEY=your_upbit_access_key
UPBIT_SECRET_KEY=your_upbit_secret_key
OPENAI_API_KEY=your_openai_api_key
```

(선택적으로 뉴스 분석용 `SERPAPI_KEY`, 유튜브 분석용 `YT_VIDEO_IDS`를 넣을 수 있습니다.)

### 2) 로컬 실행

```bash
pip install -r requirements.txt

# 자동매매 실행
python autotrade.py

# 대시보드 실행
streamlit run streamlit-app.py
```

### 3) Docker 실행

```bash
docker-compose up --build
```

---

## 대시보드 미리보기

Streamlit 대시보드에서는 다음을 확인할 수 있습니다:

* **매매 기록**

  * 매수/매도/보유 기록 필터링
  * 최근 수익률 확인
* **포트폴리오 추이**

  * 총 자산과 수익률 변화 그래프
* **Reflection**

  * 과거 매매 성과 기반 교훈 요약

---

## 사용 기술

* **Python** (데이터 수집 및 분석)
* **Streamlit** (대시보드 UI)
* **SQLite** (거래 및 포트폴리오 기록 저장)
* **OpenAI API** (의사결정 로직)
* **PyUpbit API** (실제 거래 연동)
* **Docker / docker-compose** (실행 환경 컨테이너화)

---

## 주의사항

* 이 프로젝트는 **포트폴리오 시연 목적**으로 제작되었습니다.
* 실제 투자에서는 충분히 검증되지 않았기 때문에 참고용으로만 활용하세요.

---
