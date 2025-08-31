import os
from dotenv import load_dotenv
load_dotenv()

def ai_trading():
    # 1. 업비트 차트 데이터 가져오기 (30일 데이터)
    import pyupbit
    df = pyupbit.get_ohlcv("KRW-BTC", count=30, interval="day")

    # 2. OpenAI에게 데이터 제공하고 판단받기
    from openai import OpenAI
    client = OpenAI()

    response = client.responses.create(
    model="gpt-4o",
    input=[
        {
        "role": "system",
        "content": [
            {
            "type": "input_text",
            "text": "You are an expert in Bitcoin investing. Tell me whether to buy, sell, or hold at the moment based on the chart data provided. Response in Json format.\n\nResponse Example:\n{decision: \"buy\", \"reason\": \"some technical reason\"}\n{decision: \"sell\", \"reason\": \"some technical reason\"}\n{decision: \"hold\", \"reason\": \"some technical reason\"}"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "input_text",
            "text": df.to_json()
            }
        ]
        }
    ],
    text={
        "format": {
        "type": "json_object"
        }
    }
    )

    result = response.output[0].content[0].text

    import json
    result = json.loads(result)

    import pyupbit

    access = os.getenv('UPBIT_ACCESS_KEY')
    secret = os.getenv('UPBIT_SECRET_KEY')
    upbit = pyupbit.Upbit(access, secret)

    print("### AI Decision: ",result['decision'].upper(),"###")
    print("### Reason: ",result['reason'],"###")

    if result['decision'] == "buy":
        #매수
        my_krw = upbit.get_balance("KRW")
        # 최소 주문 가능 금액
        if my_krw * 0.9995 > 5000 :
            print("### Buy Order Executed ###")
            print(upbit.buy_market_order("KRW-BTC", my_krw * 0.9995))
        else :
            print("### Buy Order Failed: Insufficient KRW (less than 5000 KRW) ###")
    elif result['decision'] == "sell":
        #매도
        #보유 코인 조회
        my_btc = upbit.get_balance("KRW-BTC")
        #현재 매도 호가 조회
        current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]

        if my_btc * current_price > 5000 :
            print("### Sell Order Executed ###")
            print(upbit.sell_market_order("KRW-BTC", upbit.get_balance("KRW-BTC")))
        else :
            print("### Sell Order Failed: Insufficient BTC (less than 5000 KRW) ###")
    elif result['decision'] == "hold":
        #보유
        print("### Hold Order No Executed ###")
        print("hold:",result["reason"])

ai_trading()

# while True:
#     import time
#     #10초마다 실행
#     time.sleep(10)
#     ai_trading()