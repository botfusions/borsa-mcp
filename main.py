from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import httpx
import json

app = FastAPI(title="Market Virtuoso API", version="2.0.0")

class AnalysisRequest(BaseModel):
    session_id: str
    user_message: str
    asset_ticker: Optional[str] = None

class TechnicalData(BaseModel):
    ticker: str
    candles: List[dict]  # OHLC data
    rsi: Optional[float] = None
    macd: Optional[dict] = None

class SentimentData(BaseModel):
    ticker: str
    sentiment_score: float  # -1 to 1
    news: List[dict]

@app.get("/")
def read_root():
    return {"message": "Market Virtuoso API çalışıyor!", "status": "ok", "persona": "Market Maestro"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "market-virtuoso"}

@app.post("/analyze/technical/{ticker}")
async def get_technical_analysis(ticker: str):
    """90 günlük teknik analiz verisi"""
    # Simulated data for now - buraya gerçek API entegrasyonu gelecek
    return {
        "ticker": ticker.upper(),
        "timeframe": "90d",
        "candles": [
            {"date": "2025-01-29", "open": 100, "high": 105, "low": 98, "close": 103, "volume": 1000000}
            # Gerçek veriler buraya gelecek
        ],
        "indicators": {
            "rsi": 65.2,
            "macd": {"value": 0.8, "signal": 0.6, "histogram": 0.2},
            "ma_200": 98.5
        }
    }

@app.post("/analyze/sentiment/{ticker}")
async def get_sentiment_analysis(ticker: str):
    """Sentiment ve haber analizi"""
    return {
        "ticker": ticker.upper(),
        "sentiment_score": 0.3,  # -1 to 1
        "sentiment_label": "Hafif Pozitif",
        "news": [
            {"title": f"{ticker} için olumlu gelişmeler", "sentiment": 0.4, "source": "Reuters"},
            {"title": f"{ticker} analisti raporları", "sentiment": 0.2, "source": "Bloomberg"}
        ],
        "analysis_date": "2025-01-29"
    }

@app.post("/maestro/analyze")
async def maestro_full_analysis(request: AnalysisRequest):
    """Market Virtuoso tam analiz endpoint'i"""
    
    # Asset ticker'ı extract et
    ticker = request.asset_ticker or extract_ticker_from_message(request.user_message)
    
    analysis_result = {
        "session_id": request.session_id,
        "detected_ticker": ticker,
        "analysis_type": "full_virtuoso",
        "ready_for_llm": True,
        "message": "Veriler hazır, LLM'e gönderilebilir"
    }
    
    if ticker:
        # Teknik ve sentiment verilerini birleştir
        technical = await get_technical_analysis(ticker)
        sentiment = await get_sentiment_analysis(ticker)
        
        analysis_result.update({
            "technical_data": technical,
            "sentiment_data": sentiment,
            "market_status": get_market_status()
        })
    
    return analysis_result

def extract_ticker_from_message(message: str) -> Optional[str]:
    """Mesajdan ticker sembolünü çıkar"""
    import re
    # $TSLA, AAPL, Bitcoin gibi formatları yakala
    patterns = [
        r'\$([A-Z]{2,5})',  # $TSLA
        r'\b([A-Z]{2,5})\b',  # AAPL
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message.upper())
        if match:
            return match.group(1).replace('$', '')
    
    return None

def get_market_status():
    """Piyasa durumu"""
    return {
        "status": "open",  # open/closed
        "timezone": "Turkey/Istanbul",
        "next_session": "2025-01-30 10:00"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
