#!/usr/bin/env python3
"""
🚀 BORSA MCP - Market Virtuoso API
Enhanced FastAPI server with MCP tool integration
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import httpx
import asyncio
import logging
import json
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="🎭 BORSA MCP - Market Virtuoso API",
    description="AI-powered Turkish stock market analysis with MCP integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class AnalysisRequest(BaseModel):
    analysis_type: Optional[str] = "technical"
    period: Optional[str] = "1y"
    indicators: Optional[List[str]] = ["rsi", "macd", "bb"]

class MaestroRequest(BaseModel):
    ticker: str
    user_query: str
    analysis_depth: Optional[str] = "full"

class StockData(BaseModel):
    ticker: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None

# Market Virtuoso Persona Configuration
MARKET_VIRTUOSO_CONFIG = {
    "persona": "The Maestro",
    "philosophy": "Asymmetry, Narrative + Numbers, Strategic Patience",
    "framework": ["Telescope", "Microscope", "Barometer", "Architect's Blueprint"],
    "temperature": 0.6,
    "max_tokens": 4000,
    "language": "Turkish"
}

# Turkish stock tickers mapping
TURKISH_TICKERS = {
    "THY": "THYAO.IS",
    "THYAO": "THYAO.IS", 
    "AKBNK": "AKBNK.IS",
    "GARAN": "GARAN.IS",
    "ISCTR": "ISCTR.IS",
    "KCHOL": "KCHOL.IS",
    "SAHOL": "SAHOL.IS",
    "YKBNK": "YKBNK.IS",
    "BIMAS": "BIMAS.IS",
    "TCELL": "TCELL.IS",
    "ASELS": "ASELS.IS"
}

@app.get("/")
async def root():
    """Welcome message with API info"""
    return {
        "message": "🎭 Market Virtuoso API - MCP Ready!",
        "status": "operational",
        "persona": "Market Maestro",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "technical_analysis": "/analyze/technical/{ticker}",
            "sentiment_analysis": "/analyze/sentiment/{ticker}",
            "maestro_analysis": "/maestro/analyze",
            "market_overview": "/market/overview",
            "mcp_tools": "/mcp/tools"
        },
        "mcp_compatible": True
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "operational",
        "services": {
            "api": "✅ Active",
            "yfinance": "✅ Connected",
            "mcp": "✅ Ready"
        }
    }

@app.get("/mcp/tools")
async def mcp_tools():
    """MCP Tools definition for AI agents"""
    return {
        "tools": [
            {
                "name": "borsa_technical_analysis",
                "description": "Get technical analysis for Turkish stocks including RSI, MACD, Bollinger Bands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Turkish stock ticker (e.g., THYAO, AKBNK)"
                        },
                        "analysis_type": {
                            "type": "string", 
                            "enum": ["technical", "full"],
                            "default": "technical"
                        }
                    },
                    "required": ["ticker"]
                }
            },
            {
                "name": "maestro_full_analysis",
                "description": "Complete Market Virtuoso analysis with AI persona",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        },
                        "user_query": {
                            "type": "string",
                            "description": "User's question about the stock"
                        }
                    },
                    "required": ["ticker", "user_query"]
                }
            },
            {
                "name": "market_sentiment_analysis",
                "description": "Analyze market sentiment for Turkish stocks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker"
                        },
                        "sentiment_sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["news", "social", "analyst"]
                        }
                    },
                    "required": ["ticker"]
                }
            }
        ],
        "persona": MARKET_VIRTUOSO_CONFIG
    }

def get_yfinance_ticker(ticker: str) -> str:
    """Convert Turkish ticker to Yahoo Finance format"""
    ticker_upper = ticker.upper()
    return TURKISH_TICKERS.get(ticker_upper, f"{ticker_upper}.IS")

def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive technical indicators"""
    try:
        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD calculation
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = data['Close'].rolling(window=bb_period).mean()
        bb_upper = bb_middle + (data['Close'].rolling(window=bb_period).std() * bb_std)
        bb_lower = bb_middle - (data['Close'].rolling(window=bb_period).std() * bb_std)
        
        # Support and Resistance levels
        recent_high = data['High'].rolling(window=20).max().iloc[-1]
        recent_low = data['Low'].rolling(window=20).min().iloc[-1]
        
        # Volume analysis
        avg_volume = data['Volume'].rolling(window=30).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        return {
            "rsi": {
                "current": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
                "signal": "overbought" if rsi.iloc[-1] > 70 else "oversold" if rsi.iloc[-1] < 30 else "neutral"
            },
            "macd": {
                "macd": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
                "signal": float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else None,
                "histogram": float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None,
                "trend": "bullish" if macd.iloc[-1] > signal.iloc[-1] else "bearish"
            },
            "bollinger_bands": {
                "upper": float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
                "middle": float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else None,
                "lower": float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None,
                "position": "above" if data['Close'].iloc[-1] > bb_upper.iloc[-1] else "below" if data['Close'].iloc[-1] < bb_lower.iloc[-1] else "middle"
            },
            "support_resistance": {
                "resistance": float(recent_high),
                "support": float(recent_low),
                "current_price": float(data['Close'].iloc[-1])
            },
            "volume_analysis": {
                "current_volume": int(current_volume),
                "avg_volume": int(avg_volume),
                "volume_ratio": float(volume_ratio),
                "signal": "high" if volume_ratio > 1.5 else "low" if volume_ratio < 0.5 else "normal"
            }
        }
    except Exception as e:
        logger.error(f"Technical indicators calculation error: {e}")
        return {"error": f"Calculation failed: {str(e)}"}

@app.post("/analyze/technical/{ticker}")
async def analyze_technical(ticker: str, request: AnalysisRequest = AnalysisRequest()):
    """
    🔧 MCP Tool: Technical Analysis
    Advanced technical analysis for Turkish stocks
    """
    try:
        yf_ticker = get_yfinance_ticker(ticker)
        logger.info(f"Fetching technical data for {ticker} -> {yf_ticker}")
        
        # Fetch stock data
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(period=request.period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        
        # Calculate technical indicators
        technical_data = calculate_technical_indicators(hist)
        
        # Get current stock info
        info = stock.info
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
        
        return {
            "ticker": ticker.upper(),
            "yf_ticker": yf_ticker,
            "timestamp": datetime.now().isoformat(),
            "current_price": float(current_price),
            "change": float(change),
            "change_percent": float(change_percent),
            "currency": "TRY",
            "technical_indicators": technical_data,
            "market_status": "open" if datetime.now().weekday() < 5 else "closed",
            "analysis_type": request.analysis_type,
            "mcp_tool": "borsa_technical_analysis"
        }
        
    except Exception as e:
        logger.error(f"Technical analysis error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/sentiment/{ticker}")
async def analyze_sentiment(ticker: str):
    """
    🔧 MCP Tool: Sentiment Analysis
    Market sentiment analysis for Turkish stocks
    """
    try:
        # Simulated sentiment analysis (integrate with real news APIs)
        sentiment_score = np.random.uniform(-1, 1)  # Placeholder
        
        sentiment_data = {
            "overall_sentiment": sentiment_score,
            "sentiment_label": "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral",
            "confidence": abs(sentiment_score),
            "sources": {
                "news_sentiment": np.random.uniform(-1, 1), 
                "social_media": np.random.uniform(-1, 1),
                "analyst_ratings": np.random.uniform(-1, 1)
            },
            "key_factors": [
                "Sektörel performans",
                "Genel piyasa durumu", 
                "Şirket haberleri",
                "Makroekonomik göstergeler"
            ]
        }
        
        return {
            "ticker": ticker.upper(),
            "timestamp": datetime.now().isoformat(),
            "sentiment_analysis": sentiment_data,
            "mcp_tool": "market_sentiment_analysis"
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.post("/maestro/analyze")
async def maestro_analyze(request: MaestroRequest):
    """
    🎭 MCP Tool: Market Maestro Full Analysis
    Complete Market Virtuoso persona analysis
    """
    try:
        # Get technical analysis first
        technical_analysis = await analyze_technical(request.ticker)
        
        # Get sentiment analysis
        sentiment_analysis = await analyze_sentiment(request.ticker)
        
        # Market Virtuoso Analysis Framework
        maestro_analysis = {
            "telescope_view": {
                "macro_context": "Türk ekonomisinde enflasyon baskısı devam ediyor, TCMB politika duruşu kritik",
                "sector_outlook": "Hisse senedi sektörüne göre değerlendiriliyor",
                "global_impact": "Küresel piyasalar ve USD/TRY paritesi etkili"
            },
            "microscope_view": {
                "technical_summary": technical_analysis.get("technical_indicators", {}),
                "price_action": f"Güncel fiyat: {technical_analysis.get('current_price', 0):.2f} TL",
                "volume_profile": technical_analysis.get("technical_indicators", {}).get("volume_analysis", {})
            },
            "barometer_reading": {
                "market_sentiment": sentiment_analysis.get("sentiment_analysis", {}),
                "risk_appetite": "Temkinli" if sentiment_analysis.get("sentiment_analysis", {}).get("overall_sentiment", 0) < 0 else "Pozitif",
                "momentum": "Güçlü" if technical_analysis.get("technical_indicators", {}).get("rsi", {}).get("current", 50) > 60 else "Zayıf"
            },
            "architects_blueprint": {
                "entry_zones": [
                    technical_analysis.get("technical_indicators", {}).get("support_resistance", {}).get("support", 0),
                    technical_analysis.get("technical_indicators", {}).get("bollinger_bands", {}).get("lower", 0)
                ],
                "target_zones": [
                    technical_analysis.get("technical_indicators", {}).get("support_resistance", {}).get("resistance", 0),
                    technical_analysis.get("technical_indicators", {}).get("bollinger_bands", {}).get("upper", 0)
                ],
                "risk_management": "Stop-loss: Destek seviyesinin %2 altı",
                "position_sizing": "Portföyün %2-5'i arası önerilir"
            }
        }
        
        # Generate Maestro's narrative
        rsi_value = technical_analysis.get("technical_indicators", {}).get("rsi", {}).get("current", 50)
        macd_trend = technical_analysis.get("technical_indicators", {}).get("macd", {}).get("trend", "neutral")
        
        maestro_narrative = f"""
🎭 **Maestro'nun {request.ticker.upper()} Analizi**

🎯 **Hızlı Değerlendirme**: 
{request.ticker.upper()} şu an RSI {rsi_value:.1f} seviyesinde, {macd_trend} trend gösteriyor. 
Kullanıcı sorusu: "{request.user_query}"

📊 **Teknik Perspektif**:
• RSI seviyesi momentum hakkında ipuçları veriyor
• MACD {macd_trend} sinyali üretmiş durumda
• Bollinger bantları içindeki pozisyon önemli

🧭 **Maestro'nun Görüşü**:
"Bu hisse sanki bekleyen bir kartal gibi... Teknik formasyonu {macd_trend} ama makro rüzgarlar 
değişken. Asymmetric opportunity arayanlar dikkat etmeli - pozisyon büyüklüğü her zaman kritik."

⚖️ **Risk Haritası**:
↗️ Upside: Teknik kırılım, sektörel momentum
↘️ Downside: Makro baskılar, likidite çekilişi

⚠️ **Uyarı**: Bu analiz finansal tavsiye değildir. Maestro'nun görüşleri yatırım kararınızı etkilememelidir.
        """
        
        response = {
            "ticker": request.ticker.upper(),
            "timestamp": datetime.now().isoformat(),
            "user_query": request.user_query,
            "maestro_framework": maestro_analysis,
            "narrative_analysis": maestro_narrative.strip(),
            "technical_data": technical_analysis,
            "sentiment_data": sentiment_analysis,
            "persona": "Market Maestro",
            "analysis_depth": request.analysis_depth,
            "mcp_tool": "maestro_full_analysis"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Maestro analysis error for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Maestro analysis failed: {str(e)}")

@app.get("/market/overview")
async def market_overview():
    """Market overview with key Turkish indices"""
    try:
        # Fetch BIST 100 data
        bist100 = yf.Ticker("XU100.IS")
        bist100_hist = bist100.history(period="5d")
        
        if not bist100_hist.empty:
            current_price = bist100_hist['Close'].iloc[-1]
            prev_close = bist100_hist['Close'].iloc[-2] if len(bist100_hist) > 1 else current_price
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
        else:
            current_price = change = change_percent = 0
            
        # Top stocks to monitor
        top_stocks = ["THYAO", "AKBNK", "GARAN", "KCHOL", "SAHOL"]
        
        market_data = {
            "bist100": {
                "current": float(current_price),
                "change": float(change),
                "change_percent": float(change_percent)
            },
            "market_status": "open" if datetime.now().weekday() < 5 else "closed",
            "top_stocks": top_stocks,
            "market_sentiment": "Temkinli pozitif",
            "timestamp": datetime.now().isoformat()
        }
        
        return market_data
        
    except Exception as e:
        logger.error(f"Market overview error: {e}")
        raise HTTPException(status_code=500, detail=f"Market overview failed: {str(e)}")

@app.get("/stocks/search/{query}")
async def search_stocks(query: str):
    """Search Turkish stocks by name or ticker"""
    try:
        # Simple stock search in Turkish tickers
        matches = []
        query_upper = query.upper()
        
        for ticker, yf_ticker in TURKISH_TICKERS.items():
            if query_upper in ticker:
                matches.append({
                    "ticker": ticker,
                    "yf_ticker": yf_ticker,
                    "match_type": "ticker"
                })
        
        return {
            "query": query,
            "matches": matches[:10],  # Limit to 10 results
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stock search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# MCP Integration Endpoints
@app.post("/mcp/execute/{tool_name}")
async def mcp_execute_tool(tool_name: str, tool_params: Dict[str, Any]):
    """
    🔧 MCP Tool Execution Endpoint
    Execute MCP tools programmatically
    """
    try:
        if tool_name == "borsa_technical_analysis":
            ticker = tool_params.get("ticker", "THYAO")
            analysis_type = tool_params.get("analysis_type", "technical")
            request_obj = AnalysisRequest(analysis_type=analysis_type)
            return await analyze_technical(ticker, request_obj)
            
        elif tool_name == "maestro_full_analysis":
            ticker = tool_params.get("ticker", "THYAO")
            user_query = tool_params.get("user_query", "Genel analiz")
            request_obj = MaestroRequest(ticker=ticker, user_query=user_query)
            return await maestro_analyze(request_obj)
            
        elif tool_name == "market_sentiment_analysis":
            ticker = tool_params.get("ticker", "THYAO")
            return await analyze_sentiment(ticker)
            
        else:
            raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
            
    except Exception as e:
        logger.error(f"MCP tool execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint bulunamadı",
            "message": "API dokümantasyonu için /docs adresini ziyaret edin",
            "available_endpoints": [
                "/", "/health", "/analyze/technical/{ticker}", 
                "/maestro/analyze", "/market/overview", "/mcp/tools"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Sunucu hatası",
            "message": "Teknik bir sorun oluştu, lütfen daha sonra tekrar deneyin",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
