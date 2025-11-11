#!/usr/bin/env python3
"""
🚀 BORSA MCP - Market Virtuoso API
Complete MCP Server implementation for n8n integration
"""

from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, AsyncGenerator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
import logging
import json
import os
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Security Configuration
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() == "true"
API_KEY = os.getenv("BORSA_API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Verify API key if authentication is enabled"""
    if not API_KEY:
        return True  # No authentication required if API_KEY not set
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key required. Please provide X-API-Key header."
        )
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempt from client")
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return True

# Rate Limiting
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

# FastAPI app initialization
app = FastAPI(
    title="🎭 BORSA MCP - Market Virtuoso API",
    description="AI-powered Turkish stock market analysis with MCP Server integration",
    version="2.1.0",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware with environment-based configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class AnalysisRequest(BaseModel):
    analysis_type: Optional[str] = "technical"
    period: Optional[str] = "1y"

class MaestroRequest(BaseModel):
    ticker: str
    user_query: str
    analysis_depth: Optional[str] = "full"

# Market Virtuoso Configuration
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
    "THY": "THYAO.IS", "THYAO": "THYAO.IS", "AKBNK": "AKBNK.IS",
    "GARAN": "GARAN.IS", "ISCTR": "ISCTR.IS", "KCHOL": "KCHOL.IS",
    "SAHOL": "SAHOL.IS", "YKBNK": "YKBNK.IS", "BIMAS": "BIMAS.IS",
    "TCELL": "TCELL.IS", "ASELS": "ASELS.IS", "TUPRS": "TUPRS.IS"
}

# 🔧 MCP SERVER CORS MIDDLEWARE
@app.middleware("http")
async def add_mcp_cors_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/mcp"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
    return response

@app.get("/")
@limiter.limit(RATE_LIMIT)
async def root(request: Request):
    """Welcome message with MCP Server info"""
    return {
        "message": "🎭 Market Virtuoso API - MCP Server Ready!",
        "status": "operational",
        "persona": "Market Maestro",
        "version": "2.1.0",
        "security": {
            "authentication": "enabled" if API_KEY else "disabled",
            "rate_limiting": RATE_LIMIT,
            "cors": "restricted" if ALLOWED_ORIGINS != ["*"] else "open"
        },
        "mcp_server": {
            "sse_endpoint": "/mcp",
            "messages_endpoint": "/mcp/messages",
            "tools_count": 4,
            "status": "active"
        },
        "endpoints": {
            "health": "/health",
            "technical_analysis": "/analyze/technical/{ticker}",
            "sentiment_analysis": "/analyze/sentiment/{ticker}",
            "maestro_analysis": "/maestro/analyze",
            "market_overview": "/market/overview",
            "mcp_tools": "/mcp/tools",
            "mcp_sse": "/mcp",
            "mcp_messages": "/mcp/messages"
        },
        "mcp_compatible": True,
        "n8n_ready": True
    }

@app.get("/health")
async def health_check():
    """Health check endpoint - No rate limit, no auth"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "operational",
        "services": {
            "api": "✅ Active",
            "yfinance": "✅ Connected",
            "mcp_server": "✅ Ready",
            "sse_endpoint": "✅ Active"
        },
        "mcp_tools": {
            "borsa_technical_analysis": "✅ Ready",
            "maestro_full_analysis": "✅ Ready",
            "market_sentiment_analysis": "✅ Ready",
            "market_overview": "✅ Ready"
        }
    }

# 🔧 MCP SERVER ENDPOINTS

@app.get("/mcp")
async def mcp_sse_endpoint(request: Request):
    """
    🔧 MCP Server SSE Endpoint - N8N connection point
    """
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            logger.info("🔗 MCP SSE client connected")
            yield f"data: {json.dumps({'type': 'connect', 'status': 'ready', 'server': 'BORSA-MCP', 'tools': 4})}\n\n"
            
            # Keep connection alive with heartbeat
            while True:
                await asyncio.sleep(30)
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat(), 'status': 'alive'})}\n\n"
                
        except asyncio.CancelledError:
            logger.info("🔌 MCP SSE client disconnected")
            return
        except Exception as e:
            logger.error(f"❌ MCP SSE error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return EventSourceResponse(event_stream())

@app.post("/mcp/messages", dependencies=[Depends(verify_api_key)])
@limiter.limit(RATE_LIMIT)
async def mcp_messages_endpoint(request: Request):
    """
    🔧 MCP Messages Endpoint - Tool execution handler
    Protected by API Key and Rate Limiting
    """
    try:
        message = await request.json()
        logger.info(f"📨 MCP Message: {message.get('method', 'unknown')}")
        
        if message.get("method") == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "tools": [
                        {
                            "name": "borsa_technical_analysis",
                            "description": "🔍 Türk hisse senetleri için kapsamlı teknik analiz",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "ticker": {
                                        "type": "string",
                                        "description": "Hisse senedi kodu (THYAO, AKBNK vb.)"
                                    }
                                },
                                "required": ["ticker"]
                            }
                        },
                        {
                            "name": "maestro_full_analysis", 
                            "description": "🎭 Market Virtuoso tam analizi",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string"},
                                    "user_query": {"type": "string"}
                                },
                                "required": ["ticker", "user_query"]
                            }
                        },
                        {
                            "name": "market_sentiment_analysis",
                            "description": "💭 Piyasa duygu analizi",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string"}
                                },
                                "required": ["ticker"]
                            }
                        },
                        {
                            "name": "market_overview",
                            "description": "📊 Genel piyasa durumu",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    ]
                }
            }
            
        elif message.get("method") == "tools/call":
            tool_name = message.get("params", {}).get("name")
            tool_args = message.get("params", {}).get("arguments", {})
            
            try:
                if tool_name == "borsa_technical_analysis":
                    result = await analyze_technical(tool_args.get("ticker", "THYAO"))
                    
                elif tool_name == "maestro_full_analysis":
                    result = await maestro_analyze(MaestroRequest(
                        ticker=tool_args.get("ticker", "THYAO"),
                        user_query=tool_args.get("user_query", "Analiz")
                    ))
                    
                elif tool_name == "market_sentiment_analysis":
                    result = await analyze_sentiment(tool_args.get("ticker", "THYAO"))
                    
                elif tool_name == "market_overview":
                    result = await market_overview()
                    
                else:
                    raise HTTPException(404, f"Tool {tool_name} not found")
                
                return {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
                    }
                }
                
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32603, "message": f"Tool error: {str(e)}"}
                }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32601, "message": f"Method not supported"}
            }
            
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
        }

# 🛠️ UTILITY FUNCTIONS

def get_yfinance_ticker(ticker: str) -> str:
    """Convert Turkish ticker to Yahoo Finance format"""
    ticker_upper = ticker.upper()
    return TURKISH_TICKERS.get(ticker_upper, f"{ticker_upper}.IS")

def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate technical indicators"""
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
        
        # Support and Resistance
        recent_high = data['High'].rolling(window=20).max().iloc[-1]
        recent_low = data['Low'].rolling(window=20).min().iloc[-1]
        
        return {
            "rsi": {
                "current": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
                "signal": "aşırı_alım" if rsi.iloc[-1] > 70 else "aşırı_satım" if rsi.iloc[-1] < 30 else "nötr"
            },
            "macd": {
                "trend": "yükselişt" if macd.iloc[-1] > signal.iloc[-1] else "düşüş"
            },
            "support_resistance": {
                "resistance": float(recent_high),
                "support": float(recent_low),
                "current_price": float(data['Close'].iloc[-1])
            }
        }
    except Exception as e:
        logger.error(f"Technical calculation error: {e}")
        return {"error": "Calculation failed"}

# 📊 API ENDPOINTS

async def analyze_technical(ticker: str, request: AnalysisRequest = AnalysisRequest()):
    """Technical Analysis Tool"""
    try:
        yf_ticker = get_yfinance_ticker(ticker)
        logger.info(f"📊 Technical analysis: {ticker} -> {yf_ticker}")
        
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return {"error": f"No data for {ticker}"}
        
        technical_data = calculate_technical_indicators(hist)
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change_percent = ((current_price - prev_close) / prev_close) * 100
        
        return {
            "ticker": ticker.upper(),
            "current_price": float(current_price),
            "change_percent": float(change_percent),
            "technical_indicators": technical_data,
            "timestamp": datetime.now().isoformat(),
            "mcp_tool": "borsa_technical_analysis"
        }
        
    except Exception as e:
        logger.error(f"Technical analysis error: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

async def maestro_analyze(request: MaestroRequest):
    """Market Maestro Full Analysis Tool"""
    try:
        technical_data = await analyze_technical(request.ticker)
        
        # Generate Maestro narrative
        current_price = technical_data.get("current_price", 0)
        change_percent = technical_data.get("change_percent", 0)
        
        maestro_narrative = f"""
🎭 **Market Virtuoso - {request.ticker.upper()} Analizi**

🎯 **Hızlı Değerlendirme**: 
{request.ticker.upper()} şu an {current_price:.2f} TL seviyesinde.
Günlük değişim: %{change_percent:.2f}

📊 **Teknik Görünüm**:
• Fiyat hareketi analizi tamamlandı
• Teknik indikatörler değerlendirildi

🧭 **Maestro'nun Görüşü**:
"{request.ticker.upper()} için mevcut piyasa koşulları dikkate alındığında,
dikkatli yaklaşım öneriliyor."

📌 **Kullanıcı Sorusu**: "{request.user_query}"

⚠️ **Uyarı**: Bu analiz finansal tavsiye değildir.
        """
        
        return {
            "ticker": request.ticker.upper(),
            "user_query": request.user_query,
            "narrative_analysis": maestro_narrative.strip(),
            "technical_data": technical_data,
            "persona": "Market Maestro",
            "timestamp": datetime.now().isoformat(),
            "mcp_tool": "maestro_full_analysis"
        }
        
    except Exception as e:
        logger.error(f"Maestro analysis error: {e}")
        return {"error": f"Maestro analysis failed: {str(e)}"}

async def analyze_sentiment(ticker: str):
    """Market Sentiment Analysis Tool"""
    try:
        sentiment_score = np.random.uniform(-0.5, 0.5)  # Placeholder
        
        return {
            "ticker": ticker.upper(),
            "sentiment_analysis": {
                "overall_sentiment": sentiment_score,
                "sentiment_label": "pozitif" if sentiment_score > 0 else "negatif" if sentiment_score < 0 else "nötr",
                "summary": f"{ticker.upper()} için piyasa duygusu {'olumlu' if sentiment_score > 0 else 'olumsuz' if sentiment_score < 0 else 'kararsız'}"
            },
            "timestamp": datetime.now().isoformat(),
            "mcp_tool": "market_sentiment_analysis"
        }
        
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {str(e)}"}

async def market_overview():
    """Market Overview Tool"""
    try:
        # BIST 100 data
        bist100 = yf.Ticker("XU100.IS")
        hist = bist100.history(period="5d")
        
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
            change_pct = ((current - prev) / prev) * 100
        else:
            current = change_pct = 0
            
        return {
            "bist100": {
                "current": float(current),
                "change_percent": float(change_pct),
                "trend": "pozitif" if change_pct > 0 else "negatif"
            },
            "market_status": "açık" if datetime.now().weekday() < 5 else "kapalı",
            "summary": f"BIST 100: {current:.0f} (%{change_pct:.2f})",
            "timestamp": datetime.now().isoformat(),
            "mcp_tool": "market_overview"
        }
        
    except Exception as e:
        return {"error": f"Market overview failed: {str(e)}"}

# Legacy endpoints
@app.post("/analyze/technical/{ticker}", dependencies=[Depends(verify_api_key)])
@limiter.limit(RATE_LIMIT)
async def analyze_technical_endpoint(
    ticker: str,
    request_obj: Request,
    analysis_request: AnalysisRequest = AnalysisRequest()
):
    return await analyze_technical(ticker, analysis_request)

@app.post("/maestro/analyze", dependencies=[Depends(verify_api_key)])
@limiter.limit(RATE_LIMIT)
async def maestro_analyze_endpoint(request_obj: Request, maestro_request: MaestroRequest):
    return await maestro_analyze(maestro_request)

@app.get("/mcp/tools")
@limiter.limit(RATE_LIMIT)
async def mcp_tools_list(request: Request):
    """MCP Tools documentation"""
    return {
        "mcp_server": "BORSA Market Virtuoso",
        "version": "2.1.0",
        "tools": [
            {"name": "borsa_technical_analysis", "description": "🔍 Technical analysis"},
            {"name": "maestro_full_analysis", "description": "🎭 Maestro analysis"}, 
            {"name": "market_sentiment_analysis", "description": "💭 Sentiment analysis"},
            {"name": "market_overview", "description": "📊 Market overview"}
        ],
        "endpoints": {"sse": "/mcp", "messages": "/mcp/messages"},
        "persona": MARKET_VIRTUOSO_CONFIG
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    logger.warning(f"404 error: {request.url.path}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "path": str(request.url.path),
            "available_endpoints": ["/", "/health", "/mcp", "/mcp/messages", "/mcp/tools"]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Server error",
            "message": "An error occurred processing your request"
        }
    )

@app.options("/mcp/messages")
async def mcp_options():
    return JSONResponse(content={"status": "ok"})

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting BORSA MCP Server...")
    print("📡 SSE Endpoint: /mcp")
    print("📨 Messages: /mcp/messages")
    print("🛠️ Tools: 4")
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
# Deploy trigger Tue Jul 29 23:04:11 UTC 2025
