from fastapi import FastAPI

app = FastAPI(title="BORSA MCP API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "BORSA MCP API çalışıyor!", "status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "borsa-mcp"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
