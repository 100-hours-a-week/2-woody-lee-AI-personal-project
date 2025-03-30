from fastapi import FastAPI
from .routes import router

app = FastAPI()

# API 라우터 등록
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    