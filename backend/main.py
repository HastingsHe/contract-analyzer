from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import contracts, training, crawler
from app.models.database import engine, Base

# 创建数据库表
Base.metadata.create_all(bind=engine)

# 创建FastAPI应用
app = FastAPI(
    title="Contract Analyzer API",
    description="API for contract analysis, training monitoring, and crawler management",
    version="1.0.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(contracts.router, prefix="/api/contracts", tags=["contracts"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(crawler.router, prefix="/api/crawler", tags=["crawler"])

# 根路径
@app.get("/")
def root():
    return {"message": "Welcome to Contract Analyzer API"}

# 健康检查
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
