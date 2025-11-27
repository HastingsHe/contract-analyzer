from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 获取数据库URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./contract_analyzer.db")

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# 创建会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()

# 数据库依赖
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 合同模型
class Contract(Base):
    __tablename__ = "contracts"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content = Column(Text)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    analyzed = Column(Boolean, default=False)
    analysis_result = Column(Text, nullable=True)
    risk_score = Column(Float, nullable=True)
    problematic_clauses = Column(Text, nullable=True)

# 训练日志模型
class TrainingLog(Base):
    __tablename__ = "training_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, index=True)  # running, completed, failed
    epoch = Column(Integer, nullable=True)
    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    log_message = Column(Text, nullable=True)

# 爬虫日志模型
class CrawlerLog(Base):
    __tablename__ = "crawler_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    spider_name = Column(String, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, index=True)  # running, completed, failed
    contracts_crawled = Column(Integer, default=0)
    log_message = Column(Text, nullable=True)

# 合同条款模型
class ContractClause(Base):
    __tablename__ = "contract_clauses"
    
    id = Column(Integer, primary_key=True, index=True)
    contract_id = Column(Integer, index=True)
    clause_text = Column(Text)
    clause_type = Column(String, index=True)
    risk_level = Column(String, index=True)  # low, medium, high
    risk_score = Column(Float)
    recommendation = Column(Text, nullable=True)
