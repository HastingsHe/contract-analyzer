from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.models.database import get_db, CrawlerLog
from app.services.crawler_service import start_crawler, stop_crawler, get_crawler_status

router = APIRouter()

@router.get("/logs", response_model=List[dict])
def get_crawler_logs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取爬虫日志列表"""
    logs = db.query(CrawlerLog).order_by(CrawlerLog.start_time.desc()).offset(skip).limit(limit).all()
    return [{
        "id": log.id,
        "spider_name": log.spider_name,
        "start_time": log.start_time,
        "end_time": log.end_time,
        "status": log.status,
        "contracts_crawled": log.contracts_crawled,
        "log_message": log.log_message
    } for log in logs]

@router.get("/logs/{log_id}", response_model=dict)
def get_crawler_log(
    log_id: int,
    db: Session = Depends(get_db)
):
    """获取单个爬虫日志"""
    log = db.query(CrawlerLog).filter(CrawlerLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Crawler log not found")
    return {
        "id": log.id,
        "spider_name": log.spider_name,
        "start_time": log.start_time,
        "end_time": log.end_time,
        "status": log.status,
        "contracts_crawled": log.contracts_crawled,
        "log_message": log.log_message
    }

@router.post("/start")
def start_crawler_endpoint(
    spider_name: str = "contract_spider",
    db: Session = Depends(get_db)
):
    """启动爬虫"""
    try:
        crawler_id = start_crawler(spider_name, db)
        return {"message": "Crawler started successfully", "crawler_id": crawler_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting crawler: {str(e)}")

@router.post("/stop")
def stop_crawler_endpoint(
    spider_name: str = "contract_spider"
):
    """停止爬虫"""
    try:
        stop_crawler(spider_name)
        return {"message": "Crawler stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping crawler: {str(e)}")

@router.get("/status")
def get_crawler_status_endpoint(
    spider_name: str = "contract_spider"
):
    """获取爬虫状态"""
    return get_crawler_status(spider_name)
