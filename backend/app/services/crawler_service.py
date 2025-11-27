from datetime import datetime
from sqlalchemy.orm import Session
from app.models.database import CrawlerLog
import threading
import time
import logging

logger = logging.getLogger(__name__)

# 全局爬虫状态
crawler_status = {
    "running": False,
    "current_spider": None
}

def start_crawler(spider_name: str, db: Session) -> int:
    """启动爬虫"""
    # 检查是否已经在运行
    if crawler_status["running"]:
        raise Exception(f"Crawler {crawler_status['current_spider']} is already running")
    
    # 创建爬虫日志
    crawler_log = CrawlerLog(
        spider_name=spider_name,
        status="running",
        log_message=f"Crawler started: {spider_name}"
    )
    db.add(crawler_log)
    db.commit()
    db.refresh(crawler_log)
    
    # 更新全局爬虫状态
    crawler_status["running"] = True
    crawler_status["current_spider"] = spider_name
    
    # 在后台线程中执行爬虫
    threading.Thread(target=_run_crawler, args=(spider_name, crawler_log.id, db)).start()
    
    return crawler_log.id

def _run_crawler(spider_name: str, crawler_id: int, db: Session):
    """实际的爬虫执行逻辑"""
    try:
        logger.info(f"Starting crawler: {spider_name}")
        
        # 模拟爬虫运行
        contracts_crawled = 0
        
        # 模拟爬取10个合同
        for i in range(10):
            # 检查是否需要停止
            if not crawler_status["running"]:
                break
            
            # 模拟爬取耗时
            time.sleep(2)
            contracts_crawled += 1
            
            # 更新爬虫日志
            crawler_log = db.query(CrawlerLog).filter(CrawlerLog.id == crawler_id).first()
            if crawler_log:
                crawler_log.contracts_crawled = contracts_crawled
                crawler_log.log_message = f"Crawled {contracts_crawled} contracts"
                db.commit()
        
        # 更新爬虫状态为完成
        crawler_log = db.query(CrawlerLog).filter(CrawlerLog.id == crawler_id).first()
        if crawler_log:
            crawler_log.status = "completed"
            crawler_log.end_time = datetime.utcnow()
            crawler_log.contracts_crawled = contracts_crawled
            crawler_log.log_message = f"Crawler completed successfully, crawled {contracts_crawled} contracts"
            db.commit()
        
        # 更新全局爬虫状态
        crawler_status["running"] = False
        crawler_status["current_spider"] = None
        
        logger.info(f"Crawler completed: {spider_name}, crawled {contracts_crawled} contracts")
        
    except Exception as e:
        logger.error(f"Error during crawling: {e}")
        
        # 更新爬虫日志为失败状态
        crawler_log = db.query(CrawlerLog).filter(CrawlerLog.id == crawler_id).first()
        if crawler_log:
            crawler_log.status = "failed"
            crawler_log.end_time = datetime.utcnow()
            crawler_log.log_message = f"Crawler failed: {str(e)}"
            db.commit()
        
        # 更新全局爬虫状态
        crawler_status["running"] = False
        crawler_status["current_spider"] = None

def stop_crawler(spider_name: str):
    """停止爬虫"""
    if crawler_status["running"] and crawler_status["current_spider"] == spider_name:
        crawler_status["running"] = False
        logger.info(f"Stopping crawler: {spider_name}")
        return True
    return False

def get_crawler_status(spider_name: str) -> dict:
    """获取爬虫状态"""
    return {
        "running": crawler_status["running"] and crawler_status["current_spider"] == spider_name,
        "current_spider": crawler_status["current_spider"]
    }
