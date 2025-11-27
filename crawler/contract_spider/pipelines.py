import scrapy
import json
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./contracts.db")

# 创建数据库引擎
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 合同数据库模型
class Contract(Base):
    __tablename__ = "crawled_contracts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    url = Column(String, unique=True, index=True)
    content = Column(Text)
    contract_type = Column(String, index=True)
    source = Column(String, index=True)
    crawled_at = Column(DateTime, default=datetime.utcnow)

# 创建数据库表
Base.metadata.create_all(bind=engine)

class ContractSpiderPipeline:
    """合同爬虫管道"""
    
    def __init__(self):
        # 初始化数据库会话
        self.db = SessionLocal()
        # 创建输出目录
        self.output_dir = "../../model/data/crawled_contracts"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_item(self, item, spider):
        """处理爬取到的合同项"""
        # 保存到数据库
        self.save_to_database(item)
        
        # 保存到文件
        self.save_to_file(item)
        
        return item
    
    def save_to_database(self, item):
        """将合同保存到数据库"""
        try:
            # 检查合同是否已存在
            existing_contract = self.db.query(Contract).filter(Contract.url == item['url']).first()
            if not existing_contract:
                # 创建新合同记录
                contract = Contract(
                    title=item['title'],
                    url=item['url'],
                    content=item['content'],
                    contract_type=item['contract_type'],
                    source=item['source'],
                    crawled_at=item['crawled_at']
                )
                self.db.add(contract)
                self.db.commit()
        except Exception as e:
            self.db.rollback()
            spider.logger.error(f"Error saving to database: {e}")
    
    def save_to_file(self, item):
        """将合同保存到文件"""
        try:
            # 创建文件名
            filename = f"{item['source']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # 保存为JSON格式
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dict(item), f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            spider.logger.error(f"Error saving to file: {e}")
    
    def close_spider(self, spider):
        """关闭爬虫时执行"""
        # 关闭数据库会话
        self.db.close()
        spider.logger.info("Pipeline closed, database session closed")

class DuplicatesPipeline:
    """去重管道"""
    
    def __init__(self):
        self.urls_seen = set()
    
    def process_item(self, item, spider):
        """检查合同是否已处理"""
        if item['url'] in self.urls_seen:
            spider.logger.info(f"Duplicate contract found: {item['url']}")
            raise scrapy.exceptions.DropItem(f"Duplicate item found: {item['url']}")
        else:
            self.urls_seen.add(item['url'])
            return item
