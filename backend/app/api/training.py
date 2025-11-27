from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.models.database import get_db, TrainingLog
from app.schemas.training import (
    TrainingLog as TrainingLogSchema,
    TrainingLogCreate,
    TrainingLogUpdate,
    TrainingStatus,
    ModelTrainRequest
)
from app.services.training_service import start_training, get_training_status

router = APIRouter()

@router.post("/logs", response_model=TrainingLogSchema)
def create_training_log(
    training_log: TrainingLogCreate,
    db: Session = Depends(get_db)
):
    """创建训练日志"""
    db_log = TrainingLog(**training_log.model_dump())
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

@router.get("/logs", response_model=List[TrainingLogSchema])
def get_training_logs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取训练日志列表"""
    logs = db.query(TrainingLog).order_by(TrainingLog.start_time.desc()).offset(skip).limit(limit).all()
    return logs

@router.get("/logs/{log_id}", response_model=TrainingLogSchema)
def get_training_log(
    log_id: int,
    db: Session = Depends(get_db)
):
    """获取单个训练日志"""
    log = db.query(TrainingLog).filter(TrainingLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Training log not found")
    return log

@router.put("/logs/{log_id}", response_model=TrainingLogSchema)
def update_training_log(
    log_id: int,
    training_log: TrainingLogUpdate,
    db: Session = Depends(get_db)
):
    """更新训练日志"""
    db_log = db.query(TrainingLog).filter(TrainingLog.id == log_id).first()
    if not db_log:
        raise HTTPException(status_code=404, detail="Training log not found")
    
    update_data = training_log.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_log, field, value)
    
    db.commit()
    db.refresh(db_log)
    return db_log

@router.post("/start")
def start_model_training(
    train_request: ModelTrainRequest,
    db: Session = Depends(get_db)
):
    """启动模型训练"""
    try:
        # 调用训练服务
        training_id = start_training(train_request, db)
        return {"message": "Training started successfully", "training_id": training_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

@router.get("/status", response_model=TrainingStatus)
def get_training_status_endpoint():
    """获取当前训练状态"""
    return get_training_status()

@router.get("/latest", response_model=TrainingLogSchema)
def get_latest_training_log(
    db: Session = Depends(get_db)
):
    """获取最新的训练日志"""
    log = db.query(TrainingLog).order_by(TrainingLog.start_time.desc()).first()
    if not log:
        raise HTTPException(status_code=404, detail="No training logs found")
    return log
