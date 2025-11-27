from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session
from app.schemas.training import ModelTrainRequest, TrainingStatus
from app.models.database import TrainingLog
import threading
import time
import logging

logger = logging.getLogger(__name__)

# 全局训练状态
global_training_status = {
    "is_running": False,
    "current_epoch": None,
    "current_loss": None,
    "current_accuracy": None
}

def start_training(train_request: ModelTrainRequest, db: Session) -> int:
    """启动模型训练"""
    # 创建训练日志
    training_log = TrainingLog(
        model_name=train_request.model_name,
        status="running",
        log_message=f"Training started with epochs={train_request.epochs}, lr={train_request.learning_rate}, batch_size={train_request.batch_size}"
    )
    db.add(training_log)
    db.commit()
    db.refresh(training_log)
    
    # 更新全局训练状态
    global global_training_status
    global_training_status["is_running"] = True
    global_training_status["current_epoch"] = 0
    global_training_status["current_loss"] = None
    global_training_status["current_accuracy"] = None
    
    # 在后台线程中执行训练
    threading.Thread(target=_train_model, args=(train_request, training_log.id, db)).start()
    
    return training_log.id

def _train_model(train_request: ModelTrainRequest, training_id: int, db: Session):
    """实际的模型训练逻辑"""
    try:
        logger.info(f"Starting training for model {train_request.model_name}")
        
        # 模拟训练过程
        for epoch in range(1, train_request.epochs + 1):
            # 更新全局训练状态
            global global_training_status
            global_training_status["current_epoch"] = epoch
            
            # 模拟训练损失和准确率
            loss = 0.5 * (1 - epoch / train_request.epochs)
            accuracy = 0.5 + 0.4 * (epoch / train_request.epochs)
            precision = 0.5 + 0.35 * (epoch / train_request.epochs)
            recall = 0.5 + 0.35 * (epoch / train_request.epochs)
            f1_score = 0.5 + 0.35 * (epoch / train_request.epochs)
            
            global_training_status["current_loss"] = loss
            global_training_status["current_accuracy"] = accuracy
            
            # 更新训练日志
            training_log = db.query(TrainingLog).filter(TrainingLog.id == training_id).first()
            if training_log:
                training_log.epoch = epoch
                training_log.loss = loss
                training_log.accuracy = accuracy
                training_log.precision = precision
                training_log.recall = recall
                training_log.f1_score = f1_score
                training_log.log_message = f"Epoch {epoch}/{train_request.epochs} completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
                db.commit()
            
            # 模拟训练耗时
            time.sleep(1)
        
        # 训练完成
        training_log = db.query(TrainingLog).filter(TrainingLog.id == training_id).first()
        if training_log:
            training_log.status = "completed"
            training_log.end_time = datetime.utcnow()
            training_log.log_message = f"Training completed successfully! Final accuracy: {accuracy:.4f}"
            db.commit()
        
        # 更新全局训练状态
        global_training_status["is_running"] = False
        
        logger.info(f"Training completed for model {train_request.model_name}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        
        # 更新训练日志为失败状态
        training_log = db.query(TrainingLog).filter(TrainingLog.id == training_id).first()
        if training_log:
            training_log.status = "failed"
            training_log.end_time = datetime.utcnow()
            training_log.log_message = f"Training failed: {str(e)}"
            db.commit()
        
        # 更新全局训练状态
        global global_training_status
        global_training_status["is_running"] = False

def get_training_status() -> TrainingStatus:
    """获取当前训练状态"""
    global global_training_status
    return TrainingStatus(**global_training_status)
