from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TrainingLogBase(BaseModel):
    model_name: str
    status: str
    log_message: Optional[str] = None

class TrainingLogCreate(TrainingLogBase):
    pass

class TrainingLogUpdate(BaseModel):
    status: Optional[str] = None
    epoch: Optional[int] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    log_message: Optional[str] = None

class TrainingLog(TrainingLogBase):
    id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    epoch: Optional[int] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    class Config:
        from_attributes = True

class TrainingStatus(BaseModel):
    is_running: bool
    current_epoch: Optional[int] = None
    current_loss: Optional[float] = None
    current_accuracy: Optional[float] = None

class ModelTrainRequest(BaseModel):
    model_name: str
    epochs: int = Field(default=10, ge=1, le=100)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
    batch_size: int = Field(default=32, ge=1, le=256)
