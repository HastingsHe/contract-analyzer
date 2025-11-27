from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ContractBase(BaseModel):
    filename: str
    content: str

class ContractCreate(ContractBase):
    pass

class ContractClause(BaseModel):
    clause_text: str
    clause_type: str
    risk_level: str
    risk_score: float
    recommendation: Optional[str] = None

class ContractAnalysisResult(BaseModel):
    risk_score: float
    problematic_clauses: List[ContractClause]
    summary: str

class Contract(ContractBase):
    id: int
    uploaded_at: datetime
    analyzed: bool
    analysis_result: Optional[str] = None
    risk_score: Optional[float] = None
    problematic_clauses: Optional[str] = None

    class Config:
        from_attributes = True

class ContractAnalysisRequest(BaseModel):
    contract_id: int

class ContractAnalysisResponse(BaseModel):
    contract_id: int
    analysis_result: ContractAnalysisResult
    analyzed_at: datetime
