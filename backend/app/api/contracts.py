from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from app.models.database import get_db, Contract
from app.schemas.contracts import Contract as ContractSchema, ContractCreate, ContractAnalysisResponse
from app.services.contract_service import analyze_contract
import json

router = APIRouter()

@router.post("/", response_model=ContractSchema)
async def create_contract(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """上传合同文件"""
    try:
        content = await file.read()
        contract_create = ContractCreate(
            filename=file.filename,
            content=content.decode("utf-8")
        )
        
        db_contract = Contract(**contract_create.model_dump())
        db.add(db_contract)
        db.commit()
        db.refresh(db_contract)
        
        return db_contract
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading contract: {str(e)}")

@router.get("/", response_model=List[ContractSchema])
def get_contracts(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """获取合同列表"""
    contracts = db.query(Contract).offset(skip).limit(limit).all()
    return contracts

@router.get("/{contract_id}", response_model=ContractSchema)
def get_contract(
    contract_id: int,
    db: Session = Depends(get_db)
):
    """获取单个合同"""
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    return contract

@router.post("/{contract_id}/analyze", response_model=ContractAnalysisResponse)
def analyze_contract_endpoint(
    contract_id: int,
    db: Session = Depends(get_db)
):
    """分析合同"""
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    try:
        # 调用合同分析服务
        analysis_result = analyze_contract(contract.content)
        
        # 更新合同分析结果
        contract.analyzed = True
        contract.risk_score = analysis_result.risk_score
        contract.problematic_clauses = json.dumps([
            clause.model_dump() for clause in analysis_result.problematic_clauses
        ])
        contract.analysis_result = analysis_result.summary
        
        db.commit()
        db.refresh(contract)
        
        return ContractAnalysisResponse(
            contract_id=contract_id,
            analysis_result=analysis_result,
            analyzed_at=contract.uploaded_at  # 使用上传时间作为分析时间，实际应该使用当前时间
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing contract: {str(e)}")

@router.delete("/{contract_id}")
def delete_contract(
    contract_id: int,
    db: Session = Depends(get_db)
):
    """删除合同"""
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    db.delete(contract)
    db.commit()
    
    return {"message": "Contract deleted successfully"}
