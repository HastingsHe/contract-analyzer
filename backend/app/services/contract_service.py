from typing import List
from app.schemas.contracts import ContractAnalysisResult, ContractClause
import re

# 简单的合同分析服务实现
def analyze_contract(contract_content: str) -> ContractAnalysisResult:
    """分析合同内容，识别有问题的条款"""
    
    # 示例实现：使用规则引擎识别常见问题条款
    problematic_clauses = []
    
    # 分割合同为条款
    clauses = split_contract_into_clauses(contract_content)
    
    # 检查每个条款
    for clause in clauses:
        risk_score = assess_clause_risk(clause)
        if risk_score > 0.5:
            clause_type = identify_clause_type(clause)
            risk_level = get_risk_level(risk_score)
            recommendation = get_recommendation(clause_type, risk_level)
            
            problematic_clauses.append(ContractClause(
                clause_text=clause,
                clause_type=clause_type,
                risk_level=risk_level,
                risk_score=risk_score,
                recommendation=recommendation
            ))
    
    # 计算整体风险分数
    if problematic_clauses:
        avg_risk_score = sum(clause.risk_score for clause in problematic_clauses) / len(problematic_clauses)
    else:
        avg_risk_score = 0.1  # 默认低风险
    
    # 生成摘要
    summary = generate_summary(problematic_clauses, avg_risk_score)
    
    return ContractAnalysisResult(
        risk_score=avg_risk_score,
        problematic_clauses=problematic_clauses,
        summary=summary
    )

def split_contract_into_clauses(contract_content: str) -> List[str]:
    """将合同分割为条款"""
    # 简单的条款分割逻辑，实际项目中可能需要更复杂的NLP处理
    clauses = []
    
    # 使用正则表达式匹配条款
    # 匹配常见的条款格式，如 "第X条"、"1. "、"一、" 等
    patterns = [
        r'(第\d+条.*?)(?=第\d+条|$)',  # 中文条款格式：第X条
        r'(\d+\. .*?)(?=\d+\. |$)',   # 数字编号：1. 
        r'([一二三四五六七八九十]+、.*?)(?=[一二三四五六七八九十]+、|$)',  # 中文数字：一、
        r'(ARTICLE \d+.*?)(?=ARTICLE \d+|$)',  # 英文条款：ARTICLE X
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, contract_content, re.DOTALL | re.IGNORECASE)
        clauses.extend(matches)
    
    # 如果没有匹配到条款，返回整个合同作为一个条款
    if not clauses:
        clauses = [contract_content]
    
    return clauses

def assess_clause_risk(clause: str) -> float:
    """评估条款风险分数（0-1之间）"""
    risk_score = 0.0
    
    # 风险关键词列表
    high_risk_keywords = [
        "不可抗力", "违约责任", "赔偿", "违约金", "解除合同",
        "保密", "知识产权", "争议解决", "管辖权", "仲裁",
        "诉讼", "赔偿限额", "免责", "担保", "保证",
        "竞业禁止", "排他性", "独占", "限制", "约束"
    ]
    
    # 计算风险关键词出现次数
    clause_lower = clause.lower()
    for keyword in high_risk_keywords:
        if keyword in clause_lower:
            risk_score += 0.1
    
    # 限制风险分数在0-1之间
    return min(max(risk_score, 0.0), 1.0)

def identify_clause_type(clause: str) -> str:
    """识别条款类型"""
    clause_lower = clause.lower()
    
    if any(keyword in clause_lower for keyword in ["保密", "confidentiality"]):
        return "保密条款"
    elif any(keyword in clause_lower for keyword in ["违约责任", "liability", "breach"]):
        return "违约责任条款"
    elif any(keyword in clause_lower for keyword in ["赔偿", "compensation"]):
        return "赔偿条款"
    elif any(keyword in clause_lower for keyword in ["不可抗力", "force majeure"]):
        return "不可抗力条款"
    elif any(keyword in clause_lower for keyword in ["争议解决", "dispute resolution", "仲裁", "arbitration"]):
        return "争议解决条款"
    elif any(keyword in clause_lower for keyword in ["知识产权", "intellectual property"]):
        return "知识产权条款"
    elif any(keyword in clause_lower for keyword in ["解除", "termination"]):
        return "合同解除条款"
    else:
        return "其他条款"

def get_risk_level(risk_score: float) -> str:
    """根据风险分数获取风险等级"""
    if risk_score >= 0.8:
        return "high"
    elif risk_score >= 0.5:
        return "medium"
    else:
        return "low"

def get_recommendation(clause_type: str, risk_level: str) -> str:
    """根据条款类型和风险等级获取建议"""
    recommendations = {
        "保密条款": {
            "high": "建议重新审查保密期限和范围，确保不超过合理限度。",
            "medium": "建议明确保密信息的定义和例外情况。",
            "low": "条款基本合理，可考虑添加保密信息的返还条款。"
        },
        "违约责任条款": {
            "high": "建议调整违约金比例，确保不超过实际损失的30%。",
            "medium": "建议明确违约责任的触发条件和计算方式。",
            "low": "条款基本合理，可考虑添加免责情形。"
        },
        "赔偿条款": {
            "high": "建议设置合理的赔偿限额，避免无限连带责任。",
            "medium": "建议明确赔偿范围和举证责任。",
            "low": "条款基本合理，可考虑添加间接损失的排除条款。"
        }
    }
    
    # 默认建议
    default_recommendations = {
        "high": "建议咨询专业律师，重新审查该条款。",
        "medium": "建议仔细审查条款内容，考虑是否需要修改。",
        "low": "条款基本合理，可以接受。"
    }
    
    return recommendations.get(clause_type, default_recommendations).get(risk_level, default_recommendations["medium"])

def generate_summary(problematic_clauses: List[ContractClause], avg_risk_score: float) -> str:
    """生成合同分析摘要"""
    if not problematic_clauses:
        return "合同整体风险较低，未发现明显问题条款。"
    
    high_risk_clauses = [clause for clause in problematic_clauses if clause.risk_level == "high"]
    medium_risk_clauses = [clause for clause in problematic_clauses if clause.risk_level == "medium"]
    
    summary = f"合同整体风险评估：{get_risk_level(avg_risk_score)}\n"
    summary += f"发现 {len(problematic_clauses)} 个有问题的条款，其中高风险 {len(high_risk_clauses)} 个，中风险 {len(medium_risk_clauses)} 个。\n\n"
    
    summary += "主要问题条款类型：\n"
    clause_types = {}
    for clause in problematic_clauses:
        clause_types[clause.clause_type] = clause_types.get(clause.clause_type, 0) + 1
    
    for clause_type, count in clause_types.items():
        summary += f"- {clause_type}：{count}个\n"
    
    summary += "\n建议：\n"
    if high_risk_clauses:
        summary += "1. 重点审查所有高风险条款，建议咨询专业律师。\n"
    if medium_risk_clauses:
        summary += "2. 仔细审查中风险条款，考虑是否需要修改。\n"
    summary += "3. 确保所有条款符合法律法规要求。\n"
    
    return summary
