import streamlit as st
import requests
import json

# 设置页面标题和布局
st.set_page_config(
    page_title="合同分析系统",
    page_icon="📝",
    layout="wide"
)

# API 基础 URL
API_BASE_URL = "http://localhost:8000/api"

# 页面标题
st.title("📝 合同分析系统")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("功能菜单")
    menu_option = st.radio(
        "选择功能",
        ["上传合同", "查看历史", "分析结果"]
    )

# 上传合同页面
if menu_option == "上传合同":
    st.header("上传合同文件")
    
    # 文件上传组件
    uploaded_file = st.file_uploader("选择合同文件", type=["txt", "pdf", "doc", "docx"])
    
    if uploaded_file is not None:
        st.success(f"文件上传成功: {uploaded_file.name}")
        
        # 显示文件内容预览
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            st.subheader("文件内容预览")
            st.text_area("", content, height=200)
        
        # 分析按钮
        if st.button("开始分析", type="primary"):
            with st.spinner("正在分析合同..."):
                # 准备文件数据
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                try:
                    # 调用 API 上传合同
                    upload_response = requests.post(f"{API_BASE_URL}/contracts/", files=files)
                    upload_response.raise_for_status()
                    
                    contract_data = upload_response.json()
                    contract_id = contract_data["id"]
                    
                    st.success(f"合同上传成功，ID: {contract_id}")
                    
                    # 调用 API 分析合同
                    analyze_response = requests.post(f"{API_BASE_URL}/contracts/{contract_id}/analyze")
                    analyze_response.raise_for_status()
                    
                    analysis_result = analyze_response.json()
                    
                    # 保存分析结果到会话状态
                    st.session_state["analysis_result"] = analysis_result
                    st.session_state["current_contract_id"] = contract_id
                    
                    st.success("合同分析完成！")
                    st.experimental_rerun()
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"分析失败: {str(e)}")

# 查看历史页面
elif menu_option == "查看历史":
    st.header("合同历史记录")
    
    try:
        # 调用 API 获取合同列表
        response = requests.get(f"{API_BASE_URL}/contracts/")
        response.raise_for_status()
        
        contracts = response.json()
        
        if contracts:
            # 显示合同列表
            for contract in contracts:
                with st.expander(f"合同: {contract['filename']} (ID: {contract['id']})"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("上传时间", contract['uploaded_at'][:19])
                    col2.metric("分析状态", "已分析" if contract['analyzed'] else "未分析")
                    if contract['risk_score']:
                        col3.metric("风险评分", f"{contract['risk_score']:.2f}")
                    
                    # 查看详情按钮
                    if st.button(f"查看详情", key=f"detail_{contract['id']}"):
                        st.session_state["current_contract_id"] = contract['id']
                        st.session_state["menu_option"] = "分析结果"
                        st.experimental_rerun()
        else:
            st.info("暂无合同记录")
            
    except requests.exceptions.RequestException as e:
        st.error(f"获取历史记录失败: {str(e)}")

# 分析结果页面
elif menu_option == "分析结果":
    st.header("合同分析结果")
    
    # 从会话状态获取分析结果
    analysis_result = st.session_state.get("analysis_result")
    current_contract_id = st.session_state.get("current_contract_id")
    
    if not analysis_result and current_contract_id:
        # 如果没有缓存结果，尝试从 API 获取
        try:
            response = requests.get(f"{API_BASE_URL}/contracts/{current_contract_id}")
            response.raise_for_status()
            
            contract_data = response.json()
            if contract_data["analyzed"]:
                # 调用分析 API 获取结果
                analyze_response = requests.post(f"{API_BASE_URL}/contracts/{current_contract_id}/analyze")
                analyze_response.raise_for_status()
                analysis_result = analyze_response.json()
                st.session_state["analysis_result"] = analysis_result
        except requests.exceptions.RequestException as e:
            st.error(f"获取分析结果失败: {str(e)}")
    
    if analysis_result:
        # 显示分析结果
        result = analysis_result["analysis_result"]
        
        # 风险评分
        st.subheader("风险评估")
        col1, col2 = st.columns(2)
        
        # 风险等级
        risk_level = "低风险" if result["risk_score"] < 0.5 else "中风险" if result["risk_score"] < 0.8 else "高风险"
        col1.metric("风险评分", f"{result['risk_score']:.2f}")
        col2.metric("风险等级", risk_level)
        
        # 风险进度条
        st.progress(result["risk_score"])
        
        # 分析摘要
        st.subheader("分析摘要")
        st.write(result["summary"])
        
        # 问题条款
        st.subheader("问题条款")
        if result["problematic_clauses"]:
            for i, clause in enumerate(result["problematic_clauses"]):
                with st.expander(f"条款 {i+1}: {clause['clause_type']} (风险: {clause['risk_level']})"):
                    st.write("**条款内容:**")
                    st.write(clause["clause_text"])
                    
                    col1, col2 = st.columns(2)
                    col1.metric("风险分数", f"{clause['risk_score']:.2f}")
                    col2.metric("条款类型", clause["clause_type"])
                    
                    st.write("**建议:**")
                    st.write(clause["recommendation"])
        else:
            st.success("未发现明显问题条款")
    else:
        st.info("请先上传并分析合同")

# 页脚
st.markdown("---")
st.markdown("© 2024 合同分析系统 | 基于深度学习的合同风险评估")
