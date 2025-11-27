import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 设置页面标题和布局
st.set_page_config(
    page_title="模型训练监控",
    page_icon="📊",
    layout="wide"
)

# API 基础 URL
API_BASE_URL = "http://localhost:8000/api"

# 页面标题
st.title("📊 模型训练监控系统")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("功能菜单")
    menu_option = st.radio(
        "选择功能",
        ["训练状态", "训练历史", "启动训练", "爬虫管理"]
    )

# 训练状态页面
if menu_option == "训练状态":
    st.header("当前训练状态")
    
    # 自动刷新按钮
    auto_refresh = st.checkbox("自动刷新", value=True)
    
    # 获取训练状态
    def get_training_status():
        try:
            response = requests.get(f"{API_BASE_URL}/training/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"获取训练状态失败: {str(e)}")
            return None
    
    # 获取最新训练日志
    def get_latest_training_log():
        try:
            response = requests.get(f"{API_BASE_URL}/training/latest")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None
    
    # 显示训练状态
    status = get_training_status()
    latest_log = get_latest_training_log()
    
    if status:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("训练状态", "正在训练" if status["is_running"] else "未训练")
        col2.metric("当前轮次", status["current_epoch"] if status["current_epoch"] else "N/A")
        col3.metric("当前损失", f"{status['current_loss']:.4f}" if status['current_loss'] is not None else "N/A")
        col4.metric("当前准确率", f"{status['current_accuracy']:.4f}" if status['current_accuracy'] is not None else "N/A")
        
        # 显示训练进度条
        if status["is_running"] and latest_log and latest_log.get("epochs"):
            progress = status["current_epoch"] / latest_log["epochs"]
            st.progress(progress)
            st.caption(f"进度: {status['current_epoch']}/{latest_log['epochs']} 轮次")
        
        # 显示训练日志
        if latest_log:
            st.subheader("训练日志")
            with st.expander("查看详细日志"):
                st.text(latest_log.get("log_message", "暂无日志"))
    
    # 自动刷新
    if auto_refresh and status and status["is_running"]:
        st.experimental_rerun()

# 训练历史页面
elif menu_option == "训练历史":
    st.header("训练历史记录")
    
    try:
        # 获取训练日志列表
        response = requests.get(f"{API_BASE_URL}/training/logs")
        response.raise_for_status()
        logs = response.json()
        
        if logs:
            # 显示训练日志列表
            for log in logs:
                with st.expander(f"模型: {log['model_name']} (状态: {log['status']})"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("开始时间", log['start_time'][:19])
                    if log['end_time']:
                        col2.metric("结束时间", log['end_time'][:19])
                    else:
                        col2.metric("结束时间", "进行中")
                    col3.metric("最终准确率", f"{log['accuracy']:.4f}" if log['accuracy'] is not None else "N/A")
                    
                    # 显示详细指标
                    if log['status'] == "completed":
                        st.subheader("训练指标")
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        metrics_col1.metric("损失值", f"{log['loss']:.4f}")
                        metrics_col2.metric("精确率", f"{log['precision']:.4f}")
                        metrics_col3.metric("召回率", f"{log['recall']:.4f}")
                        st.metric("F1分数", f"{log['f1_score']:.4f}")
                    
                    # 显示日志消息
                    if log['log_message']:
                        st.text_area("日志消息", log['log_message'], height=100)
        else:
            st.info("暂无训练记录")
            
    except requests.exceptions.RequestException as e:
        st.error(f"获取训练历史失败: {str(e)}")

# 启动训练页面
elif menu_option == "启动训练":
    st.header("启动模型训练")
    
    # 训练参数设置
    with st.form("training_form"):
        st.subheader("训练参数")
        
        model_name = st.text_input("模型名称", value="contract_classifier")
        epochs = st.slider("训练轮次", min_value=1, max_value=100, value=10)
        learning_rate = st.text_input("学习率", value="0.001")
        batch_size = st.slider("批次大小", min_value=1, max_value=256, value=32)
        
        # 提交按钮
        submitted = st.form_submit_button("开始训练", type="primary")
    
    if submitted:
        try:
            # 准备训练参数
            train_data = {
                "model_name": model_name,
                "epochs": epochs,
                "learning_rate": float(learning_rate),
                "batch_size": batch_size
            }
            
            # 调用 API 启动训练
            with st.spinner("正在启动训练..."):
                response = requests.post(f"{API_BASE_URL}/training/start", json=train_data)
                response.raise_for_status()
                
                result = response.json()
                st.success(f"训练启动成功！训练ID: {result['training_id']}")
                
                # 跳转到训练状态页面
                st.session_state["menu_option"] = "训练状态"
                st.experimental_rerun()
                
        except ValueError:
            st.error("请输入有效的学习率")
        except requests.exceptions.RequestException as e:
            st.error(f"启动训练失败: {str(e)}")

# 爬虫管理页面
elif menu_option == "爬虫管理":
    st.header("爬虫管理")
    
    # 获取爬虫状态
    def get_crawler_status():
        try:
            response = requests.get(f"{API_BASE_URL}/crawler/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"获取爬虫状态失败: {str(e)}")
            return None
    
    # 显示爬虫状态
    status = get_crawler_status()
    
    if status:
        col1, col2 = st.columns(2)
        col1.metric("爬虫状态", "运行中" if status["running"] else "已停止")
        col2.metric("当前爬虫", status["current_spider"] if status["current_spider"] else "无")
    
    # 爬虫控制按钮
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("启动爬虫", type="primary"):
            try:
                response = requests.post(f"{API_BASE_URL}/crawler/start")
                response.raise_for_status()
                result = response.json()
                st.success(f"爬虫启动成功！爬虫ID: {result['crawler_id']}")
            except requests.exceptions.RequestException as e:
                st.error(f"启动爬虫失败: {str(e)}")
    
    with col2:
        if st.button("停止爬虫"):
            try:
                response = requests.post(f"{API_BASE_URL}/crawler/stop")
                response.raise_for_status()
                st.success("爬虫停止成功！")
            except requests.exceptions.RequestException as e:
                st.error(f"停止爬虫失败: {str(e)}")
    
    # 爬虫日志
    st.subheader("爬虫日志")
    
    try:
        # 获取爬虫日志列表
        response = requests.get(f"{API_BASE_URL}/crawler/logs")
        response.raise_for_status()
        logs = response.json()
        
        if logs:
            # 显示爬虫日志列表
            for log in logs:
                with st.expander(f"爬虫: {log['spider_name']} (状态: {log['status']})"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("开始时间", log['start_time'][:19])
                    if log['end_time']:
                        col2.metric("结束时间", log['end_time'][:19])
                    else:
                        col2.metric("结束时间", "进行中")
                    col3.metric("爬取合同数", log['contracts_crawled'])
                    
                    # 显示日志消息
                    if log['log_message']:
                        st.text_area("日志消息", log['log_message'], height=100)
        else:
            st.info("暂无爬虫记录")
            
    except requests.exceptions.RequestException as e:
        st.error(f"获取爬虫日志失败: {str(e)}")

# 页脚
st.markdown("---")
st.markdown("© 2024 模型训练监控系统 | 实时监控模型训练进度")
