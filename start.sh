#!/bin/bash

echo "========================================"
echo "合同分析系统启动脚本"
echo "========================================"

# 检查是否安装了Python
if ! command -v python3 &> /dev/null
then
    echo "错误: Python 3 未安装"
    exit 1
fi

# 检查是否安装了pip
if ! command -v pip3 &> /dev/null
then
    echo "错误: pip 未安装"
    exit 1
fi

echo "1. 安装依赖..."
pip3 install -r requirements.txt

# 检查.env文件是否存在
if [ ! -f .env ]; then
    echo "2. 创建.env文件..."
    cp .env.example .env
    echo "   .env文件已创建，请根据需要修改配置"
fi

echo "3. 启动后端服务..."
cd backend
python3 main.py &
BACKEND_PID=$!
cd ..

echo "4. 等待后端服务启动..."
sleep 5

echo "5. 启动模型使用客户端..."
cd clients/model-client
streamlit run app.py &
MODEL_CLIENT_PID=$!
cd ..

echo "6. 启动训练状况查看客户端..."
cd training-client
streamlit run app.py &
TRAINING_CLIENT_PID=$!
cd ../..

echo "========================================"
echo "系统已启动！"
echo "========================================"
echo "后端服务: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo "模型使用客户端: http://localhost:8501"
echo "训练状况查看客户端: http://localhost:8502"
echo "========================================"
echo "按 Ctrl+C 停止所有服务"
echo "========================================"

# 等待用户中断
wait $BACKEND_PID $MODEL_CLIENT_PID $TRAINING_CLIENT_PID

# 清理所有进程
kill $BACKEND_PID $MODEL_CLIENT_PID $TRAINING_CLIENT_PID 2>/dev/null

echo "系统已停止"
