# 合同分析系统 (Contract Analyzer)

一个基于深度学习的合同分析系统，用于识别有问题的合同条款，并提供云端爬虫和可视化客户端。支持GitHub自动部署和模型自动更新。

## 项目结构

```
contract-analyzer/
├── backend/                 # 后端服务
│   ├── app/                # FastAPI应用
│   │   ├── api/            # API路由
│   │   ├── models/         # 数据库模型
│   │   ├── services/       # 业务逻辑
│   │   └── schemas/        # 数据模型
│   └── main.py             # 后端入口
├── model/                  # 模型相关
│   ├── data/               # 训练数据
│   ├── logs/               # 训练日志
│   ├── notebooks/          # Jupyter笔记本
│   ├── scripts/            # 训练脚本
│   └── saved_models/       # 保存的模型
├── crawler/                # 爬虫
│   ├── contract_spider/     # Scrapy爬虫项目
│   └── scrapy.cfg          # Scrapy配置
├── clients/                # 客户端
│   ├── model-client/       # 模型使用客户端
│   └── training-client/    # 训练状况查看客户端
├── .github/                # GitHub Actions配置
│   └── workflows/          # 工作流配置
├── .env.example            # 环境变量示例
├── requirements.txt        # 依赖
├── start.sh                # 启动脚本
└── README.md               # 项目说明
```

## 功能特点

1. **高级合同分析模型**：使用中文法律领域预训练模型（RoBERTa）识别有问题的合同条款
2. **多级风险评估**：支持低风险、中风险、高风险三级风险分类
3. **云端爬虫**：自动从多个来源抓取合同样本
4. **双客户端**：
   - 模型使用客户端：用于上传和分析合同
   - 训练状况查看客户端：监控模型训练进度和结果
5. **RESTful API**：提供标准化的API接口
6. **模型自动更新**：支持模型版本管理和自动部署
7. **GitHub自动部署**：通过GitHub Actions实现自动训练和部署
8. **详细的训练报告**：生成训练历史、混淆矩阵等可视化报告

## 技术栈

- **后端**：FastAPI + asyncio
- **模型**：PyTorch, Transformers (RoBERTa)
- **爬虫**：Scrapy
- **客户端**：Streamlit
- **数据库**：SQLAlchemy + PostgreSQL/SQLite
- **CI/CD**：GitHub Actions

## 安装和运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 根据需要修改.env文件中的配置
```

### 3. 运行系统

#### 使用启动脚本一键启动

```bash
chmod +x start.sh
./start.sh
```

#### 手动启动

##### 启动后端服务
```bash
cd backend
python main.py
```

##### 启动模型使用客户端
```bash
cd clients/model-client
streamlit run app.py
```

##### 启动训练状况查看客户端
```bash
cd clients/training-client
streamlit run app.py
```

##### 运行爬虫
```bash
cd crawler
scrapy crawl generic_contract
```

## API文档

启动后端服务后，访问 `http://localhost:8000/docs` 查看自动生成的API文档。

## 模型训练

### 基础训练

```bash
cd model/scripts
python train.py
```

### 高级训练（推荐）

```bash
cd model/scripts
python advanced_train.py
```

高级训练特点：
- 使用中文法律领域预训练模型
- 支持三级风险分类
- 自动保存最佳模型
- 早停机制
- 生成详细的训练报告和混淆矩阵

## GitHub自动部署

### 功能

- 自动训练：每天凌晨2点自动训练模型
- 代码推送触发：修改模型相关代码时自动训练
- 手动触发：支持手动启动训练和部署
- 自动测试：训练完成后自动测试模型
- 自动部署：自动部署最佳模型
- 结果保存：保存训练结果和日志

### 配置

1. Fork本项目到你的GitHub账号
2. 在项目Settings中配置必要的环境变量
3. 启用GitHub Actions
4. 推送代码或手动触发工作流

### 工作流触发条件

- 定时触发：每天凌晨2点
- 代码推送：修改model/scripts/、model/data/或工作流文件时
- 手动触发：通过GitHub界面手动启动

## 模型部署

### 手动部署

```bash
cd backend
python -c "from app.services.model_deployment_service import ModelDeploymentService; 
model_dir = '../model/saved_models'; 
deployed_model_dir = '../model/saved_models/deployed'; 
service = ModelDeploymentService(model_dir, deployed_model_dir); 
service.auto_deploy_best_model('contract_classifier')"
```

### 自动部署

GitHub Actions会自动部署最佳模型，无需手动操作。

## 客户端使用

### 模型使用客户端

1. 上传合同文件（支持txt、pdf、doc、docx）
2. 点击"开始分析"
3. 查看风险评估和问题条款
4. 浏览历史分析记录

### 训练状况查看客户端

1. 查看当前训练状态
2. 浏览训练历史记录
3. 启动新的训练任务
4. 管理爬虫

## 扩展建议

1. 集成更复杂的NLP模型，提高合同分析准确率
2. 添加更多爬虫，支持从不同来源获取合同样本
3. 实现更详细的风险评估和条款分类
4. 添加用户认证和权限管理
5. 实现模型自动更新和部署功能

## 许可证

MIT
