import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
class Config:
    def __init__(self):
        self.model_name = "bert-base-chinese"
        self.max_length = 512
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 2e-5
        self.num_classes = 2  # 0: 正常条款, 1: 问题条款
        self.data_path = "../data/contract_clauses.csv"
        self.model_save_path = "../saved_models/contract_classifier.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 合同条款数据集
class ContractClauseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# 合同分类模型
class ContractClassifier(nn.Module):
    def __init__(self, config):
        super(ContractClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# 训练模型
def train_model(config):
    logger.info("Loading data...")
    
    # 加载数据
    try:
        df = pd.read_csv(config.data_path)
    except FileNotFoundError:
        logger.error(f"Data file not found at {config.data_path}")
        logger.info("Please create a sample dataset with 'text' and 'label' columns.")
        # 创建示例数据
        sample_data = {
            "text": [
                "本合同自双方签字盖章之日起生效，有效期为一年。",
                "任何一方违反本合同的任何条款，应向对方支付违约金。",
                "因不可抗力导致合同无法履行的，双方互不承担责任。",
                "本合同的解释权归甲方所有。",
                "乙方必须在合同签订后3日内支付全部款项。",
                "甲方有权随时修改合同条款，无需通知乙方。",
                "乙方不得向任何第三方透露本合同的任何内容。",
                "本合同的争议由甲方所在地法院管辖。"
            ],
            "label": [0, 1, 0, 1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(config.data_path, index=False)
        logger.info(f"Created sample data at {config.data_path}")
    
    # 准备数据
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    
    # 划分训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # 加载分词器
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    
    # 创建数据集和数据加载器
    train_dataset = ContractClauseDataset(train_texts, train_labels, tokenizer, config.max_length)
    test_dataset = ContractClauseDataset(test_texts, test_labels, tokenizer, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 初始化模型
    logger.info("Initializing model...")
    model = ContractClassifier(config).to(config.device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 训练模型
    logger.info("Starting training...")
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["label"].to(config.device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        
        # 评估模型
        model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                labels = batch["label"].to(config.device)
                
                outputs = model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        # 计算评估指标
        accuracy = accuracy_score(actual_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predictions, average="binary")
        
        logger.info(f"Epoch {epoch+1}/{config.epochs}:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
    
    # 保存模型
    logger.info(f"Saving model to {config.model_save_path}")
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), config.model_save_path)
    logger.info("Training completed!")

# 主函数
if __name__ == "__main__":
    config = Config()
    train_model(config)
