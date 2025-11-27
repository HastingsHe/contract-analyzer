import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
class Config:
    def __init__(self):
        # 使用中文法律领域预训练模型
        self.model_name = "hfl/chinese-roberta-wwm-ext"
        # 可选：使用法律领域特定模型
        # self.model_name = "law-ai/legal-roberta-chinese"
        self.max_length = 512
        self.batch_size = 16
        self.epochs = 15
        self.learning_rate = 1e-5
        self.warmup_steps = 500
        self.weight_decay = 0.01
        # 多分类：更详细的条款风险等级
        self.num_classes = 3  # 0: 低风险, 1: 中风险, 2: 高风险
        self.data_path = "../data/contract_clauses.csv"
        self.model_save_path = "../saved_models/advanced_contract_classifier"
        self.log_dir = "../logs"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 自动保存最佳模型
        self.save_best_model = True
        # 早停机制
        self.early_stopping_patience = 3

# 合同条款数据集
class AdvancedContractClauseDataset(Dataset):
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

# 高级合同分类模型
class AdvancedContractClassifier:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 加载分词器
        self.logger.info(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # 加载预训练模型
        self.logger.info(f"Loading model: {config.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_classes
        ).to(config.device)
        
        # 定义优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        self.total_steps = 0  # 将在训练时设置
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=self.total_steps,
            pct_start=0.1
        )
        
        # 最佳模型保存
        self.best_val_f1 = 0.0
        self.early_stopping_counter = 0
        
        # 创建保存目录
        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 训练历史记录
        self.train_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }
    
    def load_data(self):
        """加载和预处理数据"""
        self.logger.info("Loading data...")
        
        # 加载数据
        try:
            df = pd.read_csv(self.config.data_path)
        except FileNotFoundError:
            self.logger.error(f"Data file not found at {self.config.data_path}")
            self.logger.info("Creating sample dataset with multi-class labels...")
            # 创建多分类示例数据
            sample_data = {
                "text": [
                    "本合同自双方签字盖章之日起生效，有效期为一年。",  # 低风险
                    "任何一方违反本合同的任何条款，应向对方支付违约金。",  # 中风险
                    "因不可抗力导致合同无法履行的，双方互不承担责任。",  # 低风险
                    "本合同的解释权归甲方所有。",  # 高风险
                    "乙方必须在合同签订后3日内支付全部款项。",  # 中风险
                    "甲方有权随时修改合同条款，无需通知乙方。",  # 高风险
                    "乙方不得向任何第三方透露本合同的任何内容。",  # 中风险
                    "本合同的争议由甲方所在地法院管辖。",  # 高风险
                    "合同期满前30日，任何一方均可提出续签申请。",  # 低风险
                    "如乙方逾期支付款项，应按日支付万分之五的违约金。",  # 中风险
                    "甲方有权单方面解除合同，且不承担任何责任。",  # 高风险
                    "本合同一式两份，甲乙双方各执一份。"  # 低风险
                ],
                "label": [0, 1, 0, 2, 1, 2, 1, 2, 0, 1, 2, 0]  # 0:低风险, 1:中风险, 2:高风险
            }
            df = pd.DataFrame(sample_data)
            df.to_csv(self.config.data_path, index=False)
            self.logger.info(f"Created sample data at {self.config.data_path}")
        
        # 准备数据
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        
        # 划分训练集、验证集和测试集
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )
        
        # 创建数据集
        train_dataset = AdvancedContractClauseDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        val_dataset = AdvancedContractClauseDataset(
            val_texts, val_labels, self.tokenizer, self.config.max_length
        )
        test_dataset = AdvancedContractClauseDataset(
            test_texts, test_labels, self.tokenizer, self.config.max_length
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        # 设置总步数
        self.total_steps = len(self.train_loader) * self.config.epochs
        self.scheduler.total_steps = self.total_steps
        
        self.logger.info(f"Data loaded: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
    
    def train_epoch(self):
        """训练一个轮次"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch["label"].to(self.config.device)
            
            # 前向传播
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录损失和预测
            total_loss += loss.item()
            _, predictions = torch.max(logits, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["label"].to(self.config.device)
                
                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                # 记录损失和预测
                total_loss += loss.item()
                _, predictions = torch.max(logits, dim=1)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        # 计算评估指标
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted"
        )
        
        return avg_loss, accuracy, precision, recall, f1, all_labels, all_predictions
    
    def train(self):
        """训练模型"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.epochs):
            # 训练
            train_loss, train_accuracy = self.train_epoch()
            
            # 在验证集上评估
            val_loss, val_accuracy, val_precision, val_recall, val_f1, _, _ = self.evaluate(self.val_loader)
            
            # 记录历史
            self.train_history["train_loss"].append(train_loss)
            self.train_history["train_accuracy"].append(train_accuracy)
            self.train_history["val_loss"].append(val_loss)
            self.train_history["val_accuracy"].append(val_accuracy)
            self.train_history["val_precision"].append(val_precision)
            self.train_history["val_recall"].append(val_recall)
            self.train_history["val_f1"].append(val_f1)
            
            self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}:")
            self.logger.info(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            self.logger.info(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            self.logger.info(f"  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
            
            # 保存最佳模型
            if self.config.save_best_model and val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_model(f"{self.config.model_save_path}/best_model")
                self.logger.info(f"Best model saved with F1 score: {val_f1:.4f}")
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # 早停机制
            if self.config.early_stopping_patience > 0 and self.early_stopping_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # 保存最终模型
        self.save_model(f"{self.config.model_save_path}/final_model")
        
        # 保存训练历史
        self.save_train_history()
        
        # 生成训练报告
        self.generate_training_report()
        
        # 在测试集上评估
        self.logger.info("Evaluating on test set...")
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_labels, test_predictions = self.evaluate(self.test_loader)
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"  Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")
        
        # 生成混淆矩阵
        self.generate_confusion_matrix(test_labels, test_predictions)
        
        return {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }
    
    def save_model(self, path):
        """保存模型"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def save_train_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config.log_dir, "train_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, ensure_ascii=False, indent=2)
    
    def generate_training_report(self):
        """生成训练报告"""
        report_path = os.path.join(self.config.log_dir, "training_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 模型训练报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 训练参数\n\n")
            f.write(f"- 模型名称: {self.config.model_name}\n")
            f.write(f"- 最大长度: {self.config.max_length}\n")
            f.write(f"- 批次大小: {self.config.batch_size}\n")
            f.write(f"- 训练轮次: {self.config.epochs}\n")
            f.write(f"- 学习率: {self.config.learning_rate}\n")
            f.write(f"- 权重衰减: {self.config.weight_decay}\n\n")
            
            f.write(f"## 训练结果\n\n")
            f.write(f"- 最佳验证F1分数: {self.best_val_f1:.4f}\n")
            f.write(f"- 最终训练损失: {self.train_history['train_loss'][-1]:.4f}\n")
            f.write(f"- 最终训练准确率: {self.train_history['train_accuracy'][-1]:.4f}\n")
            f.write(f"- 最终验证损失: {self.train_history['val_loss'][-1]:.4f}\n")
            f.write(f"- 最终验证准确率: {self.train_history['val_accuracy'][-1]:.4f}\n")
    
    def generate_confusion_matrix(self, y_true, y_pred):
        """生成混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['低风险', '中风险', '高风险'],
                    yticklabels=['低风险', '中风险', '高风险'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig(os.path.join(self.config.log_dir, 'confusion_matrix.png'))

# 主函数
def main():
    config = Config()
    classifier = AdvancedContractClassifier(config)
    classifier.load_data()
    results = classifier.train()
    
    # 保存结果
    results_path = os.path.join(config.log_dir, "test_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
