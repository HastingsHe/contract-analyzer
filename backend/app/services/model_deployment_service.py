import os
import json
import shutil
import requests
from datetime import datetime
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ModelDeploymentService:
    """模型部署服务"""
    
    def __init__(self, model_dir: str, deployed_model_dir: str):
        self.model_dir = model_dir
        self.deployed_model_dir = deployed_model_dir
        self.model_registry_path = os.path.join(model_dir, "model_registry.json")
        self._ensure_directories()
        self._load_model_registry()
    
    def _ensure_directories(self):
        """确保目录存在"""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.deployed_model_dir, exist_ok=True)
    
    def _load_model_registry(self):
        """加载模型注册表"""
        if os.path.exists(self.model_registry_path):
            try:
                with open(self.model_registry_path, 'r', encoding='utf-8') as f:
                    self.model_registry = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
                self.model_registry = {"models": []}
        else:
            self.model_registry = {"models": []}
    
    def _save_model_registry(self):
        """保存模型注册表"""
        try:
            with open(self.model_registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_registry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def register_model(self, model_name: str, model_path: str, metrics: Dict[str, float]) -> Dict[str, any]:
        """注册新模型"""
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 复制模型到模型目录
        dest_path = os.path.join(self.model_dir, model_id)
        try:
            if os.path.isdir(model_path):
                shutil.copytree(model_path, dest_path)
            else:
                shutil.copy(model_path, dest_path)
        except Exception as e:
            logger.error(f"Failed to copy model: {e}")
            raise
        
        # 注册模型
        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "path": dest_path,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics,
            "deployed": False
        }
        
        self.model_registry["models"].append(model_info)
        self._save_model_registry()
        
        logger.info(f"Model registered: {model_id}")
        return model_info
    
    def deploy_model(self, model_id: str) -> bool:
        """部署模型"""
        # 查找模型
        model_info = None
        for model in self.model_registry["models"]:
            if model["model_id"] == model_id:
                model_info = model
                break
        
        if not model_info:
            logger.error(f"Model not found: {model_id}")
            return False
        
        try:
            # 标记当前部署的模型为未部署
            for model in self.model_registry["models"]:
                model["deployed"] = False
            
            # 部署新模型
            # 清空部署目录
            for item in os.listdir(self.deployed_model_dir):
                item_path = os.path.join(self.deployed_model_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            
            # 复制模型到部署目录
            if os.path.isdir(model_info["path"]):
                for item in os.listdir(model_info["path"]):
                    src = os.path.join(model_info["path"], item)
                    dst = os.path.join(self.deployed_model_dir, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy(src, dst)
            else:
                shutil.copy(model_info["path"], self.deployed_model_dir)
            
            # 更新模型状态
            model_info["deployed"] = True
            model_info["deployed_at"] = datetime.now().isoformat()
            self._save_model_registry()
            
            logger.info(f"Model deployed: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, any]]:
        """获取模型信息"""
        for model in self.model_registry["models"]:
            if model["model_id"] == model_id:
                return model
        return None
    
    def get_all_models(self) -> List[Dict[str, any]]:
        """获取所有模型"""
        return self.model_registry["models"]
    
    def get_current_deployed_model(self) -> Optional[Dict[str, any]]:
        """获取当前部署的模型"""
        for model in self.model_registry["models"]:
            if model.get("deployed", False):
                return model
        return None
    
    def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        model_info = None
        index_to_remove = -1
        
        for i, model in enumerate(self.model_registry["models"]):
            if model["model_id"] == model_id:
                model_info = model
                index_to_remove = i
                break
        
        if not model_info:
            logger.error(f"Model not found: {model_id}")
            return False
        
        # 检查是否是当前部署的模型
        if model_info.get("deployed", False):
            logger.error(f"Cannot delete deployed model: {model_id}")
            return False
        
        try:
            # 删除模型文件
            if os.path.exists(model_info["path"]):
                if os.path.isdir(model_info["path"]):
                    shutil.rmtree(model_info["path"])
                else:
                    os.remove(model_info["path"])
            
            # 从注册表中移除
            self.model_registry["models"].pop(index_to_remove)
            self._save_model_registry()
            
            logger.info(f"Model deleted: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]) -> bool:
        """更新模型指标"""
        for model in self.model_registry["models"]:
            if model["model_id"] == model_id:
                model["metrics"].update(metrics)
                self._save_model_registry()
                logger.info(f"Model metrics updated: {model_id}")
                return True
        logger.error(f"Model not found: {model_id}")
        return False
    
    def auto_deploy_best_model(self, model_name: str) -> Optional[str]:
        """自动部署最佳模型"""
        # 筛选指定名称的模型
        models = [model for model in self.model_registry["models"] if model["model_name"] == model_name]
        
        if not models:
            logger.error(f"No models found with name: {model_name}")
            return None
        
        # 按F1分数排序（如果有）
        models.sort(key=lambda x: x["metrics"].get("f1", 0), reverse=True)
        
        # 部署最佳模型
        best_model = models[0]
        if self.deploy_model(best_model["model_id"]):
            logger.info(f"Auto deployed best model: {best_model['model_id']}")
            return best_model["model_id"]
        
        return None
