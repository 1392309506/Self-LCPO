import yaml
from pathlib import Path
from typing import Dict, Any, List


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path).resolve()
        self._validate_path()
        self.config = self._load_config()
        self._validate_config()

    def _validate_path(self):
        """验证配置文件路径有效性"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        if self.config_path.suffix.lower() not in ['.yaml', '.yml']:
            raise ValueError("仅支持YAML格式配置文件")

    def _load_config(self) -> Dict[str, Any]:
        """加载并解析YAML配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:  # 显式指定编码为utf-8
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"配置文件解析失败: {str(e)}")

    def _validate_config(self):
        """执行配置验证"""
        if self.config is None:
            raise ValueError("配置文件解析失败: 配置内容为空")

        # 验证必要配置段
        required_sections = ['experiment', 'datasets', 'models']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置缺少必要章节: {section}")

        # 验证实验参数
        self._validate_experiment_section()

        # 验证数据集路径
        self._validate_datasets()

        # 验证模型配置
        self._validate_models()

    def _validate_experiment_section(self):
        """验证实验参数配置"""
        exp = self.config['experiment']
        if not isinstance(exp.get('n_i_values'), list):
            raise ValueError("experiment.n_i_values 必须为列表")
        if not isinstance(exp.get('max_questions'), int):
            raise ValueError("experiment.max_questions 必须为整数")

    def _validate_datasets(self):
        """验证数据集配置"""
        for name, path in self.config['datasets'].items():
            dataset_path = self.config_path.parent.parent / 'dataset' / Path(path).name
            if not dataset_path.exists():
                raise FileNotFoundError(f"数据集 {name} 路径不存在: {dataset_path}")

    def _validate_models(self):
        """验证模型配置"""
        required_model_fields = ['name', 'api-type', 'api_keys', 'params']
        valid_api_types = ['openai']

        for model in self.config['models']:
            # 检查必要字段
            missing_fields = [f for f in required_model_fields if f not in model]
            if missing_fields:
                raise ValueError(f"模型 {model.get('name', '未命名')} 缺少必要字段: {missing_fields}")

            # 验证API类型
            if model['api-type'].lower() not in valid_api_types:
                raise ValueError(f"模型 {model['name']} 的api-type不合法，当前支持: {valid_api_types}")

            # 验证参数结构
            if not isinstance(model['params'], dict):
                raise ValueError(f"模型 {model['name']} 的params必须为字典类型")

    @property
    def experiment(self) -> Dict[str, Any]:
        """获取实验参数"""
        return {
            'n_i_values': self.config['experiment']['n_i_values'],
            'max_questions': self.config['experiment']['max_questions']
        }

    @property
    def datasets(self) -> Dict[str, str]:
        """获取数据集路径映射"""
        return {
            name: str(self.config_path.parent.parent / 'dataset' / Path(path).name)
            for name, path in self.config['datasets'].items()
        }

    @property
    def models(self) -> Dict[str, Dict]:
        """将模型配置转换为以模型名称为键的字典"""
        return {
            model['name']: {
                'api_type': model['api-type'],  # 转换为下划线命名
                'base_url': model.get('base_url'),
                'api_key': model['api_keys'],
                'params': model['params']
            }
            for model in self.config['models']
        }
