import random
from pathlib import Path
import yaml


class LoadUtils:
    def __init__(self, file_name: str):
        """
        初始化 LoadUtils
        :param file_name: YAML 配置文件名
        """
        self.file_name = file_name
        self.config_path = Path(__file__).parent.parent / "settings" / self.file_name

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file '{self.file_name}' not found in settings directory")

    def _load_yaml(self) -> dict:
        """
        加载 YAML 文件内容
        :return: 解析后的 YAML 数据
        """
        try:
            with self.config_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file '{self.file_name}': {str(e)}")
        except Exception as e:
            raise Exception(f"Error reading file '{self.file_name}': {str(e)}")

    def load_meta_data(self, sample_k: int = 3) -> tuple:
        """
        加载配置数据，包括 prompt、requirements、随机抽样的 QA 对以及 count 信息
        :return: (prompt, requirements, random_qa, count)
        """
        data = self._load_yaml()

        if "qa" not in data or not isinstance(data["qa"], list):
            raise ValueError("Invalid YAML format: Missing 'qa' section or 'qa' is not a list.")

        qa = [{"question": item["question"], "answer": item["answer"]} for item in data["qa"]]

        prompt = data.get("prompt", "")
        requirements = data.get("requirements", "")
        count = data.get("count", "")

        # 处理 count 的格式
        count_str = f", within {count} words" if isinstance(count, int) else ""

        # 随机抽样
        # If k is None or 0, return all QA pairs
        if sample_k is None or sample_k == 0:
            return prompt, requirements, qa, count

        # Otherwise, sample k QA pairs
        random_qa = random.sample(qa, min(sample_k, len(qa)))

        return prompt, requirements, random_qa, count_str