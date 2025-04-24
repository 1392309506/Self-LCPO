import json
from pathlib import Path
import yaml
from typing import Optional
import re


class LoadUtils:
    def __init__(self, file_name: str):
        """
        初始化 LoadUtils
        :param file_name: YAML 配置文件名
        """
        self.file_name = file_name
        self.config_path = Path(__file__).parent.parent / "dataset" / self.file_name

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
        加载配置数据，包括 prompt、requirements、固定选取前k条 QA 对以及 count 信息
        :return: (prompt, requirements, selected_qa, count_str)
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

        # 固定选取前 sample_k 条 QA
        if sample_k is None or sample_k == 0:
            return prompt, requirements, qa, count_str

        selected_qa = qa[:sample_k]

        return prompt, requirements, selected_qa, count_str

    def load_json(self, sample_k: int = 0) -> list:
        """
        从 JSON 文件中加载问答对。

        参数:
            sample_k (int): 要加载的问答数量。为 0 表示加载全部。

        返回:
            list: 包含问答对的列表，每个元素是一个字典，具有 'question' 和 'answer' 键。
        """
        with open(self.file_name, 'r', encoding='utf-8') as file:
            qa_list = json.load(file)

        if not isinstance(qa_list, list):
            raise ValueError(f"JSON 内容格式错误，应为 list，但实际是 {type(qa_list)}")

        if sample_k == 0:
            return qa_list
        else:
            return qa_list[:min(sample_k, len(qa_list))]

    def extract_content(text: str, tag: str) -> Optional[str]:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
