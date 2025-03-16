# 在其他文件中复用
from config_loader import ConfigLoader

def test_config_loading():
    config = ConfigLoader("../config/config_llm.yaml")
    print(config.models)
    assert len(config.models) > 0

def analyze_results(config_path: str):
    config = ConfigLoader(config_path)
    print(f"当前配置包含 {len(config.models)} 个模型")


if __name__ == "__main__":
    test_config_loading()