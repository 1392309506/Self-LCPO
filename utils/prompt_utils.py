from pathlib import Path
from utils.logger_utils import LoggerUtil
logger=LoggerUtil.get_logger("PromptUtils")
class PromptUtils:
    def __init__(self, root_path: Path):
        self.root_path = root_path

    def create_round_directory(self, prompt_path: Path, round_number: int) -> Path:
        directory = prompt_path / f"round_{round_number}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def load_prompt(self, round_number: int, prompts_path: Path):
        prompt_file = prompts_path / "prompt.txt"

        try:
            return prompt_file.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            logger.info(f"Error loading prompt for round {round_number}: {e}")
            raise

    def write_answers(self, directory: Path, answers: dict, name: str = "answers.txt"):
        answers_file = directory / name
        with answers_file.open("w", encoding="utf-8") as file:
            for item in answers:
                file.write(f"Question:\n{item['question']}\n")
                file.write(f"Answer:\n{item['answer']}\n")
                file.write("\n")

    def write_prompt(self, directory: Path, prompt: str):
        prompt_file = directory / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

    def get_final_prompt(self) -> str:
        """直接从结果目录加载最终 prompt"""
        prompt_file = self.root_path / "best_prompt.txt"
        if not prompt_file.exists():
            logger.error(f"找不到 best_prompt.txt: {prompt_file}")
            return ""
        return prompt_file.read_text(encoding="utf-8").strip()

