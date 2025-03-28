import logging

class LoggerUtil:
    """工具类，提供日志封装方法"""
    @staticmethod
    def get_logger(name: str = "app"):
        """获取一个标准化的 logger"""
        logger = logging.getLogger(name)
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            )
        return logger
