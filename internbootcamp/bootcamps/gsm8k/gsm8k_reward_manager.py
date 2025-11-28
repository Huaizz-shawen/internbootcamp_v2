from internbootcamp.src.base_reward_calculator import BaseRewardCalculator
import re

class GSM8KRewardManager(BaseRewardCalculator):

    @staticmethod
    def extract_output(output_str: str):
        """
        从模型输出中提取最终答案（假设为最后出现的数字）
        """
        if not output_str:
            return None
        # 匹配最后一个整数或小数
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", output_str)
        if matches:
            return matches[-1]
        return None

    @classmethod
    def _verify_correction(cls, extracted_output, identity: dict) -> float:
        """
        判断答案是否正确
        identity: {"answer": "42"}
        """
        if not extracted_output:
            return 0
        gold = str(identity).strip()
        pred = str(extracted_output).strip()
        # print("gold: ", gold)
        # print("pred: ", pred)
        return 1 if pred == gold else 0
