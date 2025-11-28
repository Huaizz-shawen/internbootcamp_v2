from typing import Any, Dict, Optional, Tuple, Union, List, Callable

from internbootcamp.src.base_reward_calculator import BaseRewardCalculator
from internbootcamp.src.base_evaluator import BaseEvaluator
from internbootcamp.bootcamps.gsm8k.gsm8k_reward_manager import GSM8KRewardManager

class Gsm8kEvaluator(base_evaluator):
    def __init__(self, api_url: str, api_key: str, api_model: str, reward_manager=GSM8KRewardManager, max_turns: int = 5, **kwargs):
        super().__init__(api_url=api_url, api_key=api_key, api_model=api_model, reward_manager=reward_manager, max_turns=max_turns, **kwargs)
    
    def run_evaluation(self, dataset: Optional[List[dict]] = None, dataset_path: Optional[str] = None, output_path: Optional[str] = None, yaml_tool_path: Optional[str] = 'internbootcamp/bootcamps/gsm8k/config/gsm8k_tool_config.yaml', max_concurrent: int = 1):
        super().run_evaluation(dataset=dataset, dataset_path=dataset_path, output_path=output_path, yaml_tool_path=yaml_tool_path, max_concurrent=max_concurrent)
