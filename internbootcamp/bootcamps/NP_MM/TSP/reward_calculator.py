import ast
from verl.Bootcampv2.base_reward_calculator import BaseRewardCalculator

class NpTspRewardCalculator(BaseRewardCalculator):
    @classmethod
    def extract_output(cls, response: str) -> dict:
        if "Answer:" not in response:
            return {"format": False, "answer": False, "str": "invalid answer: no 'Answer:' in answer"}
        
        path_str = response.split("Answer:")[-1].strip()

        try:
            path = ast.literal_eval(path_str)
            if not isinstance(path, list) or not all(isinstance(i, int) for i in path):
                return {"format": True, "answer": False, "str": "Answer must be a list of integers."}
            return {"format": True, "answer": True, "str": str(path)}
        except (ValueError, SyntaxError):
            return {"format": True, "answer": False, "str": "Invalid list format in answer."}

    @classmethod
    def _calculate_score(cls, extracted_output: dict, identity: dict) -> float:
        format_reward = 1.0 if extracted_output.get('format', False) else -1.0
        
        if not extracted_output.get('answer', False):
            answer_reward = -1.5
            return format_reward + answer_reward
        
        graph = identity["question"]
        num_cities = len(graph)

        try:
            path = ast.literal_eval(extracted_output["str"])
        except (ValueError, SyntaxError):
            return format_reward - 1.5

        # --- VALIDATION ---
        if not path or path[0] != path[-1]:
            return format_reward - 1.5  # Not a cycle
        
        if len(path) != num_cities + 1 or len(set(path[:-1])) != num_cities:
            return format_reward - 1.5 # Doesn't visit all cities exactly once

        total_distance = 0
        for i in range(len(path) - 1):
            u, v = str(path[i]), str(path[i+1])
            try:
                total_distance += graph[u][v]
            except KeyError:
                return format_reward - 1.5 # Invalid edge

        # --- SCORING ---
        ground_truth = identity.get("ground_truth")
        if ground_truth is not None and ground_truth > 0:
            # For TSP, lower is better, so the ratio is inverted.
            answer_reward = ground_truth / total_distance if total_distance > 0 else 1.0
        elif ground_truth == 0:
            answer_reward = 1.0 if total_distance == 0 else 0.0
        else:
            answer_reward = 0.0

        return format_reward + answer_reward

    @classmethod
    def _verify_correction(cls, extracted_output, identity: dict) -> float:
        return cls._calculate_score(extracted_output, identity)

if __name__ == "__main__":
    identity = {
        "question": {"0": {"0": 0, "1": 56, "2": 60, "3": 11, "4": 15, "5": 95, "6": 32, "7": 84, "8": 100}, "1": {"0": 56, "1": 0, "2": 95, "3": 34, "4": 41, "5": 68, "6": 15, "7": 88, "8": 28}, "2": {"0": 60, "1": 95, "2": 0, "3": 69, "4": 34, "5": 57, "6": 50, "7": 31, "8": 46}, "3": {"0": 11, "1": 34, "2": 69, "3": 0, "4": 36, "5": 51, "6": 96, "7": 16, "8": 54}, "4": {"0": 15, "1": 41, "2": 34, "3": 36, "4": 0, "5": 14, "6": 27, "7": 76, "8": 99}, "5": {"0": 95, "1": 68, "2": 57, "3": 51, "4": 14, "5": 0, "6": 85, "7": 30, "8": 20}, "6": {"0": 32, "1": 15, "2": 50, "3": 96, "4": 27, "5": 85, "6": 0, "7": 12, "8": 42}, "7": {"0": 84, "1": 88, "2": 31, "3": 16, "4": 76, "5": 30, "6": 12, "7": 0, "8": 60}, "8": {"0": 100, "1": 28, "2": 46, "3": 54, "4": 99, "5": 20, "6": 42, "7": 60, "8": 0}},
        "ground_truth": 200.0
    }
    extracted_output = {"format": True, "answer": True, "str": "[0, 8, 2, 1, 5, 3, 4, 6, 7, 0]"}
    print(NpTspRewardCalculator._calculate_score(extracted_output, identity))