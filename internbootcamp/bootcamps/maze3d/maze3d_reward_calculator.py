import re
import json
import sys
import os
from typing import Optional, Dict, Any, List, Tuple

from internbootcamp.src.base_reward_calculator import BaseRewardCalculator

# Lazy import - will be loaded when needed
def _get_maze_modules():
    """Lazy import of 3d_maze modules"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../3d_maze'))
    from main import Position, is_path_valid
    return Position, is_path_valid


class Maze3DRewardCalculator(BaseRewardCalculator):
    """
    Three-Level Reward Calculator for 3D Maze RLVR

    Level 1: Outcome Reward - Correctness of final answer
    Level 2: Physics & Logic Reward - RLVR core, verifies reasoning steps follow physical rules
    Level 3: State Tracking Reward - Verifies model's awareness of positions and state
    """

    @staticmethod
    def extract_output(output_str: str) -> Optional[Dict[str, Any]]:
        """
        Extract answer and reasoning from model output

        Args:
            output_str (str): Model's raw output

        Returns:
            Optional[Dict[str, Any]]: Extracted information including answer and reasoning steps
        """
        if not output_str:
            return None

        extracted = {
            "answer": None,
            "reasoning_steps": [],
            "position_claims": [],  # Track position statements
            "height_claims": [],    # Track height comparisons
            "move_actions": []      # Track described movements
        }

        # Extract JSON answer block
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, output_str, re.DOTALL)

        if json_matches:
            try:
                answer_data = json.loads(json_matches[-1])  # Use last JSON block
                extracted["answer"] = answer_data.get("answer")
                extracted["reasoning_steps"] = answer_data.get("reasoning_steps", [])
            except json.JSONDecodeError:
                pass

        # If no JSON, try to find answer in text
        if not extracted["answer"]:
            answer_patterns = [
                r'(?:final answer|answer|solution).*?[:=]\s*["\']?([^"\'\n]+)["\']?',
                r'<answer>(.*?)</answer>',
                r'"answer"\s*:\s*"([^"]+)"'
            ]
            for pattern in answer_patterns:
                match = re.search(pattern, output_str, re.IGNORECASE)
                if match:
                    extracted["answer"] = match.group(1).strip()
                    break

        # Extract position claims (e.g., "I am at position (3, 5, 2)")
        position_pattern = r'(?:position|located|at)\s*(?:is|:|=)?\s*\((\d+),\s*(\d+),\s*(\d+)\)'
        for match in re.finditer(position_pattern, output_str, re.IGNORECASE):
            x, y, z = int(match.group(1)), int(match.group(2)), int(match.group(3))
            extracted["position_claims"].append((x, y, z))

        # Extract height claims (e.g., "Block 1 is at height 3", "Point 2 is higher than Point 1")
        height_pattern = r'(?:block|point|position)\s+(\d+)\s+(?:is\s+)?at\s+height\s+(\d+)'
        for match in re.finditer(height_pattern, output_str, re.IGNORECASE):
            label, height = int(match.group(1)), int(match.group(2))
            extracted["height_claims"].append({"label": label, "height": height})

        comparison_pattern = r'(?:block|point)\s+(\d+)\s+is\s+(higher|lower)\s+than\s+(?:block|point)\s+(\d+)'
        for match in re.finditer(comparison_pattern, output_str, re.IGNORECASE):
            label1, relation, label2 = int(match.group(1)), match.group(2), int(match.group(3))
            extracted["height_claims"].append({
                "label1": label1,
                "relation": relation,
                "label2": label2
            })

        # Extract move actions (e.g., "move forward 2 steps", "climb ladder")
        move_pattern = r'(?:move|go|walk)\s+(forward|backward|left|right)\s*(?:by)?\s*(\d+)?'
        for match in re.finditer(move_pattern, output_str, re.IGNORECASE):
            direction = match.group(1)
            distance = int(match.group(2)) if match.group(2) else 1
            extracted["move_actions"].append({"action": "move", "direction": direction, "distance": distance})

        climb_pattern = r'climb(?:\s+ladder)?(?:\s+(?:by|up))?\s*(\d+)?'
        for match in re.finditer(climb_pattern, output_str, re.IGNORECASE):
            height = int(match.group(1)) if match.group(1) else 1
            extracted["move_actions"].append({"action": "climb", "height": height})

        return extracted

    @classmethod
    def _verify_correction(cls, extracted_output: Dict[str, Any], identity: dict, **kwargs) -> float:
        """
        Verify model output and calculate three-level reward

        Args:
            extracted_output: Extracted information from model output
            identity (dict): Task information including puzzle state and correct answer
            **kwargs: Additional parameters

        Returns:
            float: Total score from all three reward levels
        """
        if not extracted_output or not extracted_output.get("answer"):
            return 0.0

        # Get configuration
        enable_level1 = kwargs.get('enable_level1_outcome', True)
        enable_level2 = kwargs.get('enable_level2_physics', True)
        enable_level3 = kwargs.get('enable_level3_state', True)

        # Weight for each level (should sum to 1.0 for fair comparison)
        weight_level1 = kwargs.get('weight_level1', 0.4)
        weight_level2 = kwargs.get('weight_level2', 0.4)
        weight_level3 = kwargs.get('weight_level3', 0.2)

        total_score = 0.0

        # ===== LEVEL 1: Outcome Reward =====
        if enable_level1:
            level1_score = cls._level1_outcome_reward(
                extracted_output, identity, **kwargs
            )
            total_score += weight_level1 * level1_score
            # print(f"[RLVR] Level 1 (Outcome): {level1_score:.3f}")

        # ===== LEVEL 2: Physics & Logic Reward =====
        if enable_level2:
            level2_score = cls._level2_physics_reward(
                extracted_output, identity, **kwargs
            )
            total_score += weight_level2 * level2_score
            # print(f"[RLVR] Level 2 (Physics): {level2_score:.3f}")

        # ===== LEVEL 3: State Tracking Reward =====
        if enable_level3:
            level3_score = cls._level3_state_tracking_reward(
                extracted_output, identity, **kwargs
            )
            total_score += weight_level3 * level3_score
            # print(f"[RLVR] Level 3 (State): {level3_score:.3f}")

        # print(f"[RLVR] Total Score: {total_score:.3f}")
        return total_score

    @classmethod
    def _level1_outcome_reward(cls, extracted_output: Dict, identity: dict, **kwargs) -> float:
        """
        Level 1: Outcome Reward
        Direct comparison with ground truth answer

        Returns:
            float: 1.0 if correct, 0.0 if incorrect
        """
        predicted_answer = extracted_output.get("answer", "").strip()
        correct_answer = identity.get("answer", "").strip()

        # Normalize answers for comparison
        predicted_normalized = cls._normalize_answer(predicted_answer)
        correct_normalized = cls._normalize_answer(correct_answer)

        if predicted_normalized == correct_normalized:
            return 1.0
        else:
            # Check for partial credit scenarios
            return cls._calculate_partial_outcome_reward(
                predicted_normalized, correct_normalized, identity
            )

    @classmethod
    def _level2_physics_reward(cls, extracted_output: Dict, identity: dict, **kwargs) -> float:
        """
        Level 2: Physics & Logic Reward (RLVR Core)
        Verifies if reasoning steps follow physical rules of the maze

        This is the essence of RLVR - checking if the model's reasoning is grounded in physics

        Returns:
            float: Score based on physics validity of reasoning [0.0, 1.0]
        """
        # Get maze modules
        Position, is_path_valid = _get_maze_modules()

        puzzle_state = identity.get('puzzle_state', {})
        puzzle_type = identity.get('puzzle_type', 'unknown')

        # Parse puzzle state
        cubes = set()
        for cube_pos in puzzle_state.get('cubes', []):
            cubes.add(Position(cube_pos[0], cube_pos[1], cube_pos[2]))

        start_pos_list = puzzle_state.get('start_pos', [0, 0, 0])
        grid_size = tuple(puzzle_state.get('grid_size', [8, 8, 7]))
        start_pos = Position(start_pos_list[0], start_pos_list[1], start_pos_list[2])

        ladders = []
        for ladder_data in puzzle_state.get('ladders', []):
            ladder_base = Position(
                ladder_data['base_pos'][0],
                ladder_data['base_pos'][1],
                ladder_data['base_pos'][2]
            )
            ladders.append({
                'base_pos': ladder_base,
                'height': ladder_data['height']
            })

        # Scenario A: Path Finding - Verify move sequence
        if puzzle_type in ['path_finding', 'sequence_finding', 'main_path']:
            return cls._verify_path_physics(
                extracted_output, start_pos, cubes, ladders, grid_size
            )

        # Scenario B: Height Comparison - Verify height claims
        elif puzzle_type == 'height_comparison':
            return cls._verify_height_claims(
                extracted_output, puzzle_state
            )

        return 0.0

    @classmethod
    def _level3_state_tracking_reward(cls, extracted_output: Dict, identity: dict, **kwargs) -> float:
        """
        Level 3: State Tracking Reward
        Verifies if model is aware of where it is and where blocks are located

        Returns:
            float: Score based on state awareness [0.0, 1.0]
        """
        position_claims = extracted_output.get("position_claims", [])
        if not position_claims:
            return 0.5  # Neutral - no claims to verify

        # Get maze modules
        Position, _ = _get_maze_modules()

        puzzle_state = identity.get('puzzle_state', {})
        cubes = set()
        for cube_pos in puzzle_state.get('cubes', []):
            cubes.add(Position(cube_pos[0], cube_pos[1], cube_pos[2]))

        # Verify each position claim
        correct_claims = 0
        total_claims = len(position_claims)

        for x, y, z in position_claims:
            claimed_pos = Position(x, y, z)
            if claimed_pos in cubes:
                correct_claims += 1

        if total_claims == 0:
            return 0.5

        accuracy = correct_claims / total_claims

        # Reward accurate state tracking
        if accuracy == 1.0:
            return 1.0
        elif accuracy >= 0.8:
            return 0.8
        elif accuracy >= 0.5:
            return 0.5
        else:
            return 0.0  # Penalize inaccurate state tracking

    @classmethod
    def _verify_path_physics(cls, extracted_output: Dict, start_pos,
                            cubes: set, ladders: list, grid_size: tuple) -> float:
        """
        Verify if described move actions follow physics rules

        Returns:
            float: Physics validity score
        """
        # Get maze modules
        Position, is_path_valid = _get_maze_modules()

        move_actions = extracted_output.get("move_actions", [])
        if not move_actions:
            return 0.5  # Neutral if no moves described

        # Simulate moves
        current_pos = start_pos
        valid_moves = 0
        invalid_moves = 0

        for action_data in move_actions:
            action = action_data.get("action")

            if action == "move":
                direction = action_data.get("direction", "").lower()
                distance = action_data.get("distance", 1)

                delta_map = {
                    "right": (distance, 0, 0), "left": (-distance, 0, 0),
                    "forward": (0, distance, 0), "backward": (0, -distance, 0)
                }

                dx, dy, dz = delta_map.get(direction, (0, 0, 0))
                delta = Position(dx, dy, dz)

                # Check if move is valid
                if is_path_valid(current_pos, delta, cubes, grid_size):
                    current_pos = current_pos + delta
                    valid_moves += 1
                else:
                    invalid_moves += 1  # Hallucination penalty

            elif action == "climb":
                height = action_data.get("height", 1)

                # Check if ladder exists
                ladder_exists = any(
                    l['base_pos'] == current_pos and l['height'] == height
                    for l in ladders
                )

                if ladder_exists:
                    current_pos = Position(current_pos.x, current_pos.y, current_pos.z + height)
                    valid_moves += 1
                else:
                    invalid_moves += 1  # Hallucination penalty

        total_moves = valid_moves + invalid_moves
        if total_moves == 0:
            return 0.5

        # Calculate physics score with hallucination penalty
        physics_score = valid_moves / total_moves

        # Apply penalty for hallucinations
        if invalid_moves > 0:
            hallucination_penalty = min(invalid_moves * 0.2, 0.5)
            physics_score = max(0.0, physics_score - hallucination_penalty)

        return physics_score

    @classmethod
    def _verify_height_claims(cls, extracted_output: Dict, puzzle_state: dict) -> float:
        """
        Verify height comparison claims against ground truth

        Returns:
            float: Accuracy of height claims
        """
        height_claims = extracted_output.get("height_claims", [])
        if not height_claims:
            return 0.5

        # Get sequence points
        sequence_points = {}
        for sp_data in puzzle_state.get('sequence_points', []):
            label = sp_data['label']
            z = sp_data['pos'][2]
            sequence_points[label] = z

        correct_claims = 0
        total_claims = len(height_claims)

        for claim in height_claims:
            if "height" in claim:
                # Direct height claim
                label = claim["label"]
                claimed_height = claim["height"]
                actual_height = sequence_points.get(label)

                if actual_height == claimed_height:
                    correct_claims += 1

            elif "relation" in claim:
                # Comparison claim
                label1 = claim["label1"]
                relation = claim["relation"]
                label2 = claim["label2"]

                h1 = sequence_points.get(label1)
                h2 = sequence_points.get(label2)

                if h1 is not None and h2 is not None:
                    if relation == "higher" and h1 > h2:
                        correct_claims += 1
                    elif relation == "lower" and h1 < h2:
                        correct_claims += 1

        if total_claims == 0:
            return 0.5

        return correct_claims / total_claims

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Normalize answer for comparison"""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', answer.strip())
        # Lowercase
        normalized = normalized.lower()
        # Remove punctuation at the end
        normalized = normalized.rstrip('.,;:!?')
        return normalized

    @classmethod
    def _calculate_partial_outcome_reward(cls, predicted: str, correct: str, identity: dict) -> float:
        """
        Calculate partial credit for partially correct answers

        Returns:
            float: Partial score [0.0, 1.0]
        """
        puzzle_type = identity.get('puzzle_type', 'unknown')

        # For sequence finding, check partial sequence match
        if puzzle_type == 'sequence_finding':
            predicted_parts = predicted.split('->')
            correct_parts = correct.split('->')

            if len(predicted_parts) != len(correct_parts):
                return 0.0

            matches = sum(1 for p, c in zip(predicted_parts, correct_parts)
                         if p.strip() == c.strip())
            return matches / len(correct_parts) * 0.5  # Max 0.5 for partial sequence

        # For path finding, check partial branch match
        elif puzzle_type == 'path_finding':
            predicted_branches = predicted.split(',')
            correct_branches = correct.split(',')

            if len(predicted_branches) != len(correct_branches):
                return 0.0

            matches = sum(1 for p, c in zip(predicted_branches, correct_branches)
                         if p.strip() == c.strip())
            return matches / len(correct_branches) * 0.5  # Max 0.5 for partial path

        # No partial credit for other types
        return 0.0
