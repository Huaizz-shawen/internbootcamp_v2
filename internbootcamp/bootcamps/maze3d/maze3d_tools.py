import sys
import os
import json
import logging
from typing import Any, Optional, Tuple, Dict, List, Set
from uuid import uuid4

from internbootcamp.src.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op

# Add 3d_maze to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../3d_maze'))
from main import Position, is_path_valid

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Maze3DTool(BaseTool):
    """3D Maze Navigation Tool for multi-round tool calling with RLVR"""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.tool_name = tool_schema.function.name

    async def create(self, instance_id: Optional[str] = None, identity: dict = None, **kwargs) -> str:
        """Create tool instance with maze state"""
        if instance_id is None:
            instance_id = str(uuid4())

        # Parse puzzle state
        puzzle_state = identity.get('puzzle_state', {})

        # Initialize maze state
        cubes = set()
        for cube_pos in puzzle_state.get('cubes', []):
            cubes.add(Position(cube_pos[0], cube_pos[1], cube_pos[2]))

        start_pos_list = puzzle_state.get('start_pos', [0, 0, 0])
        goal_pos_list = puzzle_state.get('goal_pos', [7, 7, 6])
        grid_size = tuple(puzzle_state.get('grid_size', [8, 8, 7]))

        start_pos = Position(start_pos_list[0], start_pos_list[1], start_pos_list[2])
        goal_pos = Position(goal_pos_list[0], goal_pos_list[1], goal_pos_list[2])

        # Parse ladders
        ladders = []
        for ladder_data in puzzle_state.get('ladders', []):
            ladder_base = Position(
                ladder_data['base_pos'][0],
                ladder_data['base_pos'][1],
                ladder_data['base_pos'][2]
            )
            ladders.append({
                'base_pos': ladder_base,
                'direction': ladder_data['direction'],
                'height': ladder_data['height']
            })

        # Parse ground truth path
        gt_path = []
        for segment in puzzle_state.get('path', []):
            gt_path.append({
                'start': Position(segment['start'][0], segment['start'][1], segment['start'][2]),
                'end': Position(segment['end'][0], segment['end'][1], segment['end'][2]),
                'type': segment['type']
            })

        # Parse sequence points if present
        sequence_points = []
        for sp_data in puzzle_state.get('sequence_points', []):
            sequence_points.append({
                'pos': Position(sp_data['pos'][0], sp_data['pos'][1], sp_data['pos'][2]),
                'label': sp_data['label']
            })

        # Parse branches if present
        branches = []
        for branch_data in puzzle_state.get('branches', []):
            branches.append({
                'pos': Position(branch_data['pos'][0], branch_data['pos'][1], branch_data['pos'][2]),
                'branch_id': branch_data['branch_id']
            })

        self._instance_dict[instance_id] = {
            "cubes": cubes,
            "start_pos": start_pos,
            "goal_pos": goal_pos,
            "current_pos": start_pos,
            "grid_size": grid_size,
            "ladders": ladders,
            "ground_truth_path": gt_path,
            "sequence_points": sequence_points,
            "branches": branches,
            "move_history": [],
            "action_count": 0,
            "valid_moves": 0,
            "invalid_moves": 0,
            "hallucinations": [],  # Track physics violations
            "state_errors": [],    # Track state tracking errors
            "reached_goal": False,
            "puzzle_type": identity.get('puzzle_type', 'unknown'),
            "answer": identity.get('answer', '')
        }

        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute tool based on tool name"""
        if instance_id not in self._instance_dict:
            return "Error: Invalid instance", -1.0, {}

        state = self._instance_dict[instance_id]
        state["action_count"] += 1

        # Route to appropriate handler based on tool name
        if self.tool_name == "move":
            return await self._handle_move(instance_id, parameters)
        elif self.tool_name == "climb":
            return await self._handle_climb(instance_id, parameters)
        elif self.tool_name == "check_position":
            return await self._handle_check_position(instance_id, parameters)
        elif self.tool_name == "get_height":
            return await self._handle_get_height(instance_id, parameters)
        elif self.tool_name == "describe_surroundings":
            return await self._handle_describe_surroundings(instance_id, parameters)
        elif self.tool_name == "analyze_path":
            return await self._handle_analyze_path(instance_id, parameters)
        else:
            return f"Error: Unknown tool '{self.tool_name}'", -0.5, {}

    async def _handle_move(self, instance_id: str, parameters: dict) -> Tuple[str, float, dict]:
        """Handle movement in X or Y direction"""
        state = self._instance_dict[instance_id]
        direction = parameters.get("direction", "").lower()
        distance = parameters.get("distance", 1)

        if direction not in ["left", "right", "forward", "backward", "+x", "-x", "+y", "-y"]:
            state["invalid_moves"] += 1
            return f"Error: Invalid direction '{direction}'. Use: left, right, forward, backward", -0.3, {
                "valid": False, "reason": "invalid_direction"
            }

        # Map direction to delta
        delta_map = {
            "right": (distance, 0, 0), "+x": (distance, 0, 0),
            "left": (-distance, 0, 0), "-x": (-distance, 0, 0),
            "forward": (0, distance, 0), "+y": (0, distance, 0),
            "backward": (0, -distance, 0), "-y": (0, -distance, 0)
        }

        dx, dy, dz = delta_map.get(direction, (0, 0, 0))
        delta = Position(dx, dy, dz)
        new_pos = state["current_pos"] + delta

        # Validate move
        if not is_path_valid(state["current_pos"], delta, state["cubes"], state["grid_size"]):
            state["invalid_moves"] += 1
            state["hallucinations"].append({
                "action": "move",
                "from": state["current_pos"].to_tuple(),
                "to": new_pos.to_tuple(),
                "reason": "invalid_physics"
            })
            return f"Invalid move! Cannot move {direction} by {distance} from {state['current_pos'].to_tuple()}. " \
                   f"Either out of bounds, no cube exists, or path is blocked.", -0.5, {
                "valid": False,
                "hallucination": True,
                "current_pos": state["current_pos"].to_tuple()
            }

        # Valid move - update position
        state["current_pos"] = new_pos
        state["valid_moves"] += 1
        state["move_history"].append({
            "action": "move",
            "direction": direction,
            "distance": distance,
            "from": (new_pos - delta).to_tuple(),
            "to": new_pos.to_tuple()
        })

        # Check if reached goal
        reward = 0.1  # Small positive reward for valid move
        if new_pos == state["goal_pos"]:
            state["reached_goal"] = True
            reward = 0.5  # Bonus for reaching goal

        return f"Moved {direction} by {distance}. Current position: {new_pos.to_tuple()}", reward, {
            "valid": True,
            "current_pos": new_pos.to_tuple(),
            "at_goal": state["reached_goal"]
        }

    async def _handle_climb(self, instance_id: str, parameters: dict) -> Tuple[str, float, dict]:
        """Handle climbing a ladder"""
        state = self._instance_dict[instance_id]
        height = parameters.get("height", 1)

        # Check if there's a ladder at current position
        ladder_at_pos = None
        for ladder in state["ladders"]:
            if ladder['base_pos'] == state["current_pos"]:
                ladder_at_pos = ladder
                break

        if not ladder_at_pos:
            state["invalid_moves"] += 1
            state["hallucinations"].append({
                "action": "climb",
                "position": state["current_pos"].to_tuple(),
                "reason": "no_ladder"
            })
            return f"Invalid climb! No ladder exists at position {state['current_pos'].to_tuple()}", -0.5, {
                "valid": False,
                "hallucination": True
            }

        # Check if height matches ladder height
        if height != ladder_at_pos['height']:
            state["invalid_moves"] += 1
            return f"Invalid climb height! Ladder height is {ladder_at_pos['height']}, not {height}", -0.3, {
                "valid": False,
                "correct_height": ladder_at_pos['height']
            }

        # Valid climb
        new_pos = Position(
            state["current_pos"].x,
            state["current_pos"].y,
            state["current_pos"].z + height
        )

        state["current_pos"] = new_pos
        state["valid_moves"] += 1
        state["move_history"].append({
            "action": "climb",
            "height": height,
            "from": (new_pos.x, new_pos.y, new_pos.z - height),
            "to": new_pos.to_tuple()
        })

        # Check if reached goal
        reward = 0.1
        if new_pos == state["goal_pos"]:
            state["reached_goal"] = True
            reward = 0.5

        return f"Climbed ladder by {height} units. Current position: {new_pos.to_tuple()}", reward, {
            "valid": True,
            "current_pos": new_pos.to_tuple(),
            "at_goal": state["reached_goal"]
        }

    async def _handle_check_position(self, instance_id: str, parameters: dict) -> Tuple[str, float, dict]:
        """Check current position"""
        state = self._instance_dict[instance_id]
        current_pos = state["current_pos"]

        response = f"Current position: {current_pos.to_tuple()}\n"
        response += f"Start position: {state['start_pos'].to_tuple()}\n"
        response += f"Goal position: {state['goal_pos'].to_tuple()}"

        return response, 0.0, {
            "current_pos": current_pos.to_tuple(),
            "start_pos": state["start_pos"].to_tuple(),
            "goal_pos": state["goal_pos"].to_tuple(),
            "at_goal": current_pos == state["goal_pos"]
        }

    async def _handle_get_height(self, instance_id: str, parameters: dict) -> Tuple[str, float, dict]:
        """Get height of current or specified position"""
        state = self._instance_dict[instance_id]

        # Check if specific position requested
        x = parameters.get("x")
        y = parameters.get("y")

        if x is not None and y is not None:
            # Find cube at (x, y)
            matching_cubes = [cube for cube in state["cubes"]
                            if cube.x == x and cube.y == y]
            if not matching_cubes:
                return f"No cube found at x={x}, y={y}", 0.0, {"found": False}

            # Return all heights at this x,y position
            heights = sorted([cube.z for cube in matching_cubes])
            return f"Height(s) at ({x}, {y}): {heights}", 0.0, {
                "x": x, "y": y, "heights": heights, "found": True
            }
        else:
            # Return current position height
            height = state["current_pos"].z
            return f"Current height (z-coordinate): {height}", 0.0, {
                "height": height,
                "position": state["current_pos"].to_tuple()
            }

    async def _handle_describe_surroundings(self, instance_id: str, parameters: dict) -> Tuple[str, float, dict]:
        """Describe nearby cubes and ladders"""
        state = self._instance_dict[instance_id]
        current_pos = state["current_pos"]

        # Find nearby cubes (within distance 2 in x,y and 1 in z)
        nearby_cubes = []
        for cube in state["cubes"]:
            dx = abs(cube.x - current_pos.x)
            dy = abs(cube.y - current_pos.y)
            dz = abs(cube.z - current_pos.z)
            if dx <= 2 and dy <= 2 and dz <= 1 and cube != current_pos:
                nearby_cubes.append(cube)

        # Find ladders at current position
        ladders_here = [l for l in state["ladders"] if l['base_pos'] == current_pos]

        # Build description
        response = f"Current position: {current_pos.to_tuple()}\n\n"

        if ladders_here:
            response += "Ladders at current position:\n"
            for ladder in ladders_here:
                response += f"  - Ladder going up {ladder['height']} units\n"
        else:
            response += "No ladders at current position\n"

        response += f"\nNearby cubes ({len(nearby_cubes)} found):\n"
        for cube in sorted(nearby_cubes, key=lambda c: (c.x, c.y, c.z)):
            rel_x = cube.x - current_pos.x
            rel_y = cube.y - current_pos.y
            rel_z = cube.z - current_pos.z
            response += f"  - {cube.to_tuple()} (relative: {rel_x:+d}, {rel_y:+d}, {rel_z:+d})\n"

        return response, 0.0, {
            "current_pos": current_pos.to_tuple(),
            "nearby_cubes": [c.to_tuple() for c in nearby_cubes],
            "ladders_here": len(ladders_here) > 0
        }

    async def _handle_analyze_path(self, instance_id: str, parameters: dict) -> Tuple[str, float, dict]:
        """Analyze if a proposed path is valid"""
        state = self._instance_dict[instance_id]

        # Get proposed moves
        moves = parameters.get("moves", [])
        if not moves:
            return "Error: No moves provided", -0.1, {"valid": False}

        # Simulate path
        sim_pos = state["current_pos"]
        valid = True
        invalid_step = None

        for i, move in enumerate(moves):
            action = move.get("action", "").lower()
            if action == "move":
                direction = move.get("direction", "").lower()
                distance = move.get("distance", 1)

                delta_map = {
                    "right": (distance, 0, 0), "+x": (distance, 0, 0),
                    "left": (-distance, 0, 0), "-x": (-distance, 0, 0),
                    "forward": (0, distance, 0), "+y": (0, distance, 0),
                    "backward": (0, -distance, 0), "-y": (0, -distance, 0)
                }

                dx, dy, dz = delta_map.get(direction, (0, 0, 0))
                delta = Position(dx, dy, dz)

                if not is_path_valid(sim_pos, delta, state["cubes"], state["grid_size"]):
                    valid = False
                    invalid_step = i
                    break

                sim_pos = sim_pos + delta

            elif action == "climb":
                height = move.get("height", 0)
                ladder_exists = any(l['base_pos'] == sim_pos for l in state["ladders"])

                if not ladder_exists:
                    valid = False
                    invalid_step = i
                    break

                sim_pos = Position(sim_pos.x, sim_pos.y, sim_pos.z + height)

        if valid:
            return f"Path is valid! Final position would be: {sim_pos.to_tuple()}", 0.05, {
                "valid": True,
                "final_pos": sim_pos.to_tuple()
            }
        else:
            return f"Path is invalid at step {invalid_step + 1}", 0.0, {
                "valid": False,
                "invalid_step": invalid_step
            }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        Calculate cumulative tool reward
        This is called at the end to provide additional reward signal
        """
        if instance_id not in self._instance_dict:
            return 0.0

        state = self._instance_dict[instance_id]

        # Base reward for efficient navigation
        action_count = state["action_count"]
        valid_moves = state["valid_moves"]
        invalid_moves = state["invalid_moves"]

        # Reward for move efficiency (fewer is better)
        efficiency_reward = max(0.0, 1.0 - (action_count - valid_moves) * 0.1)

        # Penalty for invalid moves and hallucinations
        hallucination_penalty = len(state["hallucinations"]) * 0.2
        invalid_move_penalty = invalid_moves * 0.1

        # Bonus for reaching goal
        goal_bonus = 1.0 if state["reached_goal"] else 0.0

        total_reward = efficiency_reward + goal_bonus - hallucination_penalty - invalid_move_penalty

        return max(0.0, min(total_reward, 2.0))  # Clip between 0 and 2
