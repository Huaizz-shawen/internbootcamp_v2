import sys
import os
import random
import json
from typing import Dict, Any, Optional
from internbootcamp.src.base_instruction_generator import BaseInstructionGenerator

# Lazy imports - only import when needed to avoid matplotlib dependency at import time
_puzzle_generator_imported = False
_PuzzleGenerator = None
_QAGenerator = None
_get_plot_level = None
_draw_puzzle = None

def _add_3d_maze_to_path():
    """Add 3d_maze directory to sys.path"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, '../../..'))
    maze_3d_path = os.path.join(repo_root, '3d_maze_origin')
    if maze_3d_path not in sys.path:
        sys.path.insert(0, maze_3d_path)

def _ensure_imports():
    """Lazy import of 3d_maze modules"""
    global _puzzle_generator_imported, _PuzzleGenerator, _QAGenerator, _get_plot_level, _draw_puzzle
    if not _puzzle_generator_imported:
        _add_3d_maze_to_path()
        from main import PuzzleGenerator, QAGenerator, get_plot_level, draw_puzzle
        _PuzzleGenerator = PuzzleGenerator
        _QAGenerator = QAGenerator
        _get_plot_level = get_plot_level
        _draw_puzzle = draw_puzzle
        _puzzle_generator_imported = True
    return _PuzzleGenerator, _QAGenerator, _get_plot_level, _draw_puzzle


class Maze3DInstructionGenerator(BaseInstructionGenerator):
    """3D Maze Instruction Generator for RLVR training"""

    def __init__(self,
                 grid_size: tuple = (8, 8, 7),
                 puzzle_types: list = None,
                 main_path_length: tuple = (5, 7),
                 side_path_num: tuple = (3, 4),
                 side_path_length: tuple = (1, 2),
                 label_num_range: tuple = (3, 4),
                 save_images: bool = False,
                 output_dir: str = None,
                 seed: Optional[int] = None):
        """
        Initialize 3D Maze Instruction Generator

        Args:
            grid_size (tuple): Size of the 3D grid (x, y, z)
            puzzle_types (list): List of puzzle types to generate
                ['path_finding', 'sequence_finding', 'height_comparison', 'main_path']
            main_path_length (tuple): Range for main path segments
            side_path_num (tuple): Range for number of side paths
            side_path_length (tuple): Range for side path segments
            label_num_range (tuple): Range for number of labeled points
            save_images (bool): Whether to save puzzle images
            output_dir (str): Directory to save images and states
            seed (Optional[int]): Random seed for reproducibility
        """
        super().__init__()
        self.grid_size = grid_size
        self.puzzle_types = puzzle_types or ['path_finding', 'sequence_finding',
                                              'height_comparison', 'main_path']
        self.main_path_length = main_path_length
        self.side_path_num = side_path_num
        self.side_path_length = side_path_length
        self.label_num_range = label_num_range
        self.save_images = save_images
        self.output_dir = output_dir or "maze3d_outputs"
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)

        # Create output directories if saving images
        if self.save_images:
            os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'states'), exist_ok=True)

        # Lazy initialization - create generators when first needed
        self.puzzle_generator = None
        self.qa_generator = None

    def _ensure_generators(self):
        """Ensure puzzle generators are initialized (lazy loading)"""
        if self.puzzle_generator is None:
            PuzzleGenerator, QAGenerator, _, _ = _ensure_imports()
            self.puzzle_generator = PuzzleGenerator(grid_size=self.grid_size)
            self.qa_generator = QAGenerator()
            self.qa_generator.puzzle_generator = self.puzzle_generator

    def case_generator(self) -> Dict[str, Any]:
        """
        Generate a 3D maze puzzle task

        Returns:
            Dict[str, Any]: Puzzle task information including puzzle state, question, and answer
        """
        # Ensure generators are initialized
        self._ensure_generators()

        # Get helper functions
        _, _, get_plot_level, draw_puzzle = _ensure_imports()

        # Select random puzzle type
        puzzle_type = random.choice(self.puzzle_types)

        # Generate puzzle with retry logic
        max_retries = 20
        for attempt in range(max_retries):
            try:
                # Generate puzzle based on type
                if puzzle_type == 'path_finding':
                    puzzle = self.puzzle_generator.generate_path_finding_puzzle(
                        main_path_length=self.main_path_length,
                        side_path_num=self.side_path_num,
                        side_path_length=self.side_path_length
                    )
                    question_text = self._generate_path_finding_question(puzzle)
                    answer = self._get_path_finding_answer(puzzle)

                elif puzzle_type == 'sequence_finding':
                    puzzle = self.puzzle_generator.generate_sequence_puzzle(
                        label_num_range=self.label_num_range
                    )
                    question_text = self._generate_sequence_question(puzzle)
                    answer = self._get_sequence_answer(puzzle)

                elif puzzle_type == 'height_comparison':
                    puzzle = self.puzzle_generator.generate_sequence_puzzle((3, 3))
                    question_text = self._generate_height_comparison_question(puzzle)
                    answer = self._get_height_comparison_answer(puzzle)

                elif puzzle_type == 'main_path':
                    puzzle = self.puzzle_generator.generate_path_finding_puzzle(
                        main_path_length=self.main_path_length,
                        side_path_num=self.side_path_num,
                        side_path_length=self.side_path_length
                    )
                    question_text = self._generate_main_path_question(puzzle)
                    answer = self._get_main_path_answer(puzzle)
                else:
                    raise ValueError(f"Unknown puzzle type: {puzzle_type}")

                # Serialize puzzle state
                puzzle_state = self._serialize_puzzle_state(puzzle)

                # Save image if requested
                image_path = None
                if self.save_images:
                    from uuid import uuid4
                    puzzle_id = str(uuid4())
                    image_path = os.path.join(self.output_dir, 'images', f'{puzzle_id}.png')
                    draw_puzzle(puzzle, image_path)

                    # Save state
                    state_path = os.path.join(self.output_dir, 'states', f'{puzzle_id}.json')
                    with open(state_path, 'w') as f:
                        json.dump(puzzle_state, f, indent=2)

                return {
                    "puzzle_type": puzzle_type,
                    "puzzle_state": puzzle_state,
                    "question": question_text,
                    "answer": answer,
                    "grid_size": self.grid_size,
                    "image_path": image_path,
                    "difficulty": get_plot_level(puzzle.cubes)
                }

            except Exception as e:
                if "Cannot generate valid path" in str(e) and attempt < max_retries - 1:
                    continue
                raise e

        raise Exception(f"Failed to generate valid puzzle after {max_retries} attempts")

    def prompt_func(self, identity: Dict[str, Any]) -> str:
        """
        Generate prompt for the LLM

        Args:
            identity (Dict[str, Any]): Task information from case_generator

        Returns:
            str: Prompt for the LLM
        """
        puzzle_type = identity['puzzle_type']
        question = identity['question']
        puzzle_state = identity['puzzle_state']

        prompt = f"""You are an expert at solving 3D maze puzzles. You can navigate through a 3D grid maze by using the available tools.

**3D Maze Rules:**
1. You can only walk on top of cubes
2. You can climb ladders if you can reach the cube under the ladder
3. From a ladder, you can reach the top of the last cube covered by the ladder
4. Blue cube is the start position, red cube is the goal position
5. You can move in the X direction (left/right) or Y direction (forward/backward)
6. Ladders allow you to move up in the Z direction (height)

**Puzzle Information:**
{question}

**Available Tools:**
- `move`: Move in a direction (requires: direction, distance)
- `climb`: Climb a ladder (requires: height)
- `check_position`: Check your current position
- `get_height`: Get the height (z-coordinate) of your current position or a specific position
- `describe_surroundings`: Get information about nearby cubes and ladders
- `analyze_path`: Analyze a proposed path to see if it's valid

**Important Instructions:**
1. Use tools to explore and navigate the maze step by step
2. When stating positions or heights, verify them using the appropriate tools
3. Think carefully about the physical constraints before each move
4. Track your current state throughout your reasoning

**Final Answer Format:**
When you have determined the final answer, provide it in the following JSON format:
```json
{{
    "answer": "your answer here",
    "reasoning_steps": ["step 1", "step 2", ...]
}}
```

Begin solving the puzzle now."""

        return prompt

    def _serialize_puzzle_state(self, puzzle) -> Dict[str, Any]:
        """Serialize puzzle state to JSON-compatible dict"""
        state = {
            "grid_size": list(puzzle.grid_size),
            "cubes": [[p.x, p.y, p.z] for p in puzzle.cubes],
            "start_pos": [puzzle.start_pos.x, puzzle.start_pos.y, puzzle.start_pos.z],
            "goal_pos": [puzzle.goal_pos.x, puzzle.goal_pos.y, puzzle.goal_pos.z],
            "ladders": [
                {
                    "base_pos": [l.base_pos.x, l.base_pos.y, l.base_pos.z],
                    "direction": l.direction,
                    "height": l.height
                }
                for l in puzzle.ladders
            ],
            "path": [
                {
                    "start": [s.start.x, s.start.y, s.start.z],
                    "end": [s.end.x, s.end.y, s.end.z],
                    "type": s.type
                }
                for s in puzzle.path
            ]
        }

        # Add type-specific fields
        if hasattr(puzzle, 'branches'):
            state["branches"] = [
                {
                    "pos": [b.pos.x, b.pos.y, b.pos.z],
                    "branch_id": b.branch_id
                }
                for b in puzzle.branches
            ]

        if hasattr(puzzle, 'sequence_points'):
            state["sequence_points"] = [
                {
                    "pos": [sp.pos.x, sp.pos.y, sp.pos.z],
                    "label": sp.label
                }
                for sp in puzzle.sequence_points
            ]

        return state

    def _generate_path_finding_question(self, puzzle) -> str:
        """Generate path-finding question"""
        num_branches = len(puzzle.branches)
        return f"""This is a path-finding puzzle with {num_branches} branch points.
At each numbered branch point, you must choose the correct direction to reach the goal.
Determine which direction to take at each branch point to successfully reach the red goal cube from the blue start cube."""

    def _get_path_finding_answer(self, puzzle) -> str:
        """Get correct answer for path-finding puzzle"""
        _add_3d_maze_to_path()
        from main import get_branch_order, get_path_direction, Position

        def get_main_path_direction_at_branch(branch_pos, path):
            for segment in path:
                if segment.start == branch_pos:
                    return get_path_direction(segment)
                if segment.type == 'walk':
                    delta = Position(
                        segment.end.x - segment.start.x,
                        segment.end.y - segment.start.y,
                        segment.end.z - segment.start.z
                    )
                    intermediate_pos = Position(
                        segment.start.x + delta.x // 2,
                        segment.start.y + delta.y // 2,
                        segment.start.z
                    )
                    if intermediate_pos == branch_pos:
                        for next_segment in path:
                            if next_segment.start == segment.end:
                                return get_path_direction(next_segment)
            return "??"

        ordered_branches = get_branch_order(puzzle.path, puzzle.branches)
        answer_parts = []
        for b in ordered_branches:
            main_direction = get_main_path_direction_at_branch(b.pos, b.paths['A'])
            answer_parts.append(f"{b.branch_id}-{main_direction}")

        return ", ".join(answer_parts)

    def _generate_sequence_question(self, puzzle) -> str:
        """Generate sequence-finding question"""
        num_points = len(puzzle.sequence_points)
        return f"""This is a sequence-finding puzzle with {num_points} numbered checkpoints (green cubes).
Determine the correct order in which you encounter these numbered checkpoints when following the path from start to goal."""

    def _get_sequence_answer(self, puzzle) -> str:
        """Get correct answer for sequence-finding puzzle"""
        _add_3d_maze_to_path()
        from main import get_ordered_path_cubes

        ordered_cubes = get_ordered_path_cubes(puzzle.path)
        sequence = ["Start"]

        for cube in ordered_cubes:
            for sp in sorted(puzzle.sequence_points, key=lambda x: x.label):
                if sp.pos == cube:
                    sequence.append(str(sp.label))
        sequence.append("Goal")

        return " -> ".join(sequence)

    def _generate_height_comparison_question(self, puzzle) -> str:
        """Generate height comparison question"""
        return """This puzzle has 3 numbered points (green cubes).
Determine the height relationship between these three points using '<' for 'lower than' and '=' for 'same height as'.
For example: "1 < 2 = 3" means point 1 is lowest, and points 2 and 3 are at the same height."""

    def _get_height_comparison_answer(self, puzzle) -> str:
        """Get correct answer for height comparison puzzle"""
        _add_3d_maze_to_path()
        from main import normalize_height_relation

        heights = [(p.label, p.pos.z) for p in puzzle.sequence_points]
        return normalize_height_relation(heights)

    def _generate_main_path_question(self, puzzle) -> str:
        """Generate main path question"""
        # Add labels to cubes for this question type
        _add_3d_maze_to_path()
        from main import get_ordered_path_cubes, Branch

        cubes_remove = puzzle.cubes.copy()
        cubes_remove.remove(puzzle.start_pos)
        cubes_remove.remove(puzzle.goal_pos)

        label_cubes = self.puzzle_generator.choose_labeled_cubes(
            list(cubes_remove), random.randint(3, 4)
        )

        # Store labels in puzzle for answer extraction
        puzzle.labeled_cubes = label_cubes
        puzzle.branches = [Branch(cube.pos, cube.label, {'A': [], 'B': []})
                          for cube in label_cubes]

        num_labeled = len(label_cubes)
        return f"""This puzzle has {num_labeled} numbered blocks.
Determine which of these numbered blocks are on the main path from start to goal (not on dead-end side paths).
Answer with the block numbers separated by commas, or "None" if no numbered blocks are on the main path."""

    def _get_main_path_answer(self, puzzle) -> str:
        """Get correct answer for main path puzzle"""
        _add_3d_maze_to_path()
        from main import get_ordered_path_cubes

        main_path_cubes = get_ordered_path_cubes(puzzle.path)
        main_path_branches = []

        for cube in puzzle.labeled_cubes:
            if cube.pos in main_path_cubes:
                main_path_branches.append(str(cube.label))

        if not main_path_branches:
            return "None"
        return ", ".join(sorted(main_path_branches, key=int))
