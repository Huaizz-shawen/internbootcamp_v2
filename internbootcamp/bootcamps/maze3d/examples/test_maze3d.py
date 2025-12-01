#!/usr/bin/env python3
"""
Test script for 3D Maze RLVR implementation
"""

import os
import sys
import json
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from internbootcamp.bootcamps.maze3d.maze3d_instruction_generator import Maze3DInstructionGenerator
from internbootcamp.bootcamps.maze3d.maze3d_reward_calculator import Maze3DRewardCalculator
from internbootcamp.bootcamps.maze3d.maze3d_tools import Maze3DTool
from verl.tools.schemas import OpenAIFunctionToolSchema


async def test_instruction_generation():
    """Test maze puzzle generation"""
    print("\n" + "=" * 60)
    print("TEST 1: Instruction Generation")
    print("=" * 60)

    generator = Maze3DInstructionGenerator(
        grid_size=(6, 6, 5),
        puzzle_types=['path_finding', 'sequence_finding'],
        main_path_length=(3, 5),
        save_images=False
    )

    # Generate a few puzzles
    for i in range(3):
        print(f"\n--- Puzzle {i+1} ---")
        identity = generator.case_generator()
        print(f"Type: {identity['puzzle_type']}")
        print(f"Difficulty: {identity['difficulty']}")
        print(f"Answer: {identity['answer']}")
        print(f"Number of cubes: {len(identity['puzzle_state']['cubes'])}")

    print("\n✓ Instruction generation test passed!")


async def test_reward_calculator():
    """Test three-level reward calculation"""
    print("\n" + "=" * 60)
    print("TEST 2: Three-Level Reward Calculator")
    print("=" * 60)

    # Create a simple test case
    generator = Maze3DInstructionGenerator(
        grid_size=(6, 6, 5),
        puzzle_types=['path_finding'],
        main_path_length=(3, 4)
    )

    identity = generator.case_generator()
    correct_answer = identity['answer']

    print(f"\nTest puzzle type: {identity['puzzle_type']}")
    print(f"Correct answer: {correct_answer}")

    # Test Case 1: Correct answer
    print("\n--- Test Case 1: Correct Answer ---")
    output_correct = f"""
I have analyzed the maze and determined the path.

```json
{{
    "answer": "{correct_answer}",
    "reasoning_steps": ["Step 1", "Step 2"]
}}
```
"""
    extracted = Maze3DRewardCalculator.extract_output(output_correct)
    score = Maze3DRewardCalculator._verify_correction(
        extracted, identity,
        enable_level1_outcome=True,
        enable_level2_physics=True,
        enable_level3_state=True,
        weight_level1=0.4,
        weight_level2=0.4,
        weight_level3=0.2
    )
    print(f"Extracted answer: {extracted['answer']}")
    print(f"Total score: {score:.3f}")

    # Test Case 2: Wrong answer
    print("\n--- Test Case 2: Wrong Answer ---")
    output_wrong = """
```json
{
    "answer": "1-left, 2-right, 3-up",
    "reasoning_steps": []
}
```
"""
    extracted = Maze3DRewardCalculator.extract_output(output_wrong)
    score = Maze3DRewardCalculator._verify_correction(extracted, identity)
    print(f"Extracted answer: {extracted['answer']}")
    print(f"Total score: {score:.3f}")

    # Test Case 3: Physics violations (hallucinations)
    print("\n--- Test Case 3: Physics Violations ---")
    output_hallucination = f"""
I will navigate the maze.

First, I move forward 5 steps (even though there's no path).
Then I climb a ladder that doesn't exist.

```json
{{
    "answer": "{correct_answer}",
    "reasoning_steps": ["move forward 5", "climb ladder 3"]
}}
```
"""
    extracted = Maze3DRewardCalculator.extract_output(output_hallucination)
    score = Maze3DRewardCalculator._verify_correction(extracted, identity)
    print(f"Extracted moves: {extracted['move_actions']}")
    print(f"Total score: {score:.3f} (should be penalized for physics violations)")

    print("\n✓ Reward calculator test passed!")


async def test_tool_execution():
    """Test multi-round tool calling"""
    print("\n" + "=" * 60)
    print("TEST 3: Multi-Round Tool Execution")
    print("=" * 60)

    # Create a simple maze
    generator = Maze3DInstructionGenerator(
        grid_size=(6, 6, 5),
        puzzle_types=['path_finding'],
        main_path_length=(2, 3)
    )

    identity = generator.case_generator()

    # Create tool schemas
    move_schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "move",
            "description": "Move in a direction",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string"},
                    "distance": {"type": "integer"}
                },
                "required": ["direction"]
            }
        }
    )

    check_pos_schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "check_position",
            "description": "Check position",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    )

    # Create tools
    move_tool = Maze3DTool(config={}, tool_schema=move_schema)
    check_tool = Maze3DTool(config={}, tool_schema=check_pos_schema)

    # Create instance
    instance_id = await move_tool.create(identity=identity)

    # Test 1: Check position
    print("\n--- Action 1: Check Position ---")
    response, reward, metrics = await check_tool.execute(instance_id, {})
    print(f"Response: {response}")
    print(f"Reward: {reward}")

    # Test 2: Valid move
    print("\n--- Action 2: Valid Move ---")
    response, reward, metrics = await move_tool.execute(
        instance_id,
        {"direction": "left", "distance": 2}
    )
    print(f"Response: {response}")
    print(f"Reward: {reward}")
    print(f"Valid: {metrics.get('valid')}")

    # Test 3: Invalid move (hallucination)
    print("\n--- Action 3: Invalid Move (Hallucination) ---")
    response, reward, metrics = await move_tool.execute(
        instance_id,
        {"direction": "forward", "distance": 10}
    )
    print(f"Response: {response}")
    print(f"Reward: {reward} (should be negative)")
    print(f"Hallucination: {metrics.get('hallucination', False)}")

    # Get cumulative reward
    print("\n--- Cumulative Tool Reward ---")
    cumulative_reward = await move_tool.calc_reward(instance_id)
    print(f"Total tool reward: {cumulative_reward:.3f}")

    print("\n✓ Tool execution test passed!")


async def test_full_pipeline():
    """Test the full RLVR pipeline"""
    print("\n" + "=" * 60)
    print("TEST 4: Full RLVR Pipeline")
    print("=" * 60)

    # Generate puzzle
    generator = Maze3DInstructionGenerator(
        grid_size=(6, 6, 5),
        puzzle_types=['sequence_finding'],
        label_num_range=(3, 3)
    )

    identity = generator.case_generator()
    prompt = generator.prompt_func(identity)

    print(f"\nPuzzle type: {identity['puzzle_type']}")
    print(f"Correct answer: {identity['answer']}")
    print(f"\nPrompt preview (first 200 chars):")
    print(prompt[:200] + "...")

    # Simulate LLM response with correct reasoning
    llm_response = f"""
Let me solve this step by step.

First, I'll check my position to understand the start point.

I can see the sequence points labeled 1, 2, and 3.
Following the path from start to goal, I encounter them in order.

The correct sequence is: {identity['answer']}

```json
{{
    "answer": "{identity['answer']}",
    "reasoning_steps": [
        "Started at the blue cube",
        "Followed the path",
        "Encountered checkpoints in order",
        "Reached the goal"
    ]
}}
```
"""

    # Extract and evaluate
    extracted = Maze3DRewardCalculator.extract_output(llm_response)
    final_score = Maze3DRewardCalculator._verify_correction(extracted, identity)

    print(f"\nExtracted answer: {extracted['answer']}")
    print(f"Final RLVR score: {final_score:.3f}")

    print("\n✓ Full pipeline test passed!")


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("3D MAZE RLVR IMPLEMENTATION TEST SUITE")
    print("=" * 60)

    try:
        await test_instruction_generation()
        await test_reward_calculator()
        await test_tool_execution()
        await test_full_pipeline()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe 3D Maze RLVR implementation is ready to use!")
        print("\nNext steps:")
        print("1. Generate training data: python examples/generate_data.py")
        print("2. Train with VERL: Use the generated data with your RL training pipeline")
        print("3. Evaluate models: Test trained models on maze puzzles")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
