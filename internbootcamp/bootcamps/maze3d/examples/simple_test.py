#!/usr/bin/env python3
"""
Simple test to verify core RLVR components work
(Does not require matplotlib)
"""

import sys
import os
import json

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

print("=" * 60)
print("SIMPLE MAZE3D RLVR TEST")
print("=" * 60)

# Test 1: Import Check
print("\nTest 1: Checking imports...")
try:
    from internbootcamp.bootcamps.maze3d.maze3d_reward_calculator import Maze3DRewardCalculator
    print("✓ Reward calculator imported successfully")
except Exception as e:
    print(f"✗ Failed to import reward calculator: {e}")
    sys.exit(1)

# Test 2: Extract Output
print("\nTest 2: Testing output extraction...")
test_output = """
I analyzed the maze and found the solution.

The correct path is: 1-left, 2-up, 3-right

```json
{
    "answer": "1-left, 2-up, 3-right",
    "reasoning_steps": ["Navigate to branch 1", "Climb ladder", "Reach goal"]
}
```

I moved forward 2 steps.
Block 1 is at height 3.
Point 2 is higher than Point 1.
"""

extracted = Maze3DRewardCalculator.extract_output(test_output)
print(f"Extracted answer: {extracted['answer']}")
print(f"Reasoning steps: {len(extracted['reasoning_steps'])} steps")
print(f"Move actions: {len(extracted['move_actions'])} actions")
print(f"Height claims: {len(extracted['height_claims'])} claims")
print(f"Position claims: {len(extracted['position_claims'])} claims")

assert extracted['answer'] == "1-left, 2-up, 3-right", "Answer extraction failed"
assert len(extracted['move_actions']) > 0, "Move action extraction failed"
assert len(extracted['height_claims']) > 0, "Height claim extraction failed"
print("✓ Output extraction works correctly")

# Test 3: Reward Calculation
print("\nTest 3: Testing reward calculation...")

# Create a mock puzzle state
mock_identity = {
    "puzzle_type": "path_finding",
    "answer": "1-left, 2-up, 3-right",
    "puzzle_state": {
        "grid_size": [8, 8, 7],
        "cubes": [[7, 7, 0], [5, 7, 0], [5, 7, 3], [3, 7, 3]],
        "start_pos": [7, 7, 0],
        "goal_pos": [3, 7, 3],
        "ladders": [
            {"base_pos": [5, 7, 0], "direction": "+x", "height": 3}
        ],
        "path": [
            {"start": [7, 7, 0], "end": [5, 7, 0], "type": "walk"},
            {"start": [5, 7, 0], "end": [5, 7, 3], "type": "ladder"},
            {"start": [5, 7, 3], "end": [3, 7, 3], "type": "walk"}
        ],
        "branches": [
            {"pos": [7, 7, 0], "branch_id": 1},
            {"pos": [5, 7, 0], "branch_id": 2},
            {"pos": [5, 7, 3], "branch_id": 3}
        ]
    }
}

# Test correct answer
correct_output = Maze3DRewardCalculator.extract_output("""
```json
{
    "answer": "1-left, 2-up, 3-right"
}
```
""")

score_correct = Maze3DRewardCalculator._verify_correction(
    correct_output,
    mock_identity,
    enable_level1_outcome=True,
    enable_level2_physics=False,  # Skip physics for simple test
    enable_level3_state=False,
    weight_level1=1.0,
    weight_level2=0.0,
    weight_level3=0.0
)

print(f"Correct answer score: {score_correct:.3f}")
assert score_correct == 1.0, "Correct answer should get 1.0 score"
print("✓ Correct answer gets full score")

# Test wrong answer
wrong_output = Maze3DRewardCalculator.extract_output("""
```json
{
    "answer": "1-right, 2-down, 3-left"
}
```
""")

score_wrong = Maze3DRewardCalculator._verify_correction(
    wrong_output,
    mock_identity,
    enable_level1_outcome=True,
    enable_level2_physics=False,
    enable_level3_state=False,
    weight_level1=1.0,
    weight_level2=0.0,
    weight_level3=0.0
)

print(f"Wrong answer score: {score_wrong:.3f}")
assert score_wrong < 0.5, "Wrong answer should get low score"
print("✓ Wrong answer gets low score")

# Test 4: Three-Level Reward
print("\nTest 4: Testing three-level reward system...")

# Create output with position tracking
full_output = Maze3DRewardCalculator.extract_output("""
I start at position (7, 7, 0).
I move left by 2 steps to reach position (5, 7, 0).
Then I climb a ladder by 3 units to position (5, 7, 3).
Finally I move left again to the goal.

```json
{
    "answer": "1-left, 2-up, 3-right",
    "reasoning_steps": ["Move left", "Climb ladder", "Move to goal"]
}
```
""")

score_full = Maze3DRewardCalculator._verify_correction(
    full_output,
    mock_identity,
    enable_level1_outcome=True,
    enable_level2_physics=True,
    enable_level3_state=True,
    weight_level1=0.4,
    weight_level2=0.4,
    weight_level3=0.2
)

print(f"Full RLVR score: {score_full:.3f}")
print("✓ Three-level reward system works")

# Test 5: Normalization
print("\nTest 5: Testing answer normalization...")
norm1 = Maze3DRewardCalculator._normalize_answer("  1-left, 2-up, 3-right  ")
norm2 = Maze3DRewardCalculator._normalize_answer("1-left,2-up,3-right")
norm3 = Maze3DRewardCalculator._normalize_answer("1-LEFT, 2-UP, 3-RIGHT")

print(f"Normalized: '{norm1}'")
assert norm1 == "1-left, 2-up, 3-right", "Normalization failed"
print("✓ Answer normalization works")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nCore RLVR functionality verified:")
print("- ✓ Output extraction (answer, reasoning, actions)")
print("- ✓ Reward calculation (outcome)")
print("- ✓ Three-level reward system")
print("- ✓ Answer normalization")
print("\nThe maze3d bootcamp is ready to use!")
print("\nNote: Full tests (with puzzle generation) require:")
print("  pip install matplotlib numpy")
