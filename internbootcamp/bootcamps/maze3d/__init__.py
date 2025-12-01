"""
3D Maze RLVR Bootcamp

A comprehensive implementation of Reinforcement Learning with Verifiable Rewards (RLVR)
for 3D maze navigation and spatial reasoning tasks.

Features:
- Multiple puzzle types (path finding, sequence finding, height comparison, main path)
- Multi-round tool calling with 6 navigation tools
- Three-level reward system (Outcome, Physics/Logic, State Tracking)
- Hallucination detection and physics validation
"""

# Lazy imports to avoid matplotlib dependency at import time
def __getattr__(name):
    if name == 'Maze3DInstructionGenerator':
        from .maze3d_instruction_generator import Maze3DInstructionGenerator
        return Maze3DInstructionGenerator
    elif name == 'Maze3DRewardCalculator':
        from .maze3d_reward_calculator import Maze3DRewardCalculator
        return Maze3DRewardCalculator
    elif name == 'Maze3DTool':
        from .maze3d_tools import Maze3DTool
        return Maze3DTool
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'Maze3DInstructionGenerator',
    'Maze3DRewardCalculator',
    'Maze3DTool',
]

__version__ = '1.0.0'
