"""硬件模拟件：合成场景、仿真相机、仿真机器人。"""
from .scene import Scene, WorkpieceGT, random_scene, render_scene, CLASSES
from .camera import SimCamera, make_top_down_homography
from .robot_mock import RobotMock

__all__ = [
    "Scene",
    "WorkpieceGT",
    "random_scene",
    "render_scene",
    "CLASSES",
    "SimCamera",
    "make_top_down_homography",
    "RobotMock",
]
