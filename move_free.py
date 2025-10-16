import cv2
import os
import time
import numpy as np
from datetime import datetime
from lib.camera import Camera
from lib.yolo_and_sam import YOLOSegmentator
from lib.mask2pose import mask2pose, draw_pose_axes
from lib.dobot import DobotRobot
from cv2 import aruco

def main():

    robot = DobotRobot(robot_ip='192.168.5.1')  # 初始化机械臂
    robot.r_inter.StartDrag()
    return 0
    

if __name__ == "__main__":
    main()