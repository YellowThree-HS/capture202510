import numpy as np


from lib.dobot import DobotRobot

def main():
    # cam = Camera(camera_model='D405')  # 初始化相机
    robot = DobotRobot(robot_ip='192.168.5.1')  # 初始化机械臂
    robot.r_inter.StartDrag()
    # return 0
    
if __name__ == "__main__":
    main()