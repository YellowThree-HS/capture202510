import os
import cv2
import numpy as np
import time
import argparse
from lib.robot import Robot
from lib.camera import Camera
from lib.dobot_api import DobotApi
from lib.yolo_and_sam import YOLOSegmentator
from scipy.spatial.transform import Rotation


# robot getpose
def main():
    global robot, cam, CAMERA_MATRIX, DISTORTION_COEFFS
    
    try:
        # 初始化相机
        print("正在初始化相机...")
        cam = Camera(camera_model='d405')
        CAMERA_MATRIX = cam.get_camera_matrix('color')
        DISTORTION_COEFFS = cam.get_distortion_coeffs('color')
        print("✓ 相机初始化成功")
        
        # 初始化机器人
        print(f"正在连接机器人 ({ROBOT_IP})...")
        robot = Robot(ROBOT_IP)
        
        if robot.connect():
            print("✓ 机器人连接成功")
            robot.enable()
            print("✓ 机器人已启用")
            
            # 获取当前位姿
            current_pose_matrix = robot.get_pose()
            if current_pose_matrix is not None:
                print("当前机器人位姿矩阵:\n", np.round(current_pose_matrix, 2))
            
        else:
            print("✗ 机器人连接失败")
    
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot:
            robot.disconnect()
        if cam:
            cam.release()
        cv2.destroyAllWindows()
        print("\n程序退出")
        
if __name__ == "__main__":
    main()