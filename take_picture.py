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
    cam = Camera(camera_model='D435')  # 初始化相机
    # robot = DobotRobot(robot_ip='192.168.5.1')  # 初始化机械臂

    print("相机已启动，按空格键拍照，按ESC键退出")
    
    while True:
        # 获取实时图像
        color_image, depth_image = cam.get_frames()
        
        # 显示实时图像
        cv2.imshow('Camera Feed - Press SPACE to capture, ESC to exit', color_image)
        
        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # 空格键
            # 拍照
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            color_filename = f'test/color_{timestamp}.png'
            depth_filename = f'test/depth_{timestamp}.png'
            
            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_filename, depth_image)
            
            print(f"照片已保存:")
            print(f"  彩色图: {color_filename}")
            print(f"  深度图: {depth_filename}")
            
            # 显示拍照成功的提示
            cv2.putText(color_image, "Photo Captured!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera Feed - Press SPACE to capture, ESC to exit', color_image)
            cv2.waitKey(1000)  # 显示1秒提示
            
        elif key == 27:  # ESC键
            print("退出程序")
            break
    
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main()