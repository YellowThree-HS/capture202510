#!/usr/bin/env python3
"""
ArUco板子手眼标定验证程序

使用ArUco板子来验证手眼标定结果的准确性
按下空格键进行检测，显示ArUco板相对于机械臂末端和基座的位置
"""

import cv2
import numpy as np
import os
import sys
from lib.camera import Camera
from lib.dobot import DobotRobot
from scipy.spatial.transform import Rotation as R


class ArUcoVerification:
    def __init__(self, calibration_file='best_hand_eye_calibration.npy', robot_ip='192.168.5.2', arm_side='left'):
        """
        初始化ArUco验证系统
        
        参数:
            calibration_file: 手眼标定矩阵文件路径
            robot_ip: 机械臂IP地址
            arm_side: 机械臂侧别 ('left' 或 'right')
        """
        self.calibration_file = calibration_file
        self.robot_ip = robot_ip
        self.arm_side = arm_side
        
        # ArUco参数设置
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # ArUco板子参数 (8x8cm)
        self.marker_size = 0.08  # 8cm = 0.08m
        self.board_size = (4, 4)  # 4x4的ArUco板子
        
        # 创建ArUco板子
        self.create_aruco_board()
        
        # 相机方向校正矩阵（根据左右臂相机安装方向不同）
        self.camera_orientation_correction = self.get_camera_orientation_correction()
        
        # 初始化相机和机械臂
        self.camera = None
        self.robot = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.hand_eye_matrix = None
        
        self.initialize_systems()
    
    def create_aruco_board(self):
        """创建ArUco板子"""
        # 创建4x4的ArUco板子，每个marker大小为8cm
        self.board = cv2.aruco.GridBoard(
            size=(self.board_size[0], self.board_size[1]),
            markerLength=self.marker_size,
            markerSeparation=0.01,  # 1cm间距
            dictionary=self.aruco_dict
        )
        print(f"✓ ArUco板子创建完成: {self.board_size[0]}x{self.board_size[1]}, 标记大小: {self.marker_size}m")
    
    def get_camera_orientation_correction(self):
        """
        根据左右臂相机安装方向获取校正矩阵
        
        返回:
            4x4校正矩阵
        """
        if self.arm_side == 'left':
            # 左臂相机校正矩阵（假设相机正常安装）
            correction = np.eye(4)
            print(f"✓ 左臂相机使用标准方向")
        else:  # right arm
            # 右臂相机校正矩阵（相机反向安装）
            # 根据实际安装情况，可能需要不同的校正矩阵
            # 这里提供几种常见的校正方案：
            
            
            # 方案2: 翻转X和Y轴方向（保持旋转矩阵正确，只翻转坐标正负）
            correction = np.array([
                [-1,  0,  0,  0],  # 翻转X轴方向
                [ 0, -1,  0,  0],  # 翻转Y轴方向
                [ 0,  0,  1,  0],  # Z轴保持不变
                [ 0,  0,  0,  1]
            ])
            

            
            
            print(f"✓ 右臂相机方向校正矩阵已应用（翻转X和Y轴方向）")
        
        return correction
    
    def initialize_systems(self):
        """初始化相机和机械臂系统"""
        try:
            # 初始化相机
            print("正在初始化相机...")
            self.camera = Camera(camera_model='AUTO')
            
            # 获取相机内参
            self.camera_matrix = self.camera.get_camera_matrix('color')
            self.dist_coeffs = self.camera.get_distortion_coeffs('color')
            
            if self.camera_matrix is None:
                raise RuntimeError("无法获取相机内参")
            
            print("✓ 相机初始化成功")
            print(f"  相机内参矩阵:\n{self.camera_matrix}")
            print(f"  畸变系数: {self.dist_coeffs}")
            
            # 初始化机械臂
            print("正在初始化机械臂...")
            self.robot = DobotRobot(robot_ip=self.robot_ip, no_gripper=True)
            print("✓ 机械臂初始化成功")
            
            # 加载手眼标定矩阵
            self.load_hand_eye_calibration()
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            sys.exit(1)
    
    def load_hand_eye_calibration(self):
        """加载手眼标定矩阵"""
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(f"手眼标定文件不存在: {self.calibration_file}")
        
        self.hand_eye_matrix = np.load(self.calibration_file)
        print(f"✓ 手眼标定矩阵加载成功")
        print(f"  标定矩阵:\n{self.hand_eye_matrix}")
    
    def detect_aruco_pose(self, image):
        """
        检测ArUco板子的位姿
        
        参数:
            image: 输入图像
            
        返回:
            success: 是否检测成功
            rvec: 旋转向量
            tvec: 平移向量
            corners: 检测到的角点
            ids: 检测到的ID
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测ArUco标记
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None or len(ids) < 1:  # 至少需要1个标记
            return False, None, None, None, None
        
        # 对于单个标记，使用estimatePoseSingleMarkers
        if len(ids) == 1:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            success = True
        else:
            # 对于多个标记，尝试估计板子位姿
            success, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners, ids, self.board, self.camera_matrix, self.dist_coeffs, None, None
            )
        
        return success, rvec, tvec, corners, ids
    
    def draw_aruco_detection(self, image, corners, ids, rvec, tvec):
        """在图像上绘制ArUco检测结果"""
        # 绘制检测到的标记
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
        # 绘制坐标轴
        if rvec is not None and tvec is not None:
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
        
        return image
    
    def calculate_poses(self, rvec, tvec, debug=False):
        """
        计算ArUco板相对于机械臂基座的位置（考虑相机安装方向）
        
        参数:
            rvec: ArUco板相对于相机的旋转向量
            tvec: ArUco板相对于相机的平移向量
            debug: 是否返回调试信息
            
        返回:
            aruco_to_base: ArUco板相对于机械臂基座的变换矩阵
            debug_info: 调试信息（如果debug=True）
        """
        # 获取当前机械臂末端位姿
        end_pose = self.robot.get_pose_matrix()  # 4x4变换矩阵
        
        # ArUco板相对于相机的变换矩阵
        aruco_to_cam = np.eye(4)
        aruco_to_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
        aruco_to_cam[:3, 3] = tvec.flatten()


        print(f"aruco_to_cam",aruco_to_cam)

        self.hand_eye_matrix[2, 3] = -0.16371493
        print(f"self.hand_eye_matrix",self.hand_eye_matrix)
        aruco_to_base = end_pose @ self.hand_eye_matrix @ aruco_to_cam
        print(f"aruco_to_base",aruco_to_base)

        
        return aruco_to_base
    
    def format_pose_output(self, pose_matrix, debug_info=None):
        """格式化位姿输出 - 显示ArUco板相对于基座的位置和完整变换矩阵"""
        position = pose_matrix[:3, 3]
        rotation_matrix = pose_matrix[:3, :3]
        
        output = f"ArUco板相对于机械臂基座位置 (mm): X={position[0]*1000:.2f}, Y={position[1]*1000:.2f}, Z={position[2]*1000:.2f}\n"
        output += f"ArUco板相对于机械臂基座变换矩阵:\n{pose_matrix}"
        
        if debug_info:
            output += f"\n\n调试信息:\n"
            output += f"校正前位姿: {debug_info['before_correction']}\n"
            output += f"校正后位姿: {debug_info['after_correction']}\n"
            output += f"相机方向校正矩阵:\n{self.camera_orientation_correction}"
        
        return output
    
    
    def run_verification(self):
        """运行验证程序"""
        print("\n" + "="*60)
        print("ArUco手眼标定验证程序")
        print("="*60)
        print("操作说明:")
        print("  按空格键: 检测ArUco板并显示位姿信息")
        print("  按 'q' 键: 退出程序")
        print("="*60)
        
        cv2.namedWindow('ArUco Detection', cv2.WINDOW_AUTOSIZE)
        
        while True:
            # 获取相机图像
            color_image, _ = self.camera.get_frames()
            if color_image is None:
                print("无法获取相机图像")
                continue
            
            # 检测ArUco板子
            success, rvec, tvec, corners, ids = self.detect_aruco_pose(color_image)
            
            # 绘制检测结果
            display_image = color_image.copy()
            if success:
                display_image = self.draw_aruco_detection(display_image, corners, ids, rvec, tvec)
                cv2.putText(display_image, "ArUco Board Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "No ArUco Board Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示图像
            cv2.imshow('ArUco Detection', display_image)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # 空格键
                if success:
                    print("\n" + "="*50)
                    print("检测到ArUco板子，计算位姿...")
                    
                    # 计算位姿（带调试信息）
                    aruco_to_base = self.calculate_poses(rvec, tvec, debug=True)
                    
                    
                    print("="*50)
                else:
                    print("❌ 未检测到ArUco板子，请确保板子在相机视野内")
            
            elif key == ord('q'):  # 退出
                break
        
        # 清理资源
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.release()
        print("✓ 程序已退出")


def main():
    """主函数"""
    try:
        # 创建验证系统
        # 根据使用的机械臂选择对应的标定文件和侧别
        arm_side = 'right'  # 或 'left'
        calibration_file = f'./hand_eye_calib_{arm_side}.npy'
        
        print(f"使用{arm_side}臂标定文件: {calibration_file}")
        verifier = ArUcoVerification(calibration_file, arm_side=arm_side)
        
        # 运行验证
        verifier.run_verification()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
