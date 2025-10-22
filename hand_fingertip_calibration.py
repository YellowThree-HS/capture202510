#!/usr/bin/env python3
"""
灵巧手指尖相对于机械臂末端位置标定脚本

标定流程：
1. 已知eye2end的4x4矩阵（手眼标定结果）
2. 移动机械臂到初始位置，按下空格拍照片
3. 识别标定板位置，得到obj2eye坐标
4. 通过get_pose_matrix得到end2base坐标
5. 计算obj2base = end2base @ eye2end @ obj2eye
6. 将机械臂移动到标定板中心，按下键确认
7. 读取新的hand2base'矩阵
8. 计算hand2end = obj2base @ inv(hand2base')
9. 保存hand2end数据到txt文件
10. 最后取平均值

使用方法：
- 空格键：拍照并检测标定板
- 'c'键：确认灵巧手位置并记录数据
- 'r'键：重置当前标定
- 'q'键：退出并计算平均值
- ESC键：退出程序
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from lib.camera import Camera
from lib.dobot import DobotRobot
from lib.inspire_hand import InspireHand


class HandFingertipCalibration:
    def __init__(self, 
                 eye2end_file='best_hand_eye_calibration.npy',
                 robot_ip='192.168.5.2',
                 hand_port='COM5',
                 calibration_board_type='chessboard'):
        """
        初始化灵巧手指尖标定系统
        
        参数:
            eye2end_file: 手眼标定矩阵文件路径
            robot_ip: 机械臂IP地址
            hand_port: 灵巧手串口端口
            calibration_board_type: 标定板类型 ('chessboard' 或 'aruco')
        """
        self.eye2end_file = eye2end_file
        self.robot_ip = robot_ip
        self.hand_port = hand_port
        self.calibration_board_type = calibration_board_type
        
        # 标定数据存储
        self.calibration_data = []
        self.current_obj2base = None
        self.current_end2base = None
        
        # 初始化系统组件
        self.camera = None
        self.robot = None
        self.hand = None
        self.eye2end_matrix = None
        
        # 标定板参数
        if calibration_board_type == 'chessboard':
            self.setup_chessboard_params()
        else:
            self.setup_aruco_params()
        
        # 初始化所有系统
        self.initialize_systems()
        
    def setup_chessboard_params(self):
        """设置棋盘格标定板参数"""
        self.board_size = (11, 8)  # 11x8棋盘格
        self.square_size = 0.02    # 2cm方格
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 准备棋盘格角点
        self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
    def setup_aruco_params(self):
        """设置ArUco标定板参数"""
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_size = 0.08  # 8cm标记
        self.board_size = (4, 4)  # 4x4 ArUco板
        
        # 创建ArUco板
        self.board = cv2.aruco.GridBoard(
            size=self.board_size,
            markerLength=self.marker_size,
            markerSeparation=0.01,  # 1cm间距
            dictionary=self.aruco_dict
        )
    
    def initialize_systems(self):
        """初始化相机、机械臂和灵巧手系统"""
        try:
            # 初始化相机
            print("正在初始化相机...")
            self.camera = Camera(camera_model='D435')
            print("✓ 相机初始化成功")
            
            # 初始化机械臂
            print("正在初始化机械臂...")
            self.robot = DobotRobot(robot_ip=self.robot_ip)
            print("✓ 机械臂初始化成功")
            
            # 初始化灵巧手
            print("正在初始化灵巧手...")
            self.hand = InspireHand(port=self.hand_port, baudrate=115200)
            print("✓ 灵巧手初始化成功")
            
            # 加载手眼标定矩阵
            self.load_eye2end_matrix()
            
            print("\n" + "="*60)
            print("系统初始化完成！")
            print("="*60)
            print("操作说明：")
            print("  空格键：拍照并检测标定板")
            print("  'c'键：确认灵巧手位置并记录数据")
            print("  'r'键：重置当前标定")
            print("  'q'键：退出并计算平均值")
            print("  ESC键：退出程序")
            print("="*60)
            
        except Exception as e:
            print(f"系统初始化失败: {e}")
            raise
    
    def load_eye2end_matrix(self):
        """加载手眼标定矩阵"""
        if not os.path.exists(self.eye2end_file):
            raise FileNotFoundError(f"手眼标定文件不存在: {self.eye2end_file}")
        
        self.eye2end_matrix = np.load(self.eye2end_file)
        print(f"✓ 手眼标定矩阵已加载: {self.eye2end_file}")
        print(f"  矩阵:\n{self.eye2end_matrix}")
    
    def detect_calibration_board(self, image):
        """
        检测标定板位姿
        
        参数:
            image: 输入图像
            
        返回:
            success: 是否检测成功
            obj2eye: 标定板相对于相机的变换矩阵
        """
        if self.calibration_board_type == 'chessboard':
            return self.detect_chessboard(image)
        else:
            return self.detect_aruco(image)
    
    def detect_chessboard(self, image):
        """检测棋盘格标定板"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        if not ret:
            return False, None
        
        # 亚像素精度优化
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        
        # 获取相机内参
        camera_matrix = self.camera.get_camera_matrix('color')
        dist_coeffs = self.camera.get_distortion_coeffs('color')
        
        if camera_matrix is None or dist_coeffs is None:
            print("警告：无法获取相机内参")
            return False, None
        
        # 求解PnP
        success, rvec, tvec = cv2.solvePnP(
            self.objp, corners2, camera_matrix, dist_coeffs
        )
        
        if not success:
            return False, None
        
        # 构建变换矩阵
        obj2eye = np.eye(4)
        obj2eye[:3, :3] = cv2.Rodrigues(rvec)[0]
        obj2eye[:3, 3] = tvec.flatten()
        
        return True, obj2eye
    
    def detect_aruco(self, image):
        """检测ArUco标定板"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测ArUco标记
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None or len(ids) < 1:
            return False, None
        
        # 获取相机内参
        camera_matrix = self.camera.get_camera_matrix('color')
        dist_coeffs = self.camera.get_distortion_coeffs('color')
        
        if camera_matrix is None or dist_coeffs is None:
            print("警告：无法获取相机内参")
            return False, None
        
        # 估计板子位姿
        success, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, ids, self.board, camera_matrix, dist_coeffs, None, None
        )
        
        if not success:
            return False, None
        
        # 构建变换矩阵
        obj2eye = np.eye(4)
        obj2eye[:3, :3] = cv2.Rodrigues(rvec)[0]
        obj2eye[:3, 3] = tvec.flatten()
        
        return True, obj2eye
    
    def calculate_obj2base(self, obj2eye):
        """
        计算标定板相对于机械臂基座的位姿
        
        参数:
            obj2eye: 标定板相对于相机的变换矩阵
            
        返回:
            obj2base: 标定板相对于机械臂基座的变换矩阵
        """
        # 获取当前机械臂末端位姿
        end2base = self.robot.get_pose_matrix()
        
        # 计算标定板相对于基座的位姿
        # obj2base = end2base @ eye2end @ obj2eye
        obj2base = end2base @ self.eye2end_matrix @ obj2eye
        
        return obj2base, end2base
    
    def calculate_hand2end(self, obj2base, hand2base_new):
        """
        计算灵巧手指尖相对于机械臂末端的位姿
        
        参数:
            obj2base: 标定板相对于基座的位姿
            hand2base_new: 新的灵巧手相对于基座的位姿
            
        返回:
            hand2end: 灵巧手指尖相对于机械臂末端的变换矩阵
        """
        # hand2end = obj2base @ inv(hand2base_new)
        hand2end = obj2base @ np.linalg.inv(hand2base_new)
        return hand2end
    
    def save_calibration_data(self, hand2end_matrix):
        """保存标定数据到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存到列表
        self.calibration_data.append(hand2end_matrix.copy())
        
        # 保存到文件
        filename = f"hand_fingertip_calibration_data_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(f"# 灵巧手指尖标定数据 - {timestamp}\n")
            f.write(f"# 数据格式: 4x4变换矩阵\n")
            f.write(f"# 矩阵:\n")
            np.savetxt(f, hand2end_matrix, fmt='%.6f')
            f.write(f"\n")
        
        print(f"✓ 标定数据已保存: {filename}")
        print(f"  当前已收集 {len(self.calibration_data)} 组数据")
    
    def calculate_average_calibration(self):
        """计算平均标定结果"""
        if len(self.calibration_data) == 0:
            print("没有标定数据可计算")
            return None
        
        print(f"\n正在计算 {len(self.calibration_data)} 组数据的平均值...")
        
        # 计算位置平均值
        positions = np.array([data[:3, 3] for data in self.calibration_data])
        avg_position = np.mean(positions, axis=0)
        
        # 计算旋转平均值（使用四元数）
        rotations = []
        for data in self.calibration_data:
            r = R.from_matrix(data[:3, :3])
            rotations.append(r.as_quat())
        
        avg_quat = np.mean(rotations, axis=0)
        avg_rotation = R.from_quat(avg_quat).as_matrix()
        
        # 构建平均变换矩阵
        avg_hand2end = np.eye(4)
        avg_hand2end[:3, :3] = avg_rotation
        avg_hand2end[:3, 3] = avg_position
        
        # 保存平均结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        avg_filename = f"hand_fingertip_calibration_average_{timestamp}.npy"
        np.save(avg_filename, avg_hand2end)
        
        # 保存详细结果
        result_filename = f"hand_fingertip_calibration_result_{timestamp}.txt"
        with open(result_filename, 'w') as f:
            f.write("灵巧手指尖标定结果\n")
            f.write("="*50 + "\n")
            f.write(f"标定时间: {timestamp}\n")
            f.write(f"数据组数: {len(self.calibration_data)}\n")
            f.write(f"平均位置 (m): [{avg_position[0]:.6f}, {avg_position[1]:.6f}, {avg_position[2]:.6f}]\n")
            f.write(f"平均位置 (mm): [{avg_position[0]*1000:.3f}, {avg_position[1]*1000:.3f}, {avg_position[2]*1000:.3f}]\n")
            f.write("\n平均变换矩阵:\n")
            np.savetxt(f, avg_hand2end, fmt='%.6f')
            f.write("\n\n所有标定数据:\n")
            for i, data in enumerate(self.calibration_data):
                f.write(f"\n数据组 {i+1}:\n")
                np.savetxt(f, data, fmt='%.6f')
        
        print(f"✓ 平均标定结果已保存:")
        print(f"  矩阵文件: {avg_filename}")
        print(f"  详细结果: {result_filename}")
        print(f"\n平均位置 (mm): [{avg_position[0]*1000:.3f}, {avg_position[1]*1000:.3f}, {avg_position[2]*1000:.3f}]")
        print(f"\n平均变换矩阵:\n{avg_hand2end}")
        
        return avg_hand2end
    
    def draw_calibration_info(self, image, obj2eye=None):
        """在图像上绘制标定信息"""
        # 绘制标定板检测结果
        if obj2eye is not None:
            # 获取相机内参
            camera_matrix = self.camera.get_camera_matrix('color')
            dist_coeffs = self.camera.get_distortion_coeffs('color')
            
            if camera_matrix is not None and dist_coeffs is not None:
                # 绘制坐标轴
                rvec = cv2.Rodrigues(obj2eye[:3, :3])[0]
                tvec = obj2eye[:3, 3]
                cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
        
        # 绘制状态信息
        status_text = [
            f"标定数据组数: {len(self.calibration_data)}",
            f"当前状态: {'等待拍照' if self.current_obj2base is None else '等待确认灵巧手位置'}",
            "",
            "操作说明:",
            "空格键: 拍照检测标定板",
            "c键: 确认灵巧手位置",
            "r键: 重置当前标定",
            "q键: 退出并计算平均值",
            "ESC键: 退出程序"
        ]
        
        y_offset = 30
        for text in status_text:
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return image
    
    def run_calibration(self):
        """运行标定程序"""
        print("开始灵巧手指尖标定...")
        
        try:
            while True:
                # 获取实时图像
                color_image, _ = self.camera.get_frames()
                if color_image is None:
                    continue
                
                # 绘制标定信息
                display_image = self.draw_calibration_info(color_image, None)
                
                # 显示图像
                cv2.imshow('Hand Fingertip Calibration', display_image)
                
                # 等待按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # 空格键 - 拍照检测标定板
                    print("\n正在检测标定板...")
                    
                    # 检测标定板
                    success, obj2eye = self.detect_calibration_board(color_image)
                    
                    if success:
                        print("✓ 标定板检测成功")
                        
                        # 计算标定板相对于基座的位姿
                        obj2base, end2base = self.calculate_obj2base(obj2eye)
                        self.current_obj2base = obj2base
                        self.current_end2base = end2base
                        
                        print(f"标定板位置 (mm): [{obj2base[0,3]*1000:.2f}, {obj2base[1,3]*1000:.2f}, {obj2base[2,3]*1000:.2f}]")
                        print("请将灵巧手移动到标定板中心，然后按'c'键确认")
                        
                        # 显示检测结果
                        display_image = self.draw_calibration_info(color_image, obj2eye)
                        cv2.imshow('Hand Fingertip Calibration', display_image)
                        cv2.waitKey(1000)  # 显示1秒
                    else:
                        print("✗ 标定板检测失败，请调整位置后重试")
                
                elif key == ord('c'):  # 'c'键 - 确认灵巧手位置
                    if self.current_obj2base is None:
                        print("请先按空格键检测标定板")
                        continue
                    
                    print("\n正在记录灵巧手位置...")
                    
                    # 获取当前机械臂位姿（此时灵巧手在标定板中心）
                    hand2base_new = self.robot.get_pose_matrix()
                    
                    # 计算灵巧手指尖相对于机械臂末端的位姿
                    hand2end = self.calculate_hand2end(self.current_obj2base, hand2base_new)
                    
                    # 保存标定数据
                    self.save_calibration_data(hand2end)
                    
                    # 重置当前标定
                    self.current_obj2base = None
                    self.current_end2base = None
                    
                    print("✓ 标定数据已记录")
                    print("请移动机械臂到新位置，按空格键继续标定")
                
                elif key == ord('r'):  # 'r'键 - 重置当前标定
                    self.current_obj2base = None
                    self.current_end2base = None
                    print("当前标定已重置")
                
                elif key == ord('q'):  # 'q'键 - 退出并计算平均值
                    break
                
                elif key == 27:  # ESC键 - 退出程序
                    print("程序被用户中断")
                    return
        
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"标定过程中出现错误: {e}")
        finally:
            cv2.destroyAllWindows()
        
        # 计算平均标定结果
        if len(self.calibration_data) > 0:
            print("\n" + "="*60)
            print("开始计算平均标定结果...")
            print("="*60)
            self.calculate_average_calibration()
        else:
            print("没有收集到标定数据")
    
    def cleanup(self):
        """清理资源"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    try:
        # 创建标定系统
        calibration = HandFingertipCalibration(
            eye2end_file='best_hand_eye_calibration.npy',
            robot_ip='192.168.5.2',
            hand_port='COM5',
            calibration_board_type='chessboard'  # 或 'aruco'
        )
        
        # 运行标定
        calibration.run_calibration()
        
    except Exception as e:
        print(f"程序运行失败: {e}")
    finally:
        if 'calibration' in locals():
            calibration.cleanup()


if __name__ == "__main__":
    main()
