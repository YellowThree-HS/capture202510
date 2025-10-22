#!/usr/bin/env python3
"""
灵巧手指尖相对于机械臂末端位置标定脚本 (ArUco版本)

基于aruco_verification.py的ArUco检测逻辑
标定流程：
1. 已知eye2end的4x4矩阵（手眼标定结果）
2. 移动机械臂到初始位置，按下空格拍照片
3. 识别ArUco标定板位置，得到obj2eye坐标
4. 通过get_pose_matrix得到end2base坐标
5. 计算obj2base = end2base @ eye2end @ obj2eye
6. 按'd'键启动拖拽模式，手动将机械臂移动到标定板中心
7. 按下'c'键确认位置，读取新的end2base'矩阵
8. 计算hand2end = obj2base @ inv(end2base')
9. 保存hand2end数据到txt文件
10. 最后取平均值

注意：这里标定的是灵巧手指尖相对于机械臂末端的固定偏移关系

使用方法：
- 空格键：拍照并检测ArUco标定板
- 'd'键：启动拖拽模式（手动移动机械臂）
- 'c'键：确认灵巧手位置并记录数据
- 'r'键：重置当前标定
- 'q'键：退出并计算平均值
- ESC键：退出程序
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from lib.camera import Camera
from lib.dobot import DobotRobot
# from lib.inspire_hand import InspireHand  # 不需要灵巧手


class HandFingertipCalibrationArUco:
    def __init__(self, 
                 robot_ip='192.168.5.2'):
        """
        初始化灵巧手指尖标定系统
        
        参数:
            robot_ip: 机械臂IP地址
        """
        self.robot_ip = robot_ip
        
        # 标定数据存储
        self.calibration_data = []
        self.current_obj2base = None
        self.current_end2base = None
        
        # 初始化系统组件
        self.camera = None
        self.robot = None
        
        # 手眼标定矩阵 (eye2hand)
        self.hand_eye_matrix = np.array([
            [ 0.01230037,  0.99763761,  0.06758625,  0.08419052],
            [-0.99992251,  0.01240196, -0.00108365,  0.00995925],
            [-0.00191929, -0.06756769,  0.99771285, -0.15882536],
            [ 0.0,         0.0,         0.0,         1.0        ]
        ])
        
        # ArUco参数设置 (与aruco_verification.py保持一致)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # ArUco板子参数 (8x8cm)
        self.marker_size = 0.08  # 8cm = 0.08m
        self.board_size = (4, 4)  # 4x4的ArUco板子
        
        # 创建ArUco板子
        self.create_aruco_board()
        
        # 相机内参
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 初始化所有系统
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
    
    def initialize_systems(self):
        """初始化相机、机械臂和灵巧手系统"""
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
            
            # 注意：此标定不需要初始化灵巧手，只需要机械臂
            
            # 显示手眼标定矩阵信息
            self.display_hand_eye_matrix()
            
            print("\n" + "="*60)
            print("系统初始化完成！")
            print("="*60)
            print("操作说明：")
            print("  空格键：拍照并检测ArUco标定板")
            print("  'd'键：启动拖拽模式（手动移动机械臂）")
            print("  'c'键：确认灵巧手位置并记录数据")
            print("  'r'键：重置当前标定")
            print("  'q'键：退出并计算平均值")
            print("  ESC键：退出程序")
            print("="*60)
            
        except Exception as e:
            print(f"系统初始化失败: {e}")
            raise
    
    def display_hand_eye_matrix(self):
        """显示手眼标定矩阵信息"""
        print(f"✓ 手眼标定矩阵已设置 (eye2hand)")
        print(f"  矩阵:\n{self.hand_eye_matrix}")
        
        # 显示位置信息
        position = self.hand_eye_matrix[:3, 3]
        print(f"  位置 (m): [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]")
        print(f"  位置 (mm): [{position[0]*1000:.3f}, {position[1]*1000:.3f}, {position[2]*1000:.3f}]")
    
    def detect_aruco_pose(self, image):
        """
        检测ArUco板子的位姿 (与aruco_verification.py保持一致)
        
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
        """在图像上绘制ArUco检测结果 (与aruco_verification.py保持一致)"""
        # 绘制检测到的标记
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
        # 绘制坐标轴
        if rvec is not None and tvec is not None:
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)
        
        return image
    
    def calculate_obj2base(self, rvec, tvec):
        """
        计算ArUco标定板相对于机械臂基座的位姿
        
        参数:
            rvec: ArUco板相对于相机的旋转向量
            tvec: ArUco板相对于相机的平移向量
            
        返回:
            obj2base: 标定板相对于机械臂基座的变换矩阵
            end2base: 机械臂末端相对于基座的变换矩阵
        """
        # 获取当前机械臂末端位姿
        end2base = self.robot.get_pose_matrix()  # 4x4变换矩阵
        
        # ArUco板相对于相机的变换矩阵
        aruco_to_cam = np.eye(4)
        aruco_to_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
        aruco_to_cam[:3, 3] = tvec.flatten()
        
        # 计算ArUco板相对于基座的位姿
        # obj2base = end2base @ hand_eye_matrix @ aruco_to_cam
        obj2base = end2base @ self.hand_eye_matrix @ aruco_to_cam
        
        return obj2base, end2base
    
    def calculate_hand2end(self, obj2base, end2base_new):
        """
        计算灵巧手指尖相对于机械臂末端的位姿
        
        参数:
            obj2base: 标定板相对于基座的位姿
            end2base_new: 新的机械臂末端相对于基座的位姿
            
        返回:
            hand2end: 灵巧手指尖相对于机械臂末端的变换矩阵
        """
        # hand2end = obj2base @ inv(end2base_new)
        hand2end = obj2base @ np.linalg.inv(end2base_new)
        return hand2end
    
    def save_calibration_data(self, hand2end_matrix):
        """保存标定数据到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存到列表
        self.calibration_data.append(hand2end_matrix.copy())
        
        # 保存到文件
        filename = f"hand_fingertip_calibration_data_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
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
        with open(result_filename, 'w', encoding='utf-8') as f:
            f.write("灵巧手指尖标定结果 (ArUco版本)\n")
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
    
    def draw_calibration_info(self, image, corners=None, ids=None, rvec=None, tvec=None):
        """在图像上绘制标定信息"""
        # 绘制ArUco检测结果
        if corners is not None and ids is not None:
            image = self.draw_aruco_detection(image, corners, ids, rvec, tvec)
        
        # 绘制状态信息
        status_text = [
            f"标定数据组数: {len(self.calibration_data)}",
            f"当前状态: {'等待拍照' if self.current_obj2base is None else '等待确认灵巧手位置'}",
            "",
            "操作说明:",
            "空格键: 拍照检测ArUco标定板",
            "d键: 启动拖拽模式",
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
        print("开始灵巧手指尖标定 (ArUco版本)...")
        
        cv2.namedWindow('Hand Fingertip Calibration (ArUco)', cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                # 获取实时图像
                color_image, _ = self.camera.get_frames()
                if color_image is None:
                    continue
                
                # 检测ArUco板子
                success, rvec, tvec, corners, ids = self.detect_aruco_pose(color_image)
                
                # 绘制标定信息
                display_image = self.draw_calibration_info(color_image, corners, ids, rvec, tvec)
                
                # 显示图像
                cv2.imshow('Hand Fingertip Calibration (ArUco)', display_image)
                
                # 等待按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # 空格键 - 拍照检测ArUco标定板
                    if success:
                        print("\n" + "="*50)
                        print("检测到ArUco板子，计算位姿...")
                        
                        # 计算标定板相对于基座的位姿
                        obj2base, end2base = self.calculate_obj2base(rvec, tvec)
                        self.current_obj2base = obj2base
                        self.current_end2base = end2base
                        
                        print(f"obj2base: {obj2base}")
                        print(f"end2base: {end2base}")
                        print("请将灵巧手移动到标定板中心，然后按'c'键确认")
                        print("="*50)
                        
                        # 显示检测结果
                        display_image = self.draw_calibration_info(color_image, corners, ids, rvec, tvec)
                        cv2.imshow('Hand Fingertip Calibration (ArUco)', display_image)
                        cv2.waitKey(1000)  # 显示1秒
                    else:
                        print("❌ 未检测到ArUco板子，请确保板子在相机视野内")
                
                elif key == ord('d'):  # 'd'键 - 启动拖拽模式
                    print("\n启动拖拽模式...")
                    try:
                        self.robot.r_inter.StartDrag()
                        print("✓ 拖拽模式已启动，现在可以手动移动机械臂")
                        print("  将灵巧手移动到标定板中心，然后按'c'键确认位置")
                    except Exception as e:
                        print(f"✗ 启动拖拽模式失败: {e}")
                
                elif key == ord('c'):  # 'c'键 - 确认灵巧手位置
                    if self.current_obj2base is None:
                        print("请先按空格键检测ArUco标定板")
                        continue
                    
                    print("\n正在记录机械臂位置...")
                    
                    # 停止拖拽模式
                    try:
                        self.robot.r_inter.StopDrag()
                        print("✓ 已停止拖拽模式")
                    except:
                        pass
                    
                    # 获取当前机械臂位姿（此时机械臂末端在标定板中心）
                    end2base_new = self.robot.get_pose_matrix()
                    print(f"end2base_new: {end2base_new}")
                    # 计算灵巧手指尖相对于机械臂末端的位姿
                    hand2end = self.calculate_hand2end(self.current_obj2base, end2base_new)
                    print(f"hand2end: {hand2end}")
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
        calibration = HandFingertipCalibrationArUco(
            robot_ip='192.168.5.2'
        )
        
        # 运行标定
        calibration.run_calibration()
        
    except Exception as e:
        print(f"程序运行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'calibration' in locals():
            calibration.cleanup()


if __name__ == "__main__":
    main()
