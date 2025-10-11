"""
自动化手眼标定数据采集脚本
自动移动机器人到不同位姿并采集标定数据

注意：此脚本会自动控制机器人移动，使用前请确保安全！
"""

import cv2
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation

from lib.camera import Camera
from lib.robot import Robot

# ------------------- 配置参数 -------------------
ROBOT_IP = "192.168.5.1"
CALIBRATION_MODE = "eye_in_hand"  # "eye_to_hand" 或 "eye_in_hand"

# 棋盘格参数
CHESSBOARD_CORNERS_NUM_X = 9
CHESSBOARD_CORNERS_NUM_Y = 6
CHESSBOARD_SQUARE_SIZE_MM = 20

# 数据存储路径
OUTPUT_DIR = "hand_eye_calibration_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 自动采集的位姿列表（根据你的机器人工作空间调整）
AUTO_CALIBRATION_POSES = [
    {"x": 57.471699, "y": -344.245819, "z": 532.832581, "rx": 165.895203, "ry": 9.106956, "rz": 70.128647},
    {"x": -90.228745, "y": -337.145294, "z": 532.832581, "rx": 165.895203, "ry": 9.106956, "rz": 45.667847}
    # {"x": 650, "y": -300, "z": 350, "rx": 160, "ry": 20, "rz": 150},
    # {"x": 600, "y": -150, "z": 420, "rx": 175, "ry": 5, "rz": 120},
    # {"x": 500, "y": -250, "z": 360, "rx": 170, "ry": 15, "rz": 145},
    # {"x": 700, "y": -280, "z": 390, "rx": 165, "ry": 10, "rz": 135},
    # {"x": 580, "y": -220, "z": 410, "rx": 180, "ry": 0, "rz": 140},
    # {"x": 620, "y": -270, "z": 370, "rx": 170, "ry": 18, "rz": 138},
    # {"x": 550, "y": -300, "z": 400, "rx": 160, "ry": 20, "rz": 150},
    # {"x": 650, "y": -200, "z": 380, "rx": 175, "ry": 8, "rz": 125},
]

# 相机和机器人对象
cam = None
robot = None
CAMERA_MATRIX = None
DISTORTION_COEFFS = None


def find_chessboard_pose(image, obj_points):
    """在图像中寻找棋盘格"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(
        gray,
        (CHESSBOARD_CORNERS_NUM_X, CHESSBOARD_CORNERS_NUM_Y),
        flags=flags
    )
    
    vis_image = image.copy()
    
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        _, rvec, tvec = cv2.solvePnP(obj_points, corners_subpix, CAMERA_MATRIX, DISTORTION_COEFFS)
        cv2.drawChessboardCorners(vis_image, (CHESSBOARD_CORNERS_NUM_X, CHESSBOARD_CORNERS_NUM_Y), corners_subpix, ret)
        return rvec, tvec, vis_image, True
    else:
        return None, None, vis_image, False


def rtvec_to_matrix(rvec, tvec):
    """将旋转向量和平移向量转换为4x4齐次变换矩阵"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def wait_for_stable_detection(obj_points, timeout=10, stable_count=5):
    """
    等待棋盘格检测稳定
    
    Args:
        obj_points: 棋盘格3D点
        timeout: 超时时间（秒）
        stable_count: 需要连续检测成功的次数
        
    Returns:
        (rvec, tvec, success)
    """
    print("  等待棋盘格检测稳定...")
    start_time = time.time()
    success_count = 0
    last_rvec, last_tvec = None, None
    
    while time.time() - start_time < timeout:
        color_image = cam.get_color_image()
        if color_image is None:
            time.sleep(0.1)
            continue
        
        rvec, tvec, vis_image, found = find_chessboard_pose(color_image, obj_points)
        
        # 显示检测结果
        status_text = f"Detecting... ({success_count}/{stable_count})"
        if found:
            cv2.putText(vis_image, "Chessboard Detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(vis_image, "Searching for chessboard...", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_image, status_text, (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("Auto Calibration", vis_image)
        cv2.waitKey(1)
        
        if found:
            success_count += 1
            last_rvec, last_tvec = rvec, tvec
            if success_count >= stable_count:
                print("  ✓ 棋盘格检测稳定")
                return last_rvec, last_tvec, True
        else:
            success_count = 0
        
        time.sleep(0.1)
    
    print("  ✗ 检测超时")
    return None, None, False


def auto_collect_calibration_data():
    """自动采集标定数据"""
    global robot, cam, CAMERA_MATRIX, DISTORTION_COEFFS
    
    if not robot or not robot.enabled:
        print("错误: 机器人未连接或未启用")
        return False
    if not cam or CAMERA_MATRIX is None:
        print("错误: 相机未初始化")
        return False
    
    print("\n" + "="*60)
    print("自动手眼标定数据采集")
    print(f"模式: {CALIBRATION_MODE}")
    print(f"计划采集位姿数量: {len(AUTO_CALIBRATION_POSES)}")
    print("="*60)
    
    base_to_end_transforms = []
    cam_to_board_transforms = []
    
    # 定义棋盘格3D点
    obj_points = np.zeros((CHESSBOARD_CORNERS_NUM_X * CHESSBOARD_CORNERS_NUM_Y, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS_NUM_X, 0:CHESSBOARD_CORNERS_NUM_Y].T.reshape(-1, 2)
    obj_points *= CHESSBOARD_SQUARE_SIZE_MM
    
    successful_poses = 0
    
    for i, pose in enumerate(AUTO_CALIBRATION_POSES):
        print(f"\n--- 位姿 {i+1}/{len(AUTO_CALIBRATION_POSES)} ---")
        print(f"目标: X={pose['x']}, Y={pose['y']}, Z={pose['z']}, "
              f"Rx={pose['rx']}, Ry={pose['ry']}, Rz={pose['rz']}")
        
        # 移动机器人
        print("正在移动机器人...")
        success = robot.move_to(
            pose['x'], pose['y'], pose['z'],
            pose['rx'], pose['ry'], pose['rz'],
            mode='joint'
        )
        
        if not success:
            print(f"✗ 移动到位姿 {i+1} 失败，跳过")
            continue
        
        # 等待机器人稳定
        print("等待机器人稳定...")
        time.sleep(3)
        
        # 等待棋盘格检测稳定
        rvec, tvec, found = wait_for_stable_detection(obj_points, timeout=15, stable_count=5)
        
        if not found:
            print(f"✗ 位姿 {i+1} 无法检测到棋盘格，跳过")
            continue
        
        # 获取机器人位姿
        base_to_end_T = robot.get_pose()
        if base_to_end_T is None:
            print(f"✗ 无法获取机器人位姿，跳过")
            continue
        
        # 计算标定板在相机下的位姿
        cam_to_board_T = rtvec_to_matrix(rvec, tvec)
        
        # 保存数据
        base_to_end_transforms.append(base_to_end_T)
        cam_to_board_transforms.append(cam_to_board_T)
        
        # 保存图像
        color_image = cam.get_color_image()
        img_save_path = os.path.join(OUTPUT_DIR, f"auto_capture_{successful_poses + 1}.png")
        cv2.imwrite(img_save_path, color_image)
        
        successful_poses += 1
        print(f"✓ 位姿 {i+1} 采集成功！")
        
        time.sleep(1)
    
    cv2.destroyAllWindows()
    
    # 保存数据
    if successful_poses >= 3:
        np.save(os.path.join(OUTPUT_DIR, "base_to_end_transforms.npy"), 
                np.array(base_to_end_transforms))
        np.save(os.path.join(OUTPUT_DIR, "cam_to_board_transforms.npy"),
                np.array(cam_to_board_transforms))
        
        print(f"\n✓ 自动采集完成！成功采集 {successful_poses} 组数据")
        print(f"数据已保存到: {OUTPUT_DIR}")
        return True
    else:
        print(f"\n✗ 采集失败，只获得 {successful_poses} 组有效数据（至少需要3组）")
        return False


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
            
            # 安全确认
            print("\n" + "!"*60)
            print("警告: 机器人将自动移动到预设位姿！")
            print("请确保:")
            print("  1. 工作空间内无障碍物")
            print("  2. 棋盘格已放置在合适位置")
            print("  3. 相机视野正常")
            print("!"*60)
            
            response = input("\n确认开始自动采集？(yes/no): ").strip().lower()
            if response != 'yes':
                print("已取消操作")
                return
            
            # 开始自动采集
            success = auto_collect_calibration_data()
            
            if success:
                print("\n✓ 数据采集完成！")
                print("下一步: 运行 calibration_eye_hand.py 选择选项2进行标定计算")
            
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
