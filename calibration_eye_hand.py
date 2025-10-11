import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation
import time
from lib.camera import Camera
from lib.robot import Robot
# ------------------- 1. 配置参数 -------------------

# 手眼标定模式
CALIBRATION_MODE = "eye_to_hand"  # "eye_to_hand" 或 "eye_in_hand"

# 棋盘格标定板参数
CHESSBOARD_CORNERS_NUM_X = 9  # 棋盘格内部角点的列数
CHESSBOARD_CORNERS_NUM_Y = 6  # 棋盘格内部角点的行数
CHESSBOARD_SQUARE_SIZE_MM = 20  # 棋盘格方块的物理边长（单位：毫米）

# 采集数据点的数量
NUM_CALIBRATION_POSES = 15

# 数据存储路径
OUTPUT_DIR = "hand_eye_calibration_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 全局相机对象
cam = None
robot = None

# 【自动获取】相机内参矩阵和畸变系数
# 这些值将在相机初始化后自动填充
CAMERA_MATRIX = None
DISTORTION_COEFFS = None


# ------------------- 2. 模拟/替换的SDK接口 -------------------
# !!! 重点: 你需要用你真实的机器人SDK替换这个模拟函数 !!!

def get_robot_pose_from_sdk(robot):
    """
    从机器人获取当前末端执行器的位姿矩阵
    返回: 4x4 齐次变换矩阵 (Base -> End-Effector)
    """
    if not robot or not robot.connected:
        print("错误: 机器人未连接。")
        return None
    
    try:
        pose_matrix = robot.get_pose()
        if pose_matrix is None:
            print("警告: 无法获取机器人位姿")
            return None
        
        # 确保返回的是4x4矩阵
        if pose_matrix.shape != (4, 4):
            print(f"错误: 期望4x4矩阵，但得到{pose_matrix.shape}")
            return None
            
        return pose_matrix
        
    except Exception as e:
        print(f"获取机器人位姿时出错: {e}")
        return None


# ------------------- 3. 辅助函数 -------------------

def find_chessboard_pose(image, obj_points):
    """
    在图像中寻找棋盘格，并计算其在相机坐标系下的位姿。
    返回 旋转向量(rvec)、平移向量(tvec) 和 绘制了角点的图像。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNERS_NUM_X, CHESSBOARD_CORNERS_NUM_Y), None)
    
    vis_image = image.copy()

    if ret:
        # 亚像素级精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 求解PnP问题，得到标定板在相机坐标系下的姿态
        # rvec是旋转向量，tvec是平移向量
        _, rvec, tvec = cv2.solvePnP(obj_points, corners_subpix, CAMERA_MATRIX, DISTORTION_COEFFS)
        
        # 可视化
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

# ------------------- 4. 主流程: 数据采集 -------------------

def collect_calibration_data():
    """
    主循环，用于采集机器人位姿和对应的相机中标定板位姿数据。
    """

    global robot, cam, CAMERA_MATRIX, DISTORTION_COEFFS
    
    # 检查设备是否已初始化
    if not robot or not robot.enabled:
        print("错误: 机器人未连接或未启用。请重启程序。")
        return
    if not cam or CAMERA_MATRIX is None:
        print("错误: 相机未初始化或内参缺失。请重启程序。")
        return
    
    print("\n" + "="*50)
    print("开始采集手眼标定数据...")
    print(f"目标采集数量: {NUM_CALIBRATION_POSES}")
    print("操作指南:")
    print("  - 移动机器人，确保棋盘格在相机视野内清晰可见。")
    print("  - 当角点被稳定检测到（绿色连线），按下 'c' 键采集当前数据。")
    print("  - 按下 'q' 键退出采集。")
    print("="*50)

    base_to_end_transforms = []
    cam_to_board_transforms = []
    
    # 定义棋盘格在自己坐标系下的三维点坐标
    obj_points = np.zeros((CHESSBOARD_CORNERS_NUM_X * CHESSBOARD_CORNERS_NUM_Y, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS_NUM_X, 0:CHESSBOARD_CORNERS_NUM_Y].T.reshape(-1, 2)
    obj_points *= CHESSBOARD_SQUARE_SIZE_MM

    successful_poses = 0
    while successful_poses < NUM_CALIBRATION_POSES:
        # 从相机获取图像
        color_image = cam.get_color_image()
        if color_image is None:
            time.sleep(0.1)
            continue
        
        # 查找棋盘格
        rvec, tvec, vis_image, found = find_chessboard_pose(color_image, obj_points)
        
        # 在预览窗口上显示信息
        status_text = f"Collected: {successful_poses}/{NUM_CALIBRATION_POSES}"
        if found:
            cv2.putText(vis_image, "Ready to capture. Press 'c'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(vis_image, "Chessboard not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_image, status_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(vis_image, "Press 'q' to quit", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Hand-Eye Calibration - Live View", vis_image)
        key = cv2.waitKey(1) & 0xFF

        # 按 'c' 采集数据
        if key == ord('c') and found:
            print(f"\n--- 正在采集第 {successful_poses + 1}/{NUM_CALIBRATION_POSES} 组数据 ---")

            # 1. 获取机器人末端位姿 (Base -> End-Effector)
            base_to_end_T = get_robot_pose_from_sdk(robot)
            
            # 2. 获取标定板在相机下的位姿 (Camera -> Board)
            cam_to_board_T = rtvec_to_matrix(rvec, tvec)
            
            # 3. 保存数据
            base_to_end_transforms.append(base_to_end_T)
            cam_to_board_transforms.append(cam_to_board_T)

            # 保存用于检查的图像
            img_save_path = os.path.join(OUTPUT_DIR, f"capture_{successful_poses + 1}.png")
            cv2.imwrite(img_save_path, vis_image)
            
            print(f"第 {successful_poses + 1} 组有效数据采集成功！图像已保存至 {img_save_path}")
            successful_poses += 1
        
        # 按 'q' 退出
        elif key == ord('q'):
            print("用户中断采集。")
            break

    cv2.destroyAllWindows()

    if len(base_to_end_transforms) > 0:
        # 保存采集到的变换矩阵
        np.save(os.path.join(OUTPUT_DIR, "base_to_end_transforms.npy"), np.array(base_to_end_transforms))
        np.save(os.path.join(OUTPUT_DIR, "cam_to_board_transforms.npy"), np.array(cam_to_board_transforms))
        print("\n数据采集完成！变换矩阵已保存。")
    else:
        print("\n未采集任何数据。")

# ------------------- 5. 主流程: 标定计算 -------------------
def perform_calibration_eye_to_hand():
    """
    眼在手外 (Eye-to-Hand) 标定计算
    求解相机坐标系相对于机器人基坐标系的变换矩阵
    """
    print("\n" + "="*50)
    print("开始执行手眼标定计算 (眼在手外)...")
    print("="*50)

    try:
        base_to_end_transforms = np.load(os.path.join(OUTPUT_DIR, "base_to_end_transforms.npy"))
        cam_to_board_transforms = np.load(os.path.join(OUTPUT_DIR, "cam_to_board_transforms.npy"))
    except FileNotFoundError:
        print("错误: 未找到标定数据文件。请先运行数据采集流程。")
        return

    num_poses = len(base_to_end_transforms)
    if num_poses < 3:
        print(f"错误: 标定需要至少3组有效数据，当前只有 {num_poses} 组。")
        return
    
    print(f"加载了 {num_poses} 组标定数据进行计算。")

    # 眼在手外: AX = XB
    # A: 机器人运动 (gripper2base)
    # B: 标定板在相机坐标系下的运动 (target2cam)
    # X: 相机相对于机器人基坐标系的变换 (cam2base)
    
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for i in range(num_poses - 1):
        # A: 机器人从位姿i到位姿i+1的运动
        T_base_to_end_i = base_to_end_transforms[i]
        T_base_to_end_j = base_to_end_transforms[i + 1]
        T_A = np.linalg.inv(T_base_to_end_i) @ T_base_to_end_j  # end_i -> end_j
        
        R_gripper2base.append(T_A[:3, :3])
        t_gripper2base.append(T_A[:3, 3])

        # B: 标定板从位姿i到位姿i+1在相机坐标系下的运动
        T_cam_to_board_i = cam_to_board_transforms[i]
        T_cam_to_board_j = cam_to_board_transforms[i + 1]
        T_B = T_cam_to_board_i @ np.linalg.inv(T_cam_to_board_j)  # board_j -> board_i (in cam frame)
        
        R_target2cam.append(T_B[:3, :3])
        t_target2cam.append(T_B[:3, 3])

    # 使用OpenCV求解 AX = XB
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # 构建4x4变换矩阵
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()

    print("\n--- 手眼标定结果 (眼在手外) ---")
    print("相机坐标系相对于机器人基坐标系的变换矩阵 (T_cam2base):")
    np.set_printoptions(suppress=True, precision=6)
    print(T_cam2base)

    # 保存结果
    result_path = os.path.join(OUTPUT_DIR, "eye_to_hand_calibration_result.npy")
    np.save(result_path, T_cam2base)
    
    result_txt_path = os.path.join(OUTPUT_DIR, "eye_to_hand_calibration_result.txt")
    np.savetxt(result_txt_path, T_cam2base, fmt="%.6f")
    print(f"\n结果已保存至: {result_path} 和 {result_txt_path}")
    
    # 验证标定精度
    verify_eye_to_hand_calibration(base_to_end_transforms, cam_to_board_transforms, T_cam2base)
    
    return T_cam2base

def perform_calibration_eye_in_hand():
    """
    眼在手上 (Eye-in-Hand) 标定计算
    求解相机坐标系相对于机器人末端的变换矩阵
    """
    print("\n" + "="*50)
    print("开始执行手眼标定计算 (眼在手上)...")
    print("="*50)

    try:
        base_to_end_transforms = np.load(os.path.join(OUTPUT_DIR, "base_to_end_transforms.npy"))
        cam_to_board_transforms = np.load(os.path.join(OUTPUT_DIR, "cam_to_board_transforms.npy"))
    except FileNotFoundError:
        print("错误: 未找到标定数据文件。请先运行数据采集流程。")
        return

    num_poses = len(base_to_end_transforms)
    if num_poses < 3:
        print(f"错误: 标定需要至少3组有效数据，当前只有 {num_poses} 组。")
        return
    
    print(f"加载了 {num_poses} 组标定数据进行计算。")

    # 眼在手上: AX = ZB
    # A: 机器人运动 (gripper2base)
    # B: 标定板相对于基坐标系的运动 (target2base) 
    # X: 相机相对于机器人末端的变换 (cam2gripper)
    # Z: 基坐标系到标定板的固定变换 (base2target)
    
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for i in range(num_poses - 1):
        # A: 机器人从位姿i到位姿i+1的运动
        T_base_to_end_i = base_to_end_transforms[i]
        T_base_to_end_j = base_to_end_transforms[i + 1]
        T_A = np.linalg.inv(T_base_to_end_i) @ T_base_to_end_j
        
        R_gripper2base.append(T_A[:3, :3])
        t_gripper2base.append(T_A[:3, 3])

        # B: 标定板从位姿i到位姿i+1的运动 (在相机坐标系中观察到的)
        T_cam_to_board_i = cam_to_board_transforms[i]
        T_cam_to_board_j = cam_to_board_transforms[i + 1]
        T_B = np.linalg.inv(T_cam_to_board_i) @ T_cam_to_board_j
        
        R_target2cam.append(T_B[:3, :3])
        t_target2cam.append(T_B[:3, 3])

    # 使用OpenCV求解
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_PARK
    )

    # 构建4x4变换矩阵
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

    print("\n--- 手眼标定结果 (眼在手上) ---")
    print("相机坐标系相对于机器人末端的变换矩阵 (T_cam2gripper):")
    np.set_printoptions(suppress=True, precision=6)
    print(T_cam2gripper)

    # 保存结果
    result_path = os.path.join(OUTPUT_DIR, "eye_in_hand_calibration_result.npy")
    np.save(result_path, T_cam2gripper)
    
    result_txt_path = os.path.join(OUTPUT_DIR, "eye_in_hand_calibration_result.txt")
    np.savetxt(result_txt_path, T_cam2gripper, fmt="%.6f")
    print(f"\n结果已保存至: {result_path} 和 {result_txt_path}")
    
    # 验证标定精度
    verify_eye_in_hand_calibration(base_to_end_transforms, cam_to_board_transforms, T_cam2gripper)
    
    return T_cam2gripper

def verify_eye_to_hand_calibration(base_to_end_transforms, cam_to_board_transforms, T_cam2base):
    """验证眼在手外标定的精度"""
    print("\n--- 标定精度验证 (眼在手外) ---")
    
    # 对于眼在手外，标定板在基坐标系中的位置应该是一致的
    board_positions_in_base = []
    
    for i, (T_base2end, T_cam2board) in enumerate(zip(base_to_end_transforms, cam_to_board_transforms)):
        # 计算标定板在基坐标系中的位置
        T_base2board = T_cam2base @ T_cam2board
        board_positions_in_base.append(T_base2board[:3, 3])
        
        if i < 5:  # 只打印前5个结果
            print(f"位姿 {i+1}: 标定板在基坐标系位置 = {T_base2board[:3, 3]:.3f}")
    
    # 计算位置的标准差
    positions = np.array(board_positions_in_base)
    mean_position = np.mean(positions, axis=0)
    std_position = np.std(positions, axis=0)
    
    print(f"\n标定板位置统计:")
    print(f"平均位置 (mm): {mean_position:.3f}")
    print(f"标准差 (mm): {std_position:.3f}")
    print(f"位置一致性误差: {np.linalg.norm(std_position):.3f} mm")
    
    if np.linalg.norm(std_position) < 5.0:
        print("✓ 标定精度良好 (误差 < 5mm)")
    elif np.linalg.norm(std_position) < 10.0:
        print("⚠ 标定精度一般 (误差 5-10mm)")
    else:
        print("✗ 标定精度较差 (误差 > 10mm)，建议重新采集数据")

def verify_eye_in_hand_calibration(base_to_end_transforms, cam_to_board_transforms, T_cam2gripper):
    """验证眼在手上标定的精度"""
    print("\n--- 标定精度验证 (眼在手上) ---")
    
    # 对于眼在手上，相机到标定板的变换应该通过末端位姿保持一致
    board_positions_in_gripper = []
    
    for i, (T_base2end, T_cam2board) in enumerate(zip(base_to_end_transforms, cam_to_board_transforms)):
        # 计算标定板在末端坐标系中的位置
        T_gripper2board = T_cam2gripper @ T_cam2board
        board_positions_in_gripper.append(T_gripper2board[:3, 3])
        
        if i < 5:  # 只打印前5个结果
            print(f"位姿 {i+1}: 标定板在末端坐标系位置 = {T_gripper2board[:3, 3]:.3f}")
    
    # 计算位置的标准差
    positions = np.array(board_positions_in_gripper)
    mean_position = np.mean(positions, axis=0)
    std_position = np.std(positions, axis=0)
    
    print(f"\n标定板相对末端位置统计:")
    print(f"平均位置 (mm): {mean_position:.3f}")
    print(f"标准差 (mm): {std_position:.3f}")
    print(f"位置一致性误差: {np.linalg.norm(std_position):.3f} mm")
    
    if np.linalg.norm(std_position) < 5.0:
        print("✓ 标定精度良好 (误差 < 5mm)")
    elif np.linalg.norm(std_position) < 10.0:
        print("⚠ 标定精度一般 (误差 5-10mm)")
    else:
        print("✗ 标定精度较差 (误差 > 10mm)，建议重新采集数据")

def perform_calibration():
    """
    根据配置的模式执行相应的标定计算
    """
    if CALIBRATION_MODE == "eye_to_hand":
        return perform_calibration_eye_to_hand()
    elif CALIBRATION_MODE == "eye_in_hand":
        return perform_calibration_eye_in_hand()
    else:
        print(f"错误: 未知的标定模式 '{CALIBRATION_MODE}'")
        print("支持的模式: 'eye_to_hand', 'eye_in_hand'")
        return None


def switch_calibration_mode():
    """切换标定模式"""
    global CALIBRATION_MODE
    
    print(f"\n当前标定模式: {CALIBRATION_MODE}")
    print("1. eye_to_hand (眼在手外) - 相机固定在环境中")
    print("2. eye_in_hand (眼在手上) - 相机固定在机器人末端")
    
    choice = input("选择新的标定模式 (1/2): ").strip()
    
    if choice == '1':
        CALIBRATION_MODE = "eye_to_hand"
        print("已切换到: 眼在手外模式")
    elif choice == '2':
        CALIBRATION_MODE = "eye_in_hand"
        print("已切换到: 眼在手上模式")
    else:
        print("无效选择，保持当前模式")

def validate_calibration_result():
    """验证已有的标定结果"""
    print("\n验证标定结果...")
    
    # 检查是否存在标定结果文件
    eye_to_hand_file = os.path.join(OUTPUT_DIR, "eye_to_hand_calibration_result.npy")
    eye_in_hand_file = os.path.join(OUTPUT_DIR, "eye_in_hand_calibration_result.npy")
    
    if os.path.exists(eye_to_hand_file):
        print("发现眼在手外标定结果")
        T_cam2base = np.load(eye_to_hand_file)
        print("T_cam2base (相机到基坐标系):")
        print(T_cam2base)
        
        # 验证精度
        try:
            base_to_end_transforms = np.load(os.path.join(OUTPUT_DIR, "base_to_end_transforms.npy"))
            cam_to_board_transforms = np.load(os.path.join(OUTPUT_DIR, "cam_to_board_transforms.npy"))
            verify_eye_to_hand_calibration(base_to_end_transforms, cam_to_board_transforms, T_cam2base)
        except FileNotFoundError:
            print("无法找到原始数据进行验证")
    
    if os.path.exists(eye_in_hand_file):
        print("\n发现眼在手上标定结果")
        T_cam2gripper = np.load(eye_in_hand_file)
        print("T_cam2gripper (相机到末端坐标系):")
        print(T_cam2gripper)
        
        # 验证精度
        try:
            base_to_end_transforms = np.load(os.path.join(OUTPUT_DIR, "base_to_end_transforms.npy"))
            cam_to_board_transforms = np.load(os.path.join(OUTPUT_DIR, "cam_to_board_transforms.npy"))
            verify_eye_in_hand_calibration(base_to_end_transforms, cam_to_board_transforms, T_cam2gripper)
        except FileNotFoundError:
            print("无法找到原始数据进行验证")
    
    if not os.path.exists(eye_to_hand_file) and not os.path.exists(eye_in_hand_file):
        print("未找到任何标定结果文件，请先执行标定。")


# ------------------- 主函数 -------------------

if __name__ == "__main__":
    
    try:
        # ---- 初始化相机 ----
        print("正在初始化RealSense相机...")
        cam = Camera(camera_model='d435') 
        print("相机连接成功。")

        print("正在连接机器人...")
        robot_ip = "192.168.1.6"
        robot = Robot(robot_ip)
        if robot.connect():
            try:
                robot.enable()
            except Exception as e:
                print(e,'connect fail')
            print("机器人连接并启用成功。")
        else:
            print("机器人连接失败，请检查连接。")
            robot = None
        
        # 自动获取相机内参
        CAMERA_MATRIX = cam.get_camera_matrix('color')
        DISTORTION_COEFFS = cam.get_distortion_coeffs('color')

        if CAMERA_MATRIX is None:
            raise RuntimeError("无法获取相机内参矩阵，请检查相机连接和配置。")

        print("相机初始化成功，内参已自动加载。")
        print("相机矩阵:\n", CAMERA_MATRIX)
        print("畸变系数:\n", DISTORTION_COEFFS)
        
        # ---- 主循环 ----
        while True:
            print(f"\n手眼标定程序 (当前模式: {CALIBRATION_MODE})")
            print("=" * 50)
            print("可用操作:")
            print("1. 采集数据")
            print("2. 执行标定")
            print("3. 切换标定模式")
            print("4. 验证标定结果")
            print("q. 退出程序")
            print("=" * 50)
            
            choice = input("请选择要执行的操作 (1/2/3/4/q): ").strip().lower()

            if choice == '1':
                print(f"\n开始数据采集 (模式: {CALIBRATION_MODE})")
                if CALIBRATION_MODE == "eye_in_hand":
                    print("注意: 眼在手上模式 - 相机固定在机器人末端")
                    print("移动机器人时，确保标定板始终在相机视野内")
                else:
                    print("注意: 眼在手外模式 - 相机固定在环境中")
                    print("移动机器人和标定板，确保标定板在相机视野内")
                collect_calibration_data()
                
            elif choice == '2':
                result = perform_calibration()
                if result is not None:
                    print(f"\n标定完成！模式: {CALIBRATION_MODE}")
                    
            elif choice == '3':
                switch_calibration_mode()
                
            elif choice == '4':
                validate_calibration_result()
                
            elif choice == 'q':
                print("退出程序。")
                break
                
            else:
                print("无效的选择，请重试。")

    except Exception as e:
        print(f"\n程序发生错误: {e}")
    finally:
        if cam:
            cam.release()
        print("程序退出。")
        if cam:
            cam.release()
        print("程序退出。")


