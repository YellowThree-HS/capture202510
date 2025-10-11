import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation
import time
from lib.camera import Camera
from lib.robot import Robot
# ------------------- 1. 配置参数 -------------------

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
    if not robot or not robot.enabled:
        print("错误: 必须传入一个已连接并启用的 Robot 对象。")
        return None
    pose = robot.get_pose()
    return pose


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
def perform_calibration():
    """
    加载采集的数据并执行手眼标定计算。
    """
    print("\n" + "="*50)
    print("开始执行手眼标定计算...")
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

    # 为OpenCV函数准备R和t的列表
    R_end_to_base_list = []
    t_end_to_base_list = []
    R_board_to_cam_list = []
    t_board_to_cam_list = []

    for T_base_to_end, T_cam_to_board in zip(base_to_end_transforms, cam_to_board_transforms):
        # cv2.calibrateHandEye 需要的 A 是 end -> base 的变换
        # 我们有的是 base -> end, 所以需要求逆
        T_end_to_base = np.linalg.inv(T_base_to_end)
        R_end_to_base_list.append(T_end_to_base[:3, :3])
        t_end_to_base_list.append(T_end_to_base[:3, 3])

        # cv2.calibrateHandEye 需要的 B 是 board -> cam 的变换
        # 我们有的是 cam -> board, 所以需要求逆
        T_board_to_cam = np.linalg.inv(T_cam_to_board)
        R_board_to_cam_list.append(T_board_to_cam[:3, :3])
        t_board_to_cam_list.append(T_board_to_cam[:3, 3])


    R_end_to_cam, t_end_to_cam = cv2.calibrateHandEye(
        R_gripper2base=R_end_to_base_list,
        t_gripper2base=t_end_to_base_list,
        R_target2cam=R_board_to_cam_list,
        t_target2cam=t_board_to_cam_list,
        method=cv2.CALIB_HAND_EYE_PARK  # PARK方法通常很稳定
    )

    # 我们得到的是 end -> cam 的变换 T
    T_end_to_cam = np.eye(4)
    T_end_to_cam[:3, :3] = R_end_to_cam
    T_end_to_cam[:3, 3] = t_end_to_cam.flatten()

    print("Re-calculating A and B matrices for calibration...")
    R_A_list, t_A_list = [], []
    R_B_list, t_B_list = [], []

    for i in range(len(base_to_end_transforms) - 1):
        # A: Robot movement from pose i to i+1
        T_base_to_end_i = base_to_end_transforms[i]
        T_base_to_end_j = base_to_end_transforms[i+1]
        T_A = np.linalg.inv(T_base_to_end_i) @ T_base_to_end_j
        
        R_A_list.append(T_A[:3, :3])
        t_A_list.append(T_A[:3, 3])

        # B: Board movement from pose i to i+1 in camera frame
        T_cam_to_board_i = cam_to_board_transforms[i]
        T_cam_to_board_j = cam_to_board_transforms[i+1]
        T_B = np.linalg.inv(T_cam_to_board_i) @ T_cam_to_board_j

        R_B_list.append(T_B[:3, :3])
        t_B_list.append(T_B[:3, 3])

    # For Eye-to-Hand (`CALIB_HAND_EYE_TSAI`, etc.), the function solves AX=XB where X is `end_to_cam`.
    # This is not what we want. We need `base_to_cam`.
    # Let's use the dual calibration form: `calibrateHandEye(A, B)` finds X in `AX=XB`,
    # then `calibrateHandEye(B, A)` finds Y in `BY=YA`.
    # So we swap the arguments to solve for `base_to_camera_T`.
    R_base_cam, t_base_cam = cv2.calibrateHandEye(
        R_gripper2base=R_A_list, # This is actually R_B for our case
        t_gripper2base=t_A_list, # This is actually t_B for our case
        R_target2cam=R_B_list,   # This is actually R_A for our case
        t_target2cam=t_B_list,   # This is actually t_A for our case
        method=cv2.CALIB_HAND_EYE_DANIILIDIS # A more modern and accurate method
    )

    # Combine into the final 4x4 transformation matrix
    base_to_camera_T = np.eye(4)
    base_to_camera_T[:3, :3] = R_base_cam
    base_to_camera_T[:3, 3] = t_base_cam.flatten()


    print("\n--- 手眼标定结果 (眼在手外) ---")
    print("相机坐标系(Camera)相对于机器人基坐标系(Base)的变换矩阵 (base_to_camera_T):")
    np.set_printoptions(suppress=True, precision=4)
    print(base_to_camera_T)

    result_path = os.path.join(OUTPUT_DIR, "hand_eye_calibration_result.txt")
    np.savetxt(result_path, base_to_camera_T, fmt="%.6f")
    print(f"\n结果已保存至: {result_path}")
    
    # 验证
    # T_end_to_board 是固定的，也可以计算出来
    # T_end_to_board = inv(T_base_to_end_i) * T_base_to_cam * T_cam_to_board_i
    # 计算多组 T_end_to_board，看其是否稳定
    print("\n--- 结果验证 ---")
    print("正在计算 T_end_to_board (末端到标定板) 的变换矩阵...")
    print("理论上，这个矩阵对于所有采集点都应接近恒定。")
    end_to_board_transforms = []
    inv_base_to_cam = np.linalg.inv(base_to_camera_T)
    for T_base_to_end, T_cam_to_board in zip(base_to_end_transforms, cam_to_board_transforms):
        T_end_to_base = np.linalg.inv(T_base_to_end)
        T_end_to_board = T_end_to_base @ base_to_camera_T @ T_cam_to_board
        end_to_board_transforms.append(T_end_to_board)

    if len(end_to_board_transforms) > 1:
        translations = np.array([T[:3, 3] for T in end_to_board_transforms])
        mean_translation = np.mean(translations, axis=0)
        std_translation = np.std(translations, axis=0)
        print(f"平移向量 (x, y, z) 均值: {mean_translation} mm")
        print(f"平移向量 (x, y, z) 标准差: {std_translation} mm  <-- (这个值越小，说明标定结果越一致和准确)")


# ------------------- 主函数 -------------------

if __name__ == "__main__":
    
    try:
        # ---- 初始化相机 ----
        print("正在初始化RealSense相机...")
        # 假设你已经将 `Camera` 类代码放在了名为 realsense_camera.py 的文件中
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
        
        # ---- 用户选择 ----
        print("\n手眼标定程序 (眼在手外)")
        print("1. 采集数据")
        print("2. 执行标定")
        print("3. (预留) 验证模式")
        choice = input("请选择要执行的步骤 (1 或 2): ")

        if choice == '1':
            collect_calibration_data()
        elif choice == '2':
            perform_calibration()
        else:
            print("无效的选择。")

    except Exception as e:
        print(f"\n程序发生错误: {e}")
    finally:
        if cam:
            cam.release()
        print("程序退出。")


