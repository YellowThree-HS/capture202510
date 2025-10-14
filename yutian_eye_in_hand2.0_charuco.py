import numpy as np
import cv2
import cv2.aruco as aruco
import os
import csv
# import pupil_apriltags as apriltag
import transforms3d as tfs
import math
import random
import time


from lib.dobot import DobotRobot
from lib.camera import Camera



def convert_euler_to_rotation_matrix(x, y, z, rx, ry, rz):
    """
    Convert Euler angles to a rotation matrix.
    Args:
    x, y, z: Translation coordinates.
    rx, ry, rz: Rotation angles in radians.

    Returns:
    A 4x4 rotation matrix.
    """
    rx = rx / math.pi * 180
    ry = ry / math.pi * 180
    rz = rz / math.pi * 180
    rmat = tfs.euler.euler2mat(math.radians(rx), math.radians(ry), math.radians(rz))
    rotation_matrix = tfs.affines.compose(np.squeeze(np.asarray((x, y, z))), rmat, [1, 1, 1])
    return rotation_matrix


def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles.
    Args:
    R: A 3x3 rotation matrix.

    Returns:
    A tuple of Euler angles (phi, theta, psi).
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def hand_eye_calibration(
    R_end2base, 
    t_end2base, 
    R_board2camera, 
    t_board2camera, 
    method=cv2.CALIB_HAND_EYE_PARK
):
    """M_board2base
    手眼标定封装函数
    
    参数:
        R_end2base (list): 机械臂末端到基座的旋转矩阵列表 [R1, R2, ...] (每个为3x3 np.array)
        t_end2base (list): 机械臂末端到基座的平移向量列表 [t1, t2, ...] (每个为3x1 np.array)
        R_board2camera (list): 标定板到相机的旋转矩阵列表
        t_board2camera (list): 标定板到相机的平移向量列表
        method: 标定方法 (默认PARK算法)
        M_board2base_list (list): 标定板到基座的齐次变换矩阵列表 (4x4 np.array)
        
    返回:
        calibrate_mean (np.array): 标定结果的均值 (4x4 齐次矩阵)
        calibrate_std (np.array): 标定结果的标准差 (4x4 齐次矩阵)
        M_cam2end (np.array): 相机到末端的齐次变换矩阵 (4x4)
    """
    # 1. 执行手眼标定
    R_cam2end, t_cam2end = cv2.calibrateHandEye(
        R_gripper2base=R_end2base,
        t_gripper2base=t_end2base,
        R_target2cam=R_board2camera,
        t_target2cam=t_board2camera,
        method=method
    )
    
    # 2. 构建相机到末端的齐次变换矩阵
    M_cam2end = np.eye(4)
    M_cam2end[:3, :3] = R_cam2end
    M_cam2end[:3, 3] = t_cam2end.reshape(3)

    M_end2base=np.eye(4)
    M_board2camera=np.eye(4)
    M_board2base=[]

    for sub_R_end2base,sub_t_end2base,sub_R_board2camera,sub_t_board2camera in zip(R_end2base, t_end2base, R_board2camera, t_board2camera):
        M_end2base[:3, :3] =sub_R_end2base
        M_end2base[:3, 3] =sub_t_end2base.reshape(3)
        M_board2camera[:3, :3] =sub_R_board2camera
        M_board2camera[:3, 3] =sub_t_board2camera.reshape(3)
        M_board2base.append(M_end2base@M_cam2end@M_board2camera)

    # 计算均值和标准差（注意：齐次矩阵的统计需分离旋转和平移）
    calibrate_mean = np.mean(M_board2base, axis=0)
    calibrate_std = np.std([np.linalg.norm(m[:3, 3]) for m in M_board2base])
    

    return calibrate_mean, calibrate_std, M_cam2end


def enhance_image_for_detection(image):
    """
    图像预处理，提高 AprilTag 检测稳定性
    """
    # 1. 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 2. 直方图均衡化（改善光照）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. 双边滤波（保留边缘的同时降噪）
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 4. 锐化（可选，增强边缘）
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened


def main(use_camera=True):
    """
    Main function to run the camera and robot control loop.
    """
    np.set_printoptions(suppress=True)

    # cam=OrbbecCamera.OrbbecCamera(device_index=0)
    cam =  Camera(camera_model='d405')
    camera_matrix = cam.get_camera_matrix('color')
    distortion_coefficients = cam.get_distortion_coeffs('color')
    image_id = 0
    calibration_path = './Calibration_Pic/'
    os.makedirs(calibration_path, exist_ok=True)
    robot = DobotRobot("192.168.5.1", no_gripper=True)
    
    # 运动到初始位姿
    init_joint_positions = np.array([-90.0, 0.0, -90.0, 0.0, 90.0, 90.0, 1.0])
    robot.moveJ(init_joint_positions)
    
    # 初始化参数
    # squaresX = 14
    # squaresY = 9
    # squareLength = 0.02
    # markerLength = 0.015
    # aruco_type = cv2.aruco.DICT_5X5_1000
    
    # 创建字典和标定板对象
    # dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
    # board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, dictionary)
    # aruco_detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
    aruco_params = cv2.aruco.DetectorParameters()

    # Lists to store transformation data
    robot_transforms, camera_transforms = [], []
    R_end2base, t_end2base, R_board2camera, t_board2camera = [], [], [], []
    M_board2base_list = []
    M_cam2end = np.eye(4)
    method={
        "TSAI":cv2.CALIB_HAND_EYE_TSAI,
        "PARK":cv2.CALIB_HAND_EYE_PARK,
        "DANIILIDIS":cv2.CALIB_HAND_EYE_DANIILIDIS,
        "HORAUD":cv2.CALIB_HAND_EYE_HORAUD,
    }
    
    # ==================== 键盘控制参数 ====================
    # 移动步长（毫米和度）
    pos_step = 50.0   # 位置步长 10mm
    rot_step = 10.0    # 旋转步长 5度
    pos_step_fine = 1.0   # 精细位置步长 1mm
    rot_step_fine = 1.0   # 精细旋转步长 1度
    
    fine_mode = False  # 精细调整模式
    
    print("\n" + "="*70)
    print("🎮 键盘控制手眼标定程序")
    print("="*70)
    print("\n📋 控制说明:")
    print("  位置控制 (相对于当前位置):")
    print("    ↑/↓    : Y轴 前进/后退")
    print("    ←/→    : X轴 左移/右移")
    print("    W/S    : Z轴 上升/下降")
    print("\n  姿态控制:")
    print("    Q/E    : 绕Z轴旋转 (Rz)")
    print("    A/D    : 绕X轴旋转 (Rx)")
    print("    Z/C    : 绕Y轴旋转 (Ry)")
    print("\n  功能键:")
    print("    Space  : 采集当前位姿数据")
    print("    R      : 回到初始位姿")
    print("    F      : 切换精细/粗调模式 (当前: 粗调)")
    print("    P      : 显示当前位姿")
    print("    H      : 显示帮助信息")
    print("    ESC    : 退出程序")
    print("="*70 + "\n")
    
    try:
        count = 0
        while True:
            color_image, _ = cam.get_frames()

            undistorted_image = cv2.undistort(color_image, camera_matrix, distortion_coefficients)
            # marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(undistorted_image)
            gray = enhance_image_for_detection(color_image)
            # cv2.imshow('gray',gray)
            marker_corners, marker_ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

            image_copy = color_image.copy()
            
            markerLength = 0.08 # 示例：假设标记的边长为 9 厘米 (0.05米)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(marker_corners, markerLength, camera_matrix, distortion_coefficients)
            # 遍历所有检测到的 marker
            if marker_ids is not None:
                for i in range(len(marker_ids)):
                    rvec = rvecs[i][0]
                    
                    tvec = tvecs[i][0]
                    cv2.drawFrameAxes(image_copy, camera_matrix, distortion_coefficients, rvec, tvec, 0.05) # 轴的长度为 5cm


            # 显示图像
            cv2.imshow('Hand-Eye Calibration (Press H for help)', image_copy)
            
            key = cv2.waitKey(1) & 0xFF
            # ==================== 键盘控制 ====================
            if key == 255:  # 无按键
                continue
                
            # 获取当前位姿
            curr_pose = robot.get_XYZrxryrz_state()
            original_pose = curr_pose.copy()
            step_pos = pos_step_fine if fine_mode else pos_step
            step_rot = rot_step_fine if fine_mode else rot_step
            moved = False
            
            # 位置控制
            if key == ord('i') or key == ord('I'):  # Y+ 前进
                curr_pose[1] += step_pos
                moved = True
                # print(f"I: Y+ {step_pos}mm")
            elif key == ord('k') or key == ord('K'):  # Y- 后退
                curr_pose[1] -= step_pos
                moved = True
                # print(f"K: Y- {step_pos}mm")
            elif key == ord('j') or key == ord('J'):  # X- 左移
                curr_pose[0] -= step_pos
                moved = True
                # print(f"J: X- {step_pos}mm")
            elif key == ord('l') or key == ord('L'):  # X+ 右移
                curr_pose[0] += step_pos
                moved = True
                # print(f"L: X+ {step_pos}mm")
            elif key == ord('w') or key == ord('W'):  # Z+
                curr_pose[2] += step_pos
                moved = True
                # print(f"W Z+ {step_pos}mm")
            elif key == ord('s') or key == ord('S'):  # Z-
                curr_pose[2] -= step_pos
                moved = True
                # print(f"S Z- {step_pos}mm")
                
            # 姿态控制
            elif key == ord('q') or key == ord('Q'):  # Rz+
                curr_pose[5] += step_rot
                moved = True
                # print(f"Q Rz+ {step_rot}°")
            elif key == ord('e') or key == ord('E'):  # Rz-
                curr_pose[5] -= step_rot
                moved = True
                # print(f"E Rz- {step_rot}°")
            elif key == ord('a') or key == ord('A'):  # Rx+
                curr_pose[3] += step_rot
                moved = True
                # print(f"A Rx+ {step_rot}°")
            elif key == ord('d') or key == ord('D'):  # Rx-
                curr_pose[3] -= step_rot
                moved = True
                # print(f"D Rx- {step_rot}°")
            elif key == ord('z') or key == ord('Z'):  # Ry+
                curr_pose[4] += step_rot
                moved = True
                # print(f"Z Ry+ {step_rot}°")
            elif key == ord('c') or key == ord('C'):  # Ry-
                curr_pose[4] -= step_rot
                moved = True
                # print(f"C Ry- {step_rot}°")   
                
            # 执行移动
            if moved:
                try:
                    robot.moveL(curr_pose)
                    # time.sleep(0.5)  # 等待运动完成
                except Exception as e:
                    print(f"⚠️ 移动失败: {e}")
                    print("🔄 尝试回到原位...")
                    robot.moveL(original_pose)
                    # time.sleep(0.5)
            
            # ==================== 功能键 ====================
            # 采集数据
            elif key == 32:  # Space
                image_save_path = "./collect_data/"
                cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                                cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow("detection", color_image)  # 窗口显示，显示名为 Capture_Video

                k = cv2.waitKey(1) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧

                print(f"采集第{count}组数据...")
                pose = robot.get_XYZrxryrz_state()  # 获取当前机械臂状态 需要根据实际使用的机械臂获得
                x, y, z, rx, ry, rz = pose
                rx, ry, rz = np.deg2rad([rx, ry, rz])
                pose = [x/1000.0, y/1000.0, z/1000.0, rx, ry, rz]
                print(f"机械臂pose:{pose}")

                with open(f'{image_save_path}poses.txt', 'a+') as f:
                    # 将列表中的元素用空格连接成一行
                    pose_ = [str(i) for i in pose]
                    new_line = f'{",".join(pose_)}\n'
                    # 将新行附加到文件的末尾
                    f.write(new_line)

                cv2.imwrite(image_save_path + str(count) + '.jpg', color_image)
                count += 1
                # if rvec is None or tvec is None:
                #     print("⚠️ 未检测到标定板，请调整位姿后再采集！")
                #     continue
                
                # print(f"\n{'='*50}")
                # print(f"📸 采集第 {image_id + 1} 组数据...")
                
                # # 保存数据
                # R_charuco_to_camera, _ = cv2.Rodrigues(rvec)
                # M_board2camera = np.eye(4)
                # M_board2camera[:3, :3] = R_charuco_to_camera
                # M_board2camera[:3, 3] = tvec.flatten()
                
                # M_end2base = robot.get_pose_matrix()
                
                # R_board2camera.append(M_board2camera[:3, :3])
                # t_board2camera.append(M_board2camera[:3, 3])
                # R_end2base.append(M_end2base[:3, :3])
                # t_end2base.append(M_end2base[:3, 3])
                
                # print(f"✅ 已保存第 {image_id + 1} 组数据")
                # print(f"   末端位置: {M_end2base[:3, 3]}")
                # image_id += 1
                
                # # 执行标定计算
                # if len(t_end2base) > 3:
                #     std_min = []
                #     M_list = []
                #     print(f"\n🔧 执行手眼标定 (已采集 {len(t_end2base)} 组数据)...")
                    
                #     for calib_name, calib_method in method.items():
                #         try:
                #             calib_mean, calib_std, M = hand_eye_calibration(
                #                 R_end2base, t_end2base, 
                #                 R_board2camera, t_board2camera, 
                #                 calib_method
                #             )
                #             print(f"   {calib_name:12s}: 标准差 = {calib_std:.6f} mm")
                #             M_list.append(M)
                #             std_min.append(calib_std)
                #         except Exception as e:
                #             print(f"   {calib_name:12s}: 计算失败 - {e}")
                #             M_list.append(None)
                #             std_min.append(float('inf'))
                    
                #     min_value = min(std_min)
                #     print(f"\n   最佳方法: {list(method.keys())[std_min.index(min_value)]}")
                #     print(f"   最小标准差: {min_value:.6f} m")
                    
                #     if len(t_end2base) >= 10 and 0.000001 < min_value < 0.0015:
                #         print("\n🎉 标定精度达标！")
                #         break
                
                # print(f"{'='*50}\n")
            
            # 回到初始位姿
            elif key == ord('r') or key == ord('R'):
                print("\n🔄 回到初始位姿...")
                robot.moveJ(init_joint_positions)
                # time.sleep(2.0)
                print("✅ 已回到初始位姿\n")
            
            # 切换精细/粗调模式
            elif key == ord('f') or key == ord('F'):
                fine_mode = not fine_mode
                mode = "精细" if fine_mode else "粗调"
                print(f"\n🔧 切换到{mode}模式")
                print(f"   位置步长: {pos_step_fine if fine_mode else pos_step}mm")
                print(f"   旋转步长: {rot_step_fine if fine_mode else rot_step}°\n")
            
            # 显示当前位姿
            elif key == ord('p') or key == ord('P'):
                curr_pose = robot.get_XYZrxryrz_state()
                print("\n📍 当前位姿:")
                print(f"   位置: X={curr_pose[0]:.2f}, Y={curr_pose[1]:.2f}, Z={curr_pose[2]:.2f} mm")
                print(f"   姿态: Rx={curr_pose[3]:.2f}, Ry={curr_pose[4]:.2f}, Rz={curr_pose[5]:.2f} °\n")
            
            # 显示帮助
            elif key == ord('h') or key == ord('H'):
                print("\n" + "="*70)
                print("📋 键盘控制说明:")
                print("  位置: ↑↓←→ (XY平面), W/S (Z轴)")
                print("  姿态: Q/E (Rz), A/D (Rx), Z/C (Ry)")
                print("  功能: Space(采集) R(回原位) F(切换模式) P(显示位姿) ESC(退出)")
                print("="*70 + "\n")
            
            # 退出
            elif key == 27:  # ESC
                print("\n❌ 用户终止程序")
                break

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if len(t_end2base) > 0:
            print(f"\n{'='*70}")
            print("💾 保存标定结果...")
            print(std_min)
            min_value = min(std_min)
            calib_mean = list(method.keys())[std_min.index(min_value)]
            best_M = M_list[std_min.index(min_value)]
            print(f"最佳标定方法: {calib_mean}")
            print(f"最小标准差: {min_value:.6f} mm")
            print("\n相机到末端的变换矩阵:")
            print(best_M)
            np.savetxt("T_camera2end.txt", best_M)
            print("✅ 已保存到 T_camera2end.txt")
            print(f"{'='*70}\n")
        
        cv2.destroyAllWindows()

def draw_axis_on_image(image, tag, rvec, tvec, camera_matrix, distortion_coefficients):
    """
    Draw the axis on the image based on the tag pose.
    """
    axis_points = np.float32([[0.05, 0, 0], [0, 0.05, 0], [0, 0, -0.05]]).reshape(-1, 3)
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, distortion_coefficients)

    center = tuple(tag.center.astype(int))
    image = cv2.line(image, center, tuple(img_points[0].ravel().astype(int)), (255, 0, 0), 3)
    image = cv2.line(image, center, tuple(img_points[1].ravel().astype(int)), (0, 255, 0), 3)
    image = cv2.line(image, center, tuple(img_points[2].ravel().astype(int)), (0, 0, 255), 3)
    return image

def save_calibration_data(path, camera_data, robot_data):
    """
    Save the calibration data to CSV files.
    """
    with open(os.path.join(path, 'camera_data.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(camera_data)

    with open(os.path.join(path, 'robot_data.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(robot_data)

if __name__ == "__main__":
    main(use_camera=True)