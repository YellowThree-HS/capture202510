import numpy as np
import cv2
import os
import csv
# import pupil_apriltags as apriltag
# from utils import frame_to_bgr_image
import transforms3d as tfs
import math
import random
import time
# import OrbbecCamera
# import robot_controller

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

def main(use_camera=True):
    """
    Main function to run the camera and robot control loop.
    """
    np.set_printoptions(suppress=True)
    # Auboi5Robot.initialize()
    # robot = robot_controller.URRobot()
    # handle = robot.create_context()
    # robot.connect('192.168.1.137', 8899)
    robot = DobotRobot("192.168.5.1", no_gripper=True)
    # 运动到初始位姿
    init_joint_positions = np.array([-90.0, 0.0, -90.0, 0.0, 90.0, 90.0, 1.0])
    robot.moveJ(init_joint_positions)
    
    # cam=OrbbecCamera.OrbbecCamera(device_index=0)
    cam =  Camera(camera_model='d405')
    camera_matrix = cam.get_camera_matrix('color')
    distortion_coefficients = cam.get_distortion_coeffs('color')
    image_id = 0
    calibration_path = './Calibration_Pic/'
    os.makedirs(calibration_path, exist_ok=True)

    # camera_params = cam.rgb_intrinsic
    # distortion = cam.rgb_distortion
    # camera_matrix = np.array([[camera_params.fx, 0, camera_params.cx],
    #                           [0, camera_params.fy, camera_params.cy],
    #                           [0, 0, 1]], dtype=np.float64)
    # distortion_coefficients = np.array([distortion.k1, distortion.k2, distortion.p1,
                                        # distortion.p2, distortion.k3], dtype=np.float64)
    camera_matrix = cam.get_camera_matrix()
    distortion_coefficients = cam.get_distortion_coeffs()
    # 初始化参数
    squaresX = 14
    squaresY = 9
    squareLength = 0.02
    markerLength = 0.015
    aruco_type = cv2.aruco.DICT_5X5_1000
    
    # 创建字典和标定板对象[17](@ref)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
    board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, dictionary)

    # 创建检测器对象（新版API）[6](@ref)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)

    # Lists to store transformation data
    robot_transforms, camera_transforms = [], []
    R_end2base, t_end2base, R_board2camera, t_board2camera = [], [], [], []
    M_board2base_list = []
    M_cam2end = np.eye(4)
    method={
        "TSAI":cv2.CALIB_HAND_EYE_TSAI,
        # "ANDREFF":cv2.CALIB_HAND_EYE_ANDREFF,
        "PARK":cv2.CALIB_HAND_EYE_PARK,
        "DANIILIDIS":cv2.CALIB_HAND_EYE_DANIILIDIS,
        "HORAUD":cv2.CALIB_HAND_EYE_HORAUD,
    }
    
    # 移动步长（毫米和度）
    pos_step = 50.0   # 位置步长 10mm
    rot_step = 10.0    # 旋转步长 5度
    pos_step_fine = 1.0   # 精细位置步长 1mm
    rot_step_fine = 1.0   # 精细旋转步长 1度
    
    fine_mode = False  # 精细调整模式
    
    try:
        while True:
            try:
                color_image, _ = cam.get_frames()
            except Exception as e:
                print(e)
                continue
            undistorted_image = cv2.undistort(color_image, camera_matrix, distortion_coefficients)
            marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(undistorted_image)
            image_copy = color_image.copy()

            # 处理检测结果
            charuco_corners = []
            charuco_ids = []
            interpolated_corners = 0

            if marker_ids is not None and len(marker_ids) > 0:
                # 插值ChArUco角点[17](@ref)
                charuco_detector = cv2.aruco.CharucoDetector(board)
                charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(undistorted_image)
                
                interpolated_corners = 0 if charuco_corners is None else len(charuco_corners)
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, 
                                                                        charuco_ids, 
                                                                        board, 
                                                                        camera_matrix, 
                                                                        distortion_coefficients, None, None)
                
                # print("tvec", tvec)
                
                if rvec is not None and tvec is not None:
                    cv2.drawFrameAxes(image_copy, camera_matrix, distortion_coefficients, rvec, tvec, length=0.03)

            # 保存并显示结果
            cv2.imshow('Orbbec Camera', image_copy)
            key = cv2.waitKey(1)
            
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
            
            if key == ord("r") or key == ord("R"):
                robot.moveJ(init_joint_positions)
            
            if key == 32:  # Space bar to capture data
                # robot_pose = robot.get_pose_axis()
                # print(robot_pose)
                R_charuco_to_camera, _ = cv2.Rodrigues(rvec)
                M_board2camera = np.eye(4)
                M_board2camera[:3, :3] = R_charuco_to_camera
                M_board2camera[:3, 3] = tvec.flatten()

                print(f"Saved data for image {image_id}")
                image_id += 1

                # # Convert poses to transformation matrices
                R_board2camera.append(M_board2camera[:3, :3])
                t_board2camera.append(M_board2camera[:3, 3])

                M_end2base = robot.get_pose_matrix()
                R_end2base.append(M_end2base[:3, :3])
                t_end2base.append(M_end2base[:3, 3])

                if len(t_end2base) > 5:
                    std_min=[]
                    M_list=[]
                    for calib_name,calib_method in method.items():
                        print("Calibration method:",calib_name)
                        breakpoint()
                        calib_mean, calib_std, M = hand_eye_calibration(R_end2base, t_end2base, R_board2camera, t_board2camera,calib_method)
                        print("mean:",calib_mean[:3, 3].T)
                        M_list.append(M)
                        std_min.append(calib_std)
                    print(M_list)
                    min_value = min(std_min)
                    print("min_value:",min_value)
                    if len(t_end2base) > 10 and 0.000001< min_value < 0.0015:
                        break

            elif key == 27:  # ESC to exit
                print("Program terminated")
                break
    except Exception as e:
        print('e:',e)
    finally:
        calib_mean=list(method.keys())[std_min.index(min_value)]
        best_M = M_list[std_min.index(min_value)]
        print("------------------------calib_mean:",calib_mean,"------------------------")
        print(best_M)
        np.savetxt("T_camera2end.txt", best_M)
        save_calibration_data(calibration_path, camera_transforms, robot_transforms)
        # 保存变换矩阵到文件
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