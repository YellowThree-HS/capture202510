import cv2
import numpy as np
import os
from lib.dobot import DobotRobot
from lib.camera import Camera
from scipy.spatial.transform import Rotation as R


save_path = "./collect_data/"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

def collect_data(T,color_image,count):
    
    x, y, z = T[:3, 3]
    euler = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)  # 弧度
    
    pose = [x, y, z] + list(euler)
    print(f"采集第{count}组，机械臂pose: {pose}")
    
    
    with open(f"{save_path}poses.txt", "a+") as f:
        pose_str = ",".join([str(i) for i in pose])
        f.write(f"{pose_str}\n")
        
    cv2.imwrite(f"{save_path}{count}.jpg", color_image)

def detect_aruco_pose(cam, color_image):
    # 检测棋盘格,棋盘是11x8，方格边长15mm
    squaresX, squaresY, squareLength = 11, 8, 0.02
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    objp = np.zeros((squaresX * squaresY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:squaresX, 0:squaresY].T.reshape(-1, 2)
    objp = squareLength * objp
    
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (squaresX, squaresY), None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        board_rvec, board_tvec = cv2.solvePnP(objp, corners2, np.eye(3), np.zeros(5))[1:3]
        print("标定板检测成功，已保存数据。")
        return board_rvec, board_tvec
    else:
        print("未检测到标定板，请调整后重试。")
        return None, None

def hand_eye_calibrate(arm_poses, all_board_rvecs, all_board_tvecs):
    """
    手眼标定，使用多种算法并选择最优结果
    """
    R_arm = [T[:3, :3] for T in arm_poses]
    t_arm = [T[:3, 3].reshape(3, 1) for T in arm_poses]  # 确保是 (3,1) 形状
    R_board = all_board_rvecs
    t_board = all_board_tvecs
    
    methods = {
        "Tsai-Lenz": cv2.CALIB_HAND_EYE_TSAI,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS
    }
    
    results = {}
    print("开始手眼标定...")
    
    for method_name, method_flag in methods.items():
        try:
            print(f"  测试方法：{method_name}")
            R, t = cv2.calibrateHandEye(R_arm, t_arm, R_board, t_board, method_flag)
            
            if R is None or t is None:
                print(f"    ❌ {method_name} 标定失败")
                continue
            
            # 构建相机到末端的齐次变换矩阵
            M_cam2end = np.eye(4)
            M_cam2end[:3, :3] = R
            M_cam2end[:3, 3] = t.reshape(3)
            
            # 计算标定板在基座系下的位置一致性（评估标定质量）
            board_positions = []
            for sub_R_arm, sub_t_arm, sub_rvec, sub_tvec in zip(R_arm, t_arm, R_board, t_board):
                # 末端到基座
                M_end2base = np.eye(4)
                M_end2base[:3, :3] = sub_R_arm
                M_end2base[:3, 3] = sub_t_arm.reshape(3)
                
                # 标定板到相机
                R_board2cam, _ = cv2.Rodrigues(sub_rvec)
                M_board2cam = np.eye(4)
                M_board2cam[:3, :3] = R_board2cam
                M_board2cam[:3, 3] = sub_tvec.reshape(3)
                
                # 标定板到基座
                M_board2base = M_end2base @ M_cam2end @ M_board2cam
                board_positions.append(M_board2base[:3, 3])
            
            # 计算位置标准差
            board_positions = np.array(board_positions)
            mean_pos = np.mean(board_positions, axis=0)
            std_xyz = np.std(board_positions, axis=0)
            rms_error = np.sqrt(np.mean(np.sum((board_positions - mean_pos)**2, axis=1)))
            
            results[method_name] = {
                'R': R,
                't': t,
                'M_cam2end': M_cam2end,
                'std_xyz': std_xyz,
                'rms_error': rms_error,
                'mean_pos': mean_pos
            }
            
            print(f"    ✅ 完成，RMS误差: {rms_error*1000:.2f} mm")
            
        except Exception as e:
            print(f"    ❌ {method_name} 出错: {e}")
    
    if not results:
        print("❌ 所有标定方法都失败了")
        return None, None
    
    # 选择RMS误差最小的作为最优结果
    best_method = min(results, key=lambda x: results[x]['rms_error'])
    best_result = results[best_method]
    
    print(f"\n🎯 最优方法: {best_method}")
    print(f"   RMS误差: {best_result['rms_error']*1000:.2f} mm")
    print(f"   位置标准差 (XYZ): [{best_result['std_xyz'][0]*1000:.2f}, {best_result['std_xyz'][1]*1000:.2f}, {best_result['std_xyz'][2]*1000:.2f}] mm")
    
    # 保存最优结果
    np.savetxt("T_camera2end.txt", best_result['M_cam2end'])
    print("最优标定矩阵已保存到 T_camera2end.txt")
    
    return best_result['R'], best_result['t']

def main():
    count = 0 
    # 初始化相机
    cam = Camera(camera_model='D405')  # 初始化相机

    # 初始化机械臂
    robot = DobotRobot("192.168.5.2", no_gripper=True)
    robot.r_inter.StartDrag()
    print("已进入拖拽模式，请手动拖动机械臂到不同位置，对准标定板。每次按空格采集一组数据，ESC退出。")
    # breakpoint()

    arm_poses_text = []
    arm_poses = []
    all_board_rvecs = []
    all_board_tvecs = []
    T_camera2end = np.eye(4)

    # with open(f"{save_path}poses.txt", "r") as f:
    #     for line in f:
    #         arm_pose = line.strip().split(",")
    #         arm_pose = [float(i) for i in arm_pose]
    #         arm_poses_text.append(arm_pose)
    

    # for pose in arm_poses_text:
    #     T = np.eye(4)
    #     T[:3, :3] = R.from_euler('xyz', pose[3:6], degrees=False).as_matrix()
    #     T[:3, 3] = pose[:3]
    #     arm_poses.append(T)

    # image_path_dir = './collect_data/'
    # for image_name in os.listdir(image_path_dir):
    #     if image_name.endswith(".jpg"):
    #         image_path_full = os.path.join(image_path_dir, image_name)
    #         print(image_path_full)
    #         color_image = cv2.imread(image_path_full)
    #         board_rvec, board_tvec = detect_aruco_pose(cam, color_image)
    #         all_board_rvecs.append(board_rvec)
    #         all_board_tvecs.append(board_tvec)

    
    # hand_eye_calibrate(arm_poses, all_board_rvecs, all_board_tvecs)
    
    # return 0
    
    with open(f"{save_path}poses.txt", "w") as f:
        f.write("")  # 清空文件内容

    while True:
        color_image, depth_image = cam.get_frames()
        if color_image is None:
            continue
        cv2.imshow("drag_hand_eye", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # 空格采集
            T = robot.get_pose_matrix()  # 获取4x4齐次变换矩阵，位置单位为米
            board_rvec, board_tvec = detect_aruco_pose(cam, color_image)
            if board_rvec is not None and board_tvec is not None:
                all_board_rvecs.append(board_rvec)
                all_board_tvecs.append(board_tvec)
                arm_poses.append(T)  # 直接保存4x4矩阵
                collect_data(T,color_image,count)
                count += 1
                if count >= 5:  # 最少采集5组数据
                    hand_eye_calibrate(arm_poses, all_board_rvecs, all_board_tvecs)
            else:
                print("未检测到标定板，未保存数据。请调整机械臂位置后重试。")
                continue
        elif key == 27:  # ESC退出
            break

    robot.r_inter.StopDrag()
    print("采集结束")
    
            
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()