import cv2
import numpy as np
import os
from lib.dobot import DobotRobot
from lib.camera import Camera
np.set_printoptions(precision=8, suppress=True)


count = 0 

hand = 'right'

save_path = f"./calibration/calib_data_{hand}"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

def collect_data(pose,color_image):
    global count
    with open(f'{save_path}/poses.txt', 'a+') as f:
        # 将列表中的元素用空格连接成一行
        pose_ = [str(i) for i in pose]
        new_line = f'{",".join(pose_)}\n'
        # 将新行附加到文件的末尾
        f.write(new_line)

    cv2.imwrite(save_path + '/' + str(count) + '.jpg', color_image)

    count+=1
    return 


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    return R, t

def pose_to_transformation_matrix(x, y, z, rx, ry, rz):
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def process_arm_pose(arm_pose_file):
    """处理机械臂的pose文件。 采集数据时， 每行保存一个机械臂的pose信息， 该pose与拍摄的图片是对应的。
    pose信息用6个数标识， 【x,y,z,Rx, Ry, Rz】. 需要把这个pose信息用旋转矩阵表示。"""

    R_arm, t_arm = [], []
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        all_lines = f.readlines()
    for line in all_lines:
        pose = [float(v) for v in line.split(',')]
        R, t = pose_to_homogeneous_matrix(pose=pose)
        R_arm.append(R)
        t_arm.append(t)
    return R_arm, t_arm


def hand_eye_calibrate():
    # 从main.py导入的相同配置
    XX = 11
    YY = 8
    L = 0.02

    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2) * L

    objpoints = []
    imgpoints = []
    poses_file = f'{save_path}/poses.txt'
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values = [float(v) for v in line.split(',')]
                poses.append(values)

    # 读取图像并检测角点
    images_folder = f'{save_path}/'
    valid_images = []

    for i in range(len(poses)):
        img_path = os.path.join(images_folder, f'{i}.jpg')
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            valid_images.append(i)

    # 相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # 准备手眼标定数据
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for idx, img_idx in enumerate(valid_images):
        x, y, z, rx, ry, rz = poses[img_idx]
        T_gripper2base = pose_to_transformation_matrix(x, y, z, rx, ry, rz)
        R_gripper2base.append(T_gripper2base[:3, :3])
        t_gripper2base.append(T_gripper2base[:3, 3].reshape(3, 1))
        
        rvec = rvecs[idx]
        tvec = tvecs[idx]
        R_target2cam.append(cv2.Rodrigues(rvec)[0])
        t_target2cam.append(tvec.reshape(3, 1))

    # 测试所有可用的手眼标定算法
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
        (cv2.CALIB_HAND_EYE_PARK, "Park"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis")
    ]

    print("="*60)
    print("测试不同的手眼标定算法")
    print("="*60)

    results = []

    for method, name in methods:
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base,
                t_gripper2base=t_gripper2base,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,
                method=method
            )
            
            T_cam2gripper = np.eye(4)
            T_cam2gripper[:3, :3] = R_cam2gripper
            T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
            
            # 计算一致性误差
            position_errors = []
            rotation_errors = []
            
            for idx, img_idx in enumerate(valid_images):
                x, y, z, rx, ry, rz = poses[img_idx]
                T_gripper2base = pose_to_transformation_matrix(x, y, z, rx, ry, rz)
                
                R_t2c = R_target2cam[idx]
                t_t2c = t_target2cam[idx]
                T_target2cam = np.eye(4)
                T_target2cam[:3, :3] = R_t2c
                T_target2cam[:3, 3] = t_t2c.flatten()
                
                T_target2base = T_gripper2base @ T_cam2gripper @ T_target2cam
                
                if idx == 0:
                    T_target2base_ref = T_target2base.copy()
                else:
                    pos_error = np.linalg.norm(T_target2base[:3, 3] - T_target2base_ref[:3, 3])
                    position_errors.append(pos_error)
                    
                    R_diff = T_target2base[:3, :3] @ T_target2base_ref[:3, :3].T
                    trace = np.clip(np.trace(R_diff), -1, 3)
                    angle_error = np.arccos((trace - 1) / 2)
                    rotation_errors.append(np.degrees(angle_error))
            
            mean_pos_error = np.mean(position_errors) * 1000  # 转换为毫米
            mean_rot_error = np.mean(rotation_errors)
            
            results.append({
                'name': name,
                'pos_error': mean_pos_error,
                'rot_error': mean_rot_error,
                'T_matrix': T_cam2gripper
            })
            
            print(f"\n算法: {name}")
            print(f"  位置误差: {mean_pos_error:.4f} mm")
            print(f"  姿态误差: {mean_rot_error:.4f} 度")
            
        except Exception as e:
            print(f"\n算法: {name}")
            print(f"  错误: {str(e)}")

    # 找出最优算法
    if results:
        print("\n" + "="*60)
        print("算法对比总结")
        print("="*60)
        
        # 按位置误差排序
        sorted_by_pos = sorted(results, key=lambda x: x['pos_error'])
        print(f"\n位置精度最优: {sorted_by_pos[0]['name']} ({sorted_by_pos[0]['pos_error']:.4f} mm)")
        
        # 按姿态误差排序
        sorted_by_rot = sorted(results, key=lambda x: x['rot_error'])
        print(f"姿态精度最优: {sorted_by_rot[0]['name']} ({sorted_by_rot[0]['rot_error']:.4f} 度)")
        
        # 综合评分（位置权重更高）
        for r in results:
            r['score'] = r['pos_error'] * 2 + r['rot_error']
        sorted_by_score = sorted(results, key=lambda x: x['score'])
        
        print(f"\n推荐使用算法: {sorted_by_score[0]['name']}")
        print(f"  位置误差: {sorted_by_score[0]['pos_error']:.4f} mm")
        print(f"  姿态误差: {sorted_by_score[0]['rot_error']:.4f} 度")
        print(f"\n变换矩阵:\n{sorted_by_score[0]['T_matrix']}")
        
        # 保存最优结果
        np.save(f'hand_eye_calib_{hand}.npy', sorted_by_score[0]['T_matrix'])
        print(f"\n最优结果已保存到: hand_eye_calib_{hand}.npy")

    print("\n" + "="*60)










def main():
    # 初始化相机

    collect_data_flag = False  # 设置为True以采集数据，False则只进行标定
    if collect_data_flag:
        cam = Camera(camera_model='D405')  # 初始化相机

        # 初始化机械臂
        robot = DobotRobot("192.168.5.2", no_gripper=True)
        robot.r_inter.StartDrag()
        print("已进入拖拽模式，请手动拖动机械臂到不同位置，对准标定板。每次按空格采集一组数据，ESC退出。")

        
        with open(f"{save_path}poses.txt", "w") as f:
            f.write("")  # 清空文件内容

        while True:
            color_image, depth_image = cam.get_frames()
            if color_image is None:
                continue
            cv2.imshow("drag_hand_eye", color_image)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # 空格采集
                
                pose = robot.get_XYZrxryrz_state() 
                x, y, z, rx, ry, rz = pose
                rx, ry, rz = np.deg2rad([rx, ry, rz])
                pose = [x/1000.0, y/1000.0, z/1000.0, rx, ry, rz]


                XX = 11
                YY = 8      
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
                
                if ret:
                    collect_data(pose,color_image)
                else:
                    print('没有检测到标定板')
                # if count >= 5:  # 最少采集5组数据
                #     hand_eye_calibrate()
                #     continue
                print(f'记录第{count}组数据')
            elif key == 27:  # ESC退出
                break

        robot.r_inter.StopDrag()
        print("采集结束")
        hand_eye_calibrate()
    else:
        hand_eye_calibrate()


    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()