# 这个文件是读取collect_data/collect_data/poses.txt和collect_data/collect_data/images/中的位姿和图像，
# 用来测试不同的手眼标定算法，并找出最优的算法
import cv2
import numpy as np
import os

# 从main.py导入的相同配置
XX = 11
YY = 8
L = 0.02

objp = np.zeros((XX * YY, 3), np.float32)
objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2) * L

objpoints = []
imgpoints = []

def euler_to_rotation_matrix(rx, ry, rz):
    """
    根据越疆机器人官方文档实现欧拉角到旋转矩阵的转换
    旋转顺序：X -> Y -> Z（外旋，绕固定轴）
    rx=γ, ry=β, rz=α
    
    注意：虽然旋转顺序是X->Y->Z，但矩阵乘法顺序是Rz @ Ry @ Rx
    这是因为外旋的定义：先绕X轴旋转，再绕Y轴旋转，最后绕Z轴旋转
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # 正确的实现：R = Rz @ Ry @ Rx
    # 这对应于外旋X->Y->Z的顺序
    R = Rz @ Ry @ Rx
    return R

def pose_to_transformation_matrix(x, y, z, rx, ry, rz):
    R = euler_to_rotation_matrix(rx, ry, rz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

# 读取位姿数据
poses_file = 'collect_data/poses.txt'
poses = []
with open(poses_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            values = [float(v) for v in line.split(',')]
            poses.append(values)

# 读取图像并检测角点
images_folder = 'collect_data/'
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
    np.save('best_hand_eye_calibration.npy', sorted_by_score[0]['T_matrix'])
    print(f"\n最优结果已保存到: best_hand_eye_calibration.npy")

print("\n" + "="*60)

