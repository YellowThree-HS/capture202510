import cv2
import os
import time
import numpy as np
from datetime import datetime
from lib.camera import Camera
from lib.yolo_and_sam import YOLOSegmentator
from lib.mask2pose import mask2pose, draw_pose_axes
from lib.dobot import DobotRobot
from cv2 import aruco

def detect_aruco_pose(camera,image):
    """
    检测ArUco板子的位姿
    
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
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    aruco_params = cv2.aruco.DetectorParameters()
    # ArUco板子参数 (8x8cm)
    marker_size = 0.08  # 8cm = 0.08m
    board_size = (4, 4)  # 4x4的ArUco板子
    
    camera_matrix = camera.get_camera_matrix('color')
    dist_coeffs = camera.get_distortion_coeffs('color')
    
    # 检测ArUco标记
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is None or len(ids) < 1:  # 至少需要1个标记
        return False, None, None, None, None
    
    # 对于单个标记，使用estimatePoseSingleMarkers
    if len(ids) == 1:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs
        )
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]
        success = True
    else:
        # 对于多个标记，暂时只处理第一个标记
        # TODO: 实现真正的板子位姿估计
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[:1], marker_size, camera_matrix, dist_coeffs
        )
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]
        success = True

    
    aruco_to_cam = np.eye(4)
    aruco_to_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
    aruco_to_cam[:3, 3] = tvec.flatten()
    
    return aruco_to_cam



def main():

    cam = Camera(camera_model='D405')  # 初始化相机
    robot = DobotRobot(robot_ip='192.168.5.1')  # 初始化机械臂
    # robot.r_inter.StartDrag()
    # return 0
    
    
    
    # # 读取原始图像
    color_image_path = 'test/color.png'
    depth_image_path = 'test/depth.png'
    color_image = cv2.imread(color_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    # color_image, depth_image = cam.get_frames()
    # cv2.imwrite('test/color.png',color_image)
    # cv2.imwrite('test/depth.png',depth_image)
    # return 0

    # 确保深度图是二维的
    if len(depth_image.shape) == 3:
        print(f"  深度图是3通道,取第一个通道")
        depth_image = depth_image[:, :, 0]
    
    # 转换为浮点数并使用正确的深度比例转换为米
    # RealSense D405的深度比例是0.0001 (不是0.001!)
    # 所以要除以10000 (不是1000)
    depth_scale = 0.0001  # 从相机信息中获取
    depth_image = depth_image.astype(np.float32) * depth_scale
    
    
    T_cam2base = np.eye(4)  # 假设相机位姿已知

    categories_to_find = ['cup']
    segmentator = YOLOSegmentator()
    
    # 打印图像信息用于调试
    print(f"\n🔍 图像信息:")
    print(f"  图像尺寸: {color_image.shape}")
    print(f"  数据类型: {color_image.dtype}")
    print(f"  检测类别: {categories_to_find}")
    
    # 使用调整后的图像进行检测
    result = segmentator.detect_and_segment_all(
        image=color_image,
        categories=categories_to_find,
        save_result=False,
        conf=0.1  # 降低置信度阈值，从0.1降到0.051
    )

    # result['objects']是一个字典的list，每个字典代表一个物体包含类别、置信度、边界框和掩码

    intrinsics = cam.get_camera_matrix()
    
    # 检查检测是否成功
    if not result['success']:
        print("❌ 检测失败：没有检测到任何物体")
        print("可能的原因：")
        print("1. 图像中没有 'cup' 物体")
        print("2. 物体置信度太低（当前阈值: 0.1）")
        print("3. 图像质量问题")
        return
    
    if 'objects' not in result or len(result['objects']) == 0:
        print("❌ 没有检测到任何物体")
        return
    
    for idx, obj in enumerate(result['objects']):
        print(f"\n物体 {idx + 1}:")
        print(f"  类别: {obj['class']}")
        print(f"  置信度: {obj['confidence']:.2f}")
        print(f"  边界框: {obj['bbox_xyxy']}")
        
        # 如果有掩码，计算位姿
        if obj['mask'] is not None:
            print(f"  掩码尺寸: {obj['mask'].shape}")
            print(f"  彩色图像尺寸: {color_image.shape}")
            print(f"  深度图像尺寸: {depth_image.shape}")
            
            pose, T_object2cam = mask2pose(
                mask=obj['mask'],
                depth_image=depth_image,
                color_image=color_image,
                intrinsics=intrinsics,
                T_cam2base=T_cam2base,
                object_class=obj['class']
            )
            
            if pose is not None:
                print(f"   📍 相机坐标系下位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
                print(f"   📐 相机坐标系下姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
                break  # 只处理第一个检测到的物体
            else:
                print(f"   ❌ 位姿估计失败")
        else:
            print(f"   ⚠️ 未获取到掩码")
    
    
    

    calibration_file = 'best_hand_eye_calibration.npy'
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"手眼标定文件不存在: {calibration_file}")
    
    hand_eye_matrix = np.load(calibration_file)
            
    # aruco_to_cam = detect_aruco_pose(cam,color_image)
    
    target_to_cam = T_object2cam
    
    pose_matrix =  robot.get_pose_matrix() @ hand_eye_matrix @ target_to_cam
    
    # robot.moveL(pose_matrix)
    print("目标物体相对于机械臂基座的位姿矩阵:\n", pose_matrix)

    # init_joint_state = np.array([-90.0, 0.0, -90.0, 0.0, 90.0, 90.0])  # left arm
    # robot.moveJ(init_joint_state)
    # pose_matrix = np.array([[    0.26074  ,  -0.10167  ,  -0.96004  ,  -0.56438],
    #                         [   -0.96455  ,  0.014468 ,    -0.2635  ,  -0.18789],
    #                         [   0.040679  ,  0.99471  , -0.094291   ,   0.5605],
    #                         [          0  ,         0 ,          0   ,        1]])
#     pose_matrix = np.array( [[    0.99875    0.031514   -0.038725     0.40659]
#  [    0.04272    -0.94085     0.33611      0.5097]
#  [  -0.025842    -0.33734    -0.94103     0.45623]
#  [          0           0           0           1]])
    draw_pose_axes(color_image, intrinsics, T_object2cam)
    from scipy.spatial.transform import Rotation as R
    rx, ry, rz = R.from_matrix(pose_matrix[:3, :3]).as_euler('xyz', degrees=True)
    x,y,z = np.array(pose_matrix[:3, 3]) *1000.0
    ry = ry + 90 
    robot.moveL(np.array([x,y,z,rx,ry,rz]))
    # print(f"ArUco板相对于机械臂基座位置 (mm): X={position[0]*1000:.2f}, Y={position[1]*1000:.2f}, Z={position[2]*1000:.2f}\n")
    # print(f"ArUco板相对于机械臂基座变换矩阵:\n{pose_matrix}")



if __name__ == "__main__":
    main()