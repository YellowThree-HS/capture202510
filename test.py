import cv2
import os
import time
import numpy as np
from datetime import datetime
from lib.camera import Camera
from lib.yolo_and_sam import YOLOSegmentator
from lib.mask2pose import mask2pose, visualize_result
from lib.dobot import DobotRobot

def main():

    cam = Camera(camera_model='D405')  # 初始化相机
    robot = DobotRobot(robot_ip='192.168.5.1')  # 初始化机械臂
    # robot.r_inter.StartDrag()
    # return 0
    
    
    # # 读取原始图像
    # color_image_path = 'test/color.png'
    # depth_image_path = 'test/depth.png'
    # color_image = cv2.imread(color_image_path)
    # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    color_image, depth_image = cam.get_frames()

    
    # # 确保深度图是二维的
    # if len(depth_image.shape) == 3:
    #     print(f"  深度图是3通道,取第一个通道")
    #     depth_image = depth_image[:, :, 0]
    
    # # 转换为浮点数并使用正确的深度比例转换为米
    # # RealSense D405的深度比例是0.0001 (不是0.001!)
    # # 所以要除以10000 (不是1000)
    # depth_scale = 0.0001  # 从相机信息中获取
    # depth_image = depth_image.astype(np.float32) * depth_scale
    
    
    # T_cam2base = np.eye(4)  # 假设相机位姿已知

    # categories_to_find = ['cup']
    # segmentator = YOLOSegmentator()
    
    # # 使用调整后的图像进行检测
    # result = segmentator.detect_and_segment_all(
    #     image=color_image,
    #     categories=categories_to_find,
    #     save_result=True
    # )
    # print("Detection and segmentation result:", result)



    # # result['objects']是一个字典的list，每个字典代表一个物体包含类别、置信度、边界框和掩码
    # pose = None
    # T_cam2object = None
    
    # for idx, obj in enumerate(result['objects']):
    #     print(f"\n物体 {idx + 1}:")
    #     print(f"  类别: {obj['class']}")
    #     print(f"  置信度: {obj['confidence']:.2f}")
    #     print(f"  边界框: {obj['bbox_xyxy']}")
        
    #     # 如果有掩码，计算位姿
    #     if obj['mask'] is not None:
    #         print(f"  掩码尺寸: {obj['mask'].shape}")
    #         print(f"  彩色图像尺寸: {color_image.shape}")
    #         print(f"  深度图像尺寸: {depth_image.shape}")
            
    #         pose, T_cam2object = mask2pose(
    #             mask=obj['mask'],
    #             depth_image=depth_image,
    #             color_image=color_image,
    #             intrinsics=cam.get_camera_matrix(),
    #             T_cam2base=T_cam2base,
    #             object_class=obj['class']
    #         )
            
    #         if pose is not None:
    #             print(f"   📍 相机坐标系下位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
    #             print(f"   📐 相机坐标系下姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
    #             print(f"T:{T_cam2object}")
    #             break  # 只处理第一个检测到的物体
    #         else:
    #             print(f"   ❌ 位姿估计失败")
    #     else:
    #         print(f"   ⚠️ 未获取到掩码")
    
    # ArUco标记检测和位姿估计
    from cv2 import aruco
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
    aruco_params = cv2.aruco.DetectorParameters()

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # 获取相机内参
    camera_matrix = cam.get_camera_matrix()
    distortion_coefficients = np.zeros(5)  # D405通常畸变很小,可以设为0
    
    markerLength = 0.08  # ArUco标记的边长,单位:米 (根据实际标记尺寸修改)
    
    T_cam2object = None
    pose = None
    
    # 遍历所有检测到的 marker
    if marker_ids is not None:
        print(f"\n检测到 {len(marker_ids)} 个ArUco标记")
        
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, markerLength, camera_matrix, distortion_coefficients
        )
        
        for i in range(len(marker_ids)):
            marker_id = marker_ids[i][0]
            rvec = rvecs[i][0]  # 旋转向量
            tvec = tvecs[i][0]  # 平移向量
            
            print(f"\nArUco标记 ID: {marker_id}")
            print(f"  平移向量(相机坐标系): {tvec}")
            print(f"  旋转向量: {rvec}")
            
            # 将旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # 构建4x4变换矩阵
            T_cam2object = np.eye(4)
            T_cam2object[:3, :3] = rotation_matrix
            T_cam2object[:3, 3] = tvec
            T_object2cam = T_cam2object #hs
            
            # 转换为欧拉角用于显示
            from scipy.spatial.transform import Rotation as R
            r = R.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)
            
            # 构建pose格式 [x, y, z, rx, ry, rz]
            pose = [tvec[0], tvec[1], tvec[2], euler_angles[0], euler_angles[1], euler_angles[2]]
            
            print(f"  📍 相机坐标系下位置: [{tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}] 米")
            print(f"  📐 相机坐标系下姿态: [{euler_angles[0]:.1f}°, {euler_angles[1]:.1f}°, {euler_angles[2]:.1f}°]")
            print(f"  T_cam2object:\n{T_cam2object}")
            
            # 在图像上绘制坐标轴
            # image_copy = color_image.copy()
            # cv2.drawFrameAxes(image_copy, camera_matrix, distortion_coefficients, rvec, tvec, 0.05)
            # cv2.imwrite(f'aruco_detected_id_{marker_id}.png', image_copy)
            # print(f"  已保存检测结果图像: aruco_detected_id_{marker_id}.png")
            
            # 只使用第一个检测到的标记
            break
    else:
        print("\n❌ 未检测到ArUco标记")     

    # 坐标变换: 相机坐标系 -> 基座坐标系
    if pose is not None and T_cam2object is not None:
        T_cam2tool = np.array([
            [ 0.06012576,  0.99535858, -0.07514113, -0.08242748],
            [-0.99807343,  0.06110251,  0.01076621,  0.00971775],
            [ 0.01530755,  0.07434904,  0.99711479, -0.16371493],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])  # 相机到工具坐标系的变换(通过手眼标定获得)
        
        T_tool2base = robot.get_pose_matrix()  # 获取当前机械臂末端位姿
        # T_base2tool = np.linalg.inv(T_tool2base)
        
        # T_cam2object 是物体在相机坐标系下的位姿矩阵 (4x4)
        # 变换链: T_base2object = T_base2tool @ T_tool2cam @ T_cam2object
        # T_tool2cam = np.linalg.inv(T_cam2tool)
        # T_base2object = T_base2tool @ T_tool2cam @ T_cam2object
        
        T_object2base = T_object2cam @ T_cam2tool @ T_tool2base
        
        # 提取基座坐标系下的位置
        position_base = T_object2base[:3, 3]
        
        # 提取基座坐标系下的姿态(欧拉角)
        from scipy.spatial.transform import Rotation as R
        rotation_base = R.from_matrix(T_object2base[:3, :3])
        euler_base = rotation_base.as_euler('xyz', degrees=True)
        
        print(f"\n🤖 基座坐标系下的目标位姿:")
        print(f"   位置: [{position_base[0]:.3f}, {position_base[1]:.3f}, {position_base[2]:.3f}] 米")
        print(f"   姿态: [{euler_base[0]:.1f}°, {euler_base[1]:.1f}°, {euler_base[2]:.1f}°]")
        
        # 构建目标位姿数组 [x, y, z, rx, ry, rz]
        target_pose = np.array([
            position_base[0], position_base[1], position_base[2],
            euler_base[0], euler_base[1], euler_base[2]
        ])
        
        # 移动到目标位置
        print(f"\n➡️  移动机械臂到目标位置...")
        joints= robot.r_inter.InverseSolution(*target_pose,1,1)
        print(joints)
        # robot.moveL(target_pose)
        print(f"✅ 移动完成!")
    else:
        print(f"\n❌ 未能获取有效的物体位姿，无法移动机械臂")    
    

if __name__ == "__main__":
    main()