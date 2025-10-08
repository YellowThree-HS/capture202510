"""
杯子位姿估计模块
从掩码和深度图中提取杯子的位姿信息
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def create_point_cloud(depth_image, intrinsics, color_image):
    """
    根据深度图像和相机内参，创建点云数据，并结合彩色图像为点云上色
    
    参数:
        depth_image (numpy.ndarray): 深度图像数据
        intrinsics (numpy.ndarray): 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        color_image (numpy.ndarray): 彩色图像数据
    
    返回:
        pcd (open3d.geometry.PointCloud): 创建的点云对象
    """
    if depth_image.shape[:2] != color_image.shape[:2]:
        raise ValueError("深度图和彩色图尺寸必须一致")
    
    height, width = depth_image.shape[:2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 过滤无效深度值
    valid_depth = depth_image.copy().astype(float)
    valid_depth[depth_image > 3.5] = 0
    valid_depth[depth_image < 0.1] = 0
    
    # 计算3D坐标
    z = valid_depth
    x = (u - intrinsics[0][2]) * z / intrinsics[0][0]
    y = (v - intrinsics[1][2]) * z / intrinsics[1][1]
    
    # 组合点云坐标和颜色
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image[..., ::-1].reshape(-1, 3) / 255.0
    
    # 过滤无效点
    valid_mask = (z.reshape(-1) > 0)
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def extract_cup_features(point_cloud):
    """
    提取杯子的几何特征（顶面和中心）
    
    参数:
        point_cloud: 杯子的点云数据
    
    返回:
        center: 杯子中心位置 (x, y, z)
        normal: 杯子顶面法向量 (nx, ny, nz)
        radius: 杯口半径（可选）
    """
    try:
        # 1. 使用RANSAC检测平面（杯子顶面）
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=0.005,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) < 10:
            print("⚠️ 检测到的平面点太少")
            return None, None, None
        
        # 2. 提取顶面点云
        top_surface = point_cloud.select_by_index(inliers)
        top_points = np.asarray(top_surface.points)
        
        # 3. 计算杯子中心（顶面点云的几何中心）
        center = np.mean(top_points, axis=0)
        
        # 4. 提取法向量（指向上方）
        normal = -plane_model[:3] / np.linalg.norm(plane_model[:3])
        
        # 确保法向量指向上方（z方向为正）
        if normal[2] < 0:
            normal = -normal
        
        # 5. 估计杯口半径（可选）
        distances = np.linalg.norm(top_points - center, axis=1)
        radius = np.mean(distances)
        
        print(f"🔍 杯子特征提取:")
        print(f"   中心位置: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        print(f"   法向量: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
        print(f"   估计半径: {radius:.3f}m ({radius*100:.1f}cm)")
        
        return center, normal, radius
        
    except Exception as e:
        print(f"❌ 提取杯子特征时出错: {e}")
        return None, None, None


def calculate_cup_pose(center, normal):
    """
    根据杯子中心和法向量计算位姿变换矩阵
    
    参数:
        center: 杯子中心位置 (x, y, z)
        normal: 杯子顶面法向量 (nx, ny, nz)
    
    返回:
        4x4变换矩阵
    """
    # Z轴：法向量方向
    z_axis = normal / np.linalg.norm(normal)
    
    # X轴：选择一个与z轴垂直的方向
    # 如果z轴接近竖直，选择[1,0,0]作为参考
    if abs(z_axis[2]) > 0.9:
        x_axis = np.cross(z_axis, np.array([0, 1, 0]))
    else:
        x_axis = np.cross(z_axis, np.array([0, 0, 1]))
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y轴：通过叉乘得到
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 构建变换矩阵
    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = center
    
    return T


def transform_matrix_to_pos_euler(T):
    """
    将变换矩阵转换为位置和欧拉角
    
    参数:
        T: 4x4变换矩阵
    
    返回:
        [x, y, z, roll, pitch, yaw] (位置单位：米，角度单位：度)
    """
    x, y, z = T[:3, 3]
    rotation_matrix = T[:3, :3]
    
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=True)
    
    return [x, y, z] + list(euler_angles)


def mask2pose(mask, depth_image, color_image, intrinsics, T_cam2base=None):
    """
    从掩码、深度图和彩色图中估计杯子的位姿
    
    参数:
        mask (numpy.ndarray): 物体掩码，形状为(H, W)，值为0或1
        depth_image (numpy.ndarray): 深度图像，形状为(H, W)，单位为米
        color_image (numpy.ndarray): 彩色图像，形状为(H, W, 3)
        intrinsics (numpy.ndarray): 3x3相机内参矩阵
        T_cam2base (numpy.ndarray): 4x4相机到基坐标系的变换矩阵（可选）
    
    返回:
        pose: [x, y, z, roll, pitch, yaw] 在基坐标系中的位姿
        T: 4x4变换矩阵
    """
    try:
        # 1. 根据掩码提取点云
        color_masked = color_image * mask[:, :, np.newaxis]
        depth_masked = depth_image * mask
        
        point_cloud = create_point_cloud(depth_masked, intrinsics, color_masked)
        
        if len(point_cloud.points) < 50:
            print("❌ 点云数据太少，无法估计位姿")
            return None, None
        
        # 2. 如果提供了相机到基坐标系的变换，先转换到基坐标系
        if T_cam2base is not None:
            point_cloud.transform(T_cam2base)
        
        # 3. 提取杯子特征
        center, normal, radius = extract_cup_features(point_cloud)
        
        if center is None:
            return None, None
        
        # 4. 计算位姿变换矩阵
        T = calculate_cup_pose(center, normal)
        
        # 5. 转换为位置和欧拉角
        pose = transform_matrix_to_pos_euler(T)
        
        print(f"✅ 杯子位姿估计成功:")
        print(f"   位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
        print(f"   姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
        
        return pose, T
        
    except Exception as e:
        print(f"❌ 位姿估计失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def visualize_result(color_image, depth_image, T_cam2base, intrinsics, pose):
    """
    可视化检测结果
    
    参数:
        color_image: 彩色图像
        depth_image: 深度图像
        T_cam2base: 相机到基坐标系的变换
        intrinsics: 相机内参
        pose: 物体位姿 [x, y, z, roll, pitch, yaw]
    """
    try:
        # 创建完整点云
        pcd = create_point_cloud(depth_image, intrinsics, color_image)
        if T_cam2base is not None:
            pcd.transform(T_cam2base)
        
        # 创建坐标系
        pose_matrix = np.eye(4)
        pose_matrix[:3, 3] = pose[:3]
        r = R.from_euler('xyz', pose[3:], degrees=True)
        pose_matrix[:3, :3] = r.as_matrix()
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coordinate_frame.transform(pose_matrix)
        
        # 可视化
        o3d.visualization.draw_geometries([pcd, coordinate_frame])
        
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")


def visualize_multi_objects(color_image, depth_image, T_cam2base, intrinsics, poses_info):
    """
    可视化多个物体的检测结果
    
    参数:
        color_image: 彩色图像
        depth_image: 深度图像
        T_cam2base: 相机到基坐标系的变换
        intrinsics: 相机内参
        poses_info: 物体位姿信息列表 [
            {'class': str, 'pose': [x, y, z, roll, pitch, yaw], 'confidence': float},
            ...
        ]
    """
    try:
        # 创建完整点云
        pcd = create_point_cloud(depth_image, intrinsics, color_image)
        if T_cam2base is not None:
            pcd.transform(T_cam2base)
        
        # 为每个物体创建坐标系
        geometries = [pcd]
        
        # 为不同类别定义不同颜色的坐标系
        colors = [
            [1, 0, 0],  # 红色
            [0, 1, 0],  # 绿色
            [0, 0, 1],  # 蓝色
            [1, 1, 0],  # 黄色
            [1, 0, 1],  # 紫色
            [0, 1, 1],  # 青色
        ]
        
        print(f"\n🎨 创建 {len(poses_info)} 个物体的坐标系...")
        
        for idx, pose_info in enumerate(poses_info):
            pose = pose_info['pose']
            obj_class = pose_info['class']
            
            # 创建位姿变换矩阵
            pose_matrix = np.eye(4)
            pose_matrix[:3, 3] = pose[:3]
            r = R.from_euler('xyz', pose[3:], degrees=True)
            pose_matrix[:3, :3] = r.as_matrix()
            
            # 创建坐标系（大小根据物体索引略有变化）
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.08 + idx * 0.02
            )
            coordinate_frame.transform(pose_matrix)
            
            # 添加到几何体列表
            geometries.append(coordinate_frame)
            
            print(f"  {idx+1}. {obj_class}: 坐标系大小 {0.08 + idx * 0.02:.2f}m")
        
        print("\n💡 可视化说明:")
        print("  - 白色点云: 场景")
        for idx, pose_info in enumerate(poses_info):
            print(f"  - 坐标系 {idx+1}: {pose_info['class']}")
        print("  - X轴(红), Y轴(绿), Z轴(蓝)")
        
        # 可视化所有几何体
        o3d.visualization.draw_geometries(geometries)
        
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
