"""
物体位姿估计模块
从掩码和深度图中提取物体的位姿信息
支持两种方法：
1. 平面检测法 - 适用于杯子、碗、盒子等有明显顶面的物体
2. PCA主轴法 - 适用于勺子、刀叉、笔等细长物体
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import cv2

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
    
    # 统计各范围的深度值 (放宽阈值)
    too_far = np.sum(depth_image > 10.0)  # 调整为10米
    too_close = np.sum((depth_image > 0) & (depth_image < 0.01))  # 调整为1cm
    valid_range = np.sum((depth_image >= 0.01) & (depth_image <= 10.0))
    
    
    valid_depth[depth_image > 10.0] = 0
    valid_depth[depth_image < 0.01] = 0
    
    valid_pixels = np.sum(valid_depth > 0)
    print(f"  过滤后有效像素: {valid_pixels}")
    
    if valid_pixels == 0:
        print(f"  ❌ 警告: 没有有效的深度数据!")
        print(f"  可能原因:")
        print(f"    1. 深度值全为0 (掩码区域没有深度信息)")
        print(f"    2. 深度值超出范围 (需要调整阈值)")
        print(f"    3. 深度图单位不对 (应该是米)")
    
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
    
    # print(f"  最终点云数量: {len(points)}")
    # if len(points) > 0:
    #     print(f"  点云范围: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
    #           f"Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
    #           f"Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
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


def extract_cup_side_contour(mask, color_image, depth_image, intrinsics):
    """
    从RGB图像中提取杯子的侧边轮廓并拟合成梯形
    
    参数:
        mask: 杯子掩码 (H, W)
        color_image: 彩色图像 (H, W, 3)
        depth_image: 深度图像 (H, W)
        intrinsics: 相机内参矩阵
    
    返回:
        trapezoid_points: 梯形四个顶点坐标 (4, 2)
        center_3d: 杯子中心3D坐标 (x, y, z)
        normal_3d: 杯子向上方向向量 (nx, ny, nz)
        success: 是否成功提取
    """
    try:
        print("🔍 开始提取杯子侧边轮廓...")
        
        # 1. 预处理掩码
        mask_2d = mask[:, :, 0] if len(mask.shape) == 3 else mask
        mask_2d = mask_2d.astype(np.uint8)
        
        # 保存原始掩码用于调试
        import os
        from datetime import datetime
        os.makedirs("result", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_mask_path = f"result/mask_original_{timestamp}.png"
        cv2.imwrite(original_mask_path, mask_2d*255)
        print(f"🔍 原始掩码已保存: {original_mask_path}")
        
        # 形态学操作：开运算去除噪点
        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_2d, cv2.MORPH_OPEN, kernel)
        
        # 保存清理后的掩码
        cleaned_mask_path = f"result/mask_cleaned_{timestamp}.png"
        cv2.imwrite(cleaned_mask_path, mask_cleaned*255)
        print(f"🔍 清理后掩码已保存: {cleaned_mask_path}")
        
        # 2. 查找轮廓
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("❌ 未找到轮廓")
            return None, None, None, False
        
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 3. 轮廓近似
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        print(f"   轮廓点数: {len(largest_contour)} -> 近似后: {len(approx_contour)}")
        
        # 4. 梯形拟合
        trapezoid_points = fit_trapezoid(approx_contour)
        
        if trapezoid_points is None:
            print("❌ 梯形拟合失败")
            return None, None, None, False
        
        print(f"   梯形顶点: {trapezoid_points}")
        
        # 5. 计算3D位姿
        center_3d, normal_3d = calculate_cup_pose_from_trapezoid(
            trapezoid_points, depth_image, intrinsics
        )
        
        if center_3d is None:
            print("❌ 3D位姿计算失败")
            return None, None, None, False
        
        print(f"✅ 杯子侧边轮廓提取成功")
        print(f"   梯形顶点: {trapezoid_points}")
        print(f"   3D中心: [{center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f}]")
        print(f"   向上向量: [{normal_3d[0]:.3f}, {normal_3d[1]:.3f}, {normal_3d[2]:.3f}]")
        
        return trapezoid_points, center_3d, normal_3d, True
        
    except Exception as e:
        print(f"❌ 提取杯子侧边轮廓时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, False


def fit_trapezoid(contour):
    """
    将轮廓拟合成梯形
    
    参数:
        contour: 近似轮廓点
    
    返回:
        trapezoid_points: 梯形四个顶点 (4, 2)，顺序为[左上, 右上, 右下, 左下]
    """
    try:
        if len(contour) < 4:
            print("❌ 轮廓点数太少，无法拟合梯形")
            return None
        
        # 获取轮廓的边界框
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 按y坐标排序，分为上下两组
        sorted_box = box[np.argsort(box[:, 1])]
        top_points = sorted_box[:2]  # 上边两点
        bottom_points = sorted_box[2:]  # 下边两点
        
        # 按x坐标排序
        top_points = top_points[np.argsort(top_points[:, 0])]  # [左, 右]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]  # [左, 右]
        
        # 构造梯形顶点 [左上, 右上, 右下, 左下]
        trapezoid_points = np.array([
            top_points[0],      # 左上
            top_points[1],      # 右上
            bottom_points[1],   # 右下
            bottom_points[0]    # 左下
        ])
        
        # 验证是否为有效梯形（上边比下边窄）
        top_width = np.linalg.norm(top_points[1] - top_points[0])
        bottom_width = np.linalg.norm(bottom_points[1] - bottom_points[0])
        
        if top_width >= bottom_width:
            print(f"⚠️ 警告: 检测到的形状上宽下窄，可能不是杯子 (上宽: {top_width:.1f}, 下宽: {bottom_width:.1f})")
            # 仍然返回，但标记为可能不准确
        
        print(f"   梯形尺寸: 上宽 {top_width:.1f}px, 下宽 {bottom_width:.1f}px")
        
        return trapezoid_points
        
    except Exception as e:
        print(f"❌ 梯形拟合失败: {e}")
        return None


def calculate_cup_pose_from_trapezoid(trapezoid_points, depth_image, intrinsics):
    """
    基于梯形几何特征计算杯子的3D位姿
    
    参数:
        trapezoid_points: 梯形四个顶点 (4, 2)
        depth_image: 深度图像 (H, W)
        intrinsics: 相机内参矩阵
    
    返回:
        center_3d: 杯子中心3D坐标 (x, y, z)
        normal_3d: 杯子向上方向向量 (nx, ny, nz)
    """
    try:
        # 1. 计算梯形的几何中心（2D）
        center_2d = np.mean(trapezoid_points, axis=0)
        
        # 2. 从深度图获取中心点的深度值
        center_x, center_y = int(center_2d[0]), int(center_2d[1])
        
        # 确保坐标在图像范围内
        h, w = depth_image.shape[:2]
        center_x = max(0, min(w-1, center_x))
        center_y = max(0, min(h-1, center_y))
        
        depth_value = depth_image[center_y, center_x]
        
        if depth_value <= 0:
            # 如果中心点没有深度值，尝试周围区域
            search_radius = 5
            valid_depths = []
            for dy in range(-search_radius, search_radius+1):
                for dx in range(-search_radius, search_radius+1):
                    ny, nx = center_y + dy, center_x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        d = depth_image[ny, nx]
                        if d > 0:
                            valid_depths.append(d)
            
            if len(valid_depths) > 0:
                depth_value = np.mean(valid_depths)
            else:
                print("❌ 无法获取有效的深度值")
                return None, None
        
        # 3. 将2D中心转换为3D坐标
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_3d = (center_x - cx) * depth_value / fx
        y_3d = (center_y - cy) * depth_value / fy
        z_3d = depth_value
        
        center_3d = np.array([x_3d, y_3d, z_3d])
        
        # 4. 计算杯子的向上方向向量
        # 基于梯形的对称轴方向
        top_mid = (trapezoid_points[0] + trapezoid_points[1]) / 2  # 上边中点
        bottom_mid = (trapezoid_points[2] + trapezoid_points[3]) / 2  # 下边中点
        
        # 对称轴方向（从上到下）
        symmetry_axis_2d = bottom_mid - top_mid
        symmetry_axis_2d = symmetry_axis_2d / np.linalg.norm(symmetry_axis_2d)
        
        # 将2D方向向量转换为3D方向向量
        # 假设杯子基本垂直放置，主要变化在XY平面
        normal_3d = np.array([symmetry_axis_2d[0], symmetry_axis_2d[1], 0])
        
        # 归一化并确保指向上方（z分量为正）
        if normal_3d[2] < 0:
            normal_3d = -normal_3d
        
        # 添加一些向上的分量，因为杯子通常是向上开口的
        normal_3d[2] = 0.3  # 给z分量一个正值
        normal_3d = normal_3d / np.linalg.norm(normal_3d)
        
        # 5. 计算杯子的X方向（水平方向）
        # 使用梯形的上边方向作为X轴参考
        top_edge = trapezoid_points[1] - trapezoid_points[0]  # 上边向量
        top_edge = top_edge / np.linalg.norm(top_edge)
        
        # 转换为3D
        x_direction_3d = np.array([top_edge[0], top_edge[1], 0])
        x_direction_3d = x_direction_3d / np.linalg.norm(x_direction_3d)
        
        print(f"   几何分析:")
        print(f"     对称轴方向: [{symmetry_axis_2d[0]:.3f}, {symmetry_axis_2d[1]:.3f}]")
        print(f"     上边方向: [{top_edge[0]:.3f}, {top_edge[1]:.3f}]")
        print(f"     3D向上向量: [{normal_3d[0]:.3f}, {normal_3d[1]:.3f}, {normal_3d[2]:.3f}]")
        print(f"     3D X方向: [{x_direction_3d[0]:.3f}, {x_direction_3d[1]:.3f}, {x_direction_3d[2]:.3f}]")
        
        return center_3d, normal_3d
        
    except Exception as e:
        print(f"❌ 3D位姿计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def vector_to_euler(direction_vector):
    """
    将方向向量转换为欧拉角（roll, pitch, yaw）
    
    参数:
        direction_vector: 方向向量 (3,)
    
    返回:
        roll, pitch, yaw (度)
    """
    # 归一化
    v = direction_vector / np.linalg.norm(direction_vector)
    
    # 计算pitch和yaw
    pitch = np.arcsin(-v[2])  # 俯仰角
    yaw = np.arctan2(v[1], v[0])  # 偏航角
    roll = 0  # 对于单个向量，roll角度无法唯一确定，设为0
    
    # 转换为度
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def extract_spoon_head_center(point_cloud, main_axis, centroid, head_ratio=0.50):
    """
    识别勺子圆头中心的位置和勺柄姿态
    
    参数:
        point_cloud: 勺子的点云数据
        main_axis: 主轴方向（已经指向勺头）
        centroid: 点云质心
        head_ratio: 勺头占整体长度的比例（默认0.50，即前50%）
    
    返回:
        head_center: 勺头圆形中心位置 (x, y, z)
        head_radius: 勺头半径估计
        handle_direction: 勺柄方向向量（从勺柄指向勺头）
        handle_pose: 勺柄姿态 [roll, pitch, yaw] (度)
    """
    try:
        points = np.asarray(point_cloud.points)
        
        # 1. 将点云投影到主轴上
        centered_points = points - centroid
        projections = centered_points @ main_axis
        
        # 2. 找到投影的最大值（勺头端）
        max_proj = np.max(projections)
        min_proj = np.min(projections)
        length = max_proj - min_proj
        
        # 3. 确定勺头区域的阈值（前端部分）
        head_threshold = max_proj - length * head_ratio
        
        # 4. 提取勺头区域的点云
        head_mask = projections >= head_threshold
        head_points = points[head_mask]
        
        if len(head_points) < 10:
            print("⚠️ 勺头点云数据太少")
            return None, None
        
        # 5. 计算勺头中心（勺头区域点云的质心）
        head_center = np.mean(head_points, axis=0)
        
        # 6. 估计勺头半径（在垂直于主轴的平面上）
        # 将勺头点投影到垂直于主轴的平面上
        head_centered = head_points - head_center
        # 去除主轴方向的分量
        perpendicular_components = head_centered - (head_centered @ main_axis)[:, np.newaxis] * main_axis
        # 计算垂直距离
        perpendicular_distances = np.linalg.norm(perpendicular_components, axis=1)
        head_radius = np.mean(perpendicular_distances)
        
        # 7. 计算勺柄方向和姿态
        # 勺柄方向就是主轴方向（从勺柄指向勺头）
        handle_direction = main_axis
        
        # 将勺柄方向转换为欧拉角
        handle_roll, handle_pitch, handle_yaw = vector_to_euler(handle_direction)
        handle_pose = [handle_roll, handle_pitch, handle_yaw]
        
        print(f"🥄 勺子特征提取:")
        print(f"   勺头中心: [{head_center[0]:.3f}, {head_center[1]:.3f}, {head_center[2]:.3f}]")
        print(f"   勺头半径: {head_radius:.3f}m ({head_radius*100:.1f}cm)")
        print(f"   勺头点数: {len(head_points)} / {len(points)} ({len(head_points)/len(points)*100:.1f}%)")
        print(f"   勺柄方向: [{handle_direction[0]:.3f}, {handle_direction[1]:.3f}, {handle_direction[2]:.3f}]")
        print(f"   勺柄姿态: [roll={handle_roll:.1f}°, pitch={handle_pitch:.1f}°, yaw={handle_yaw:.1f}°]")
        
        return head_center, head_radius, handle_direction, handle_pose
        
    except Exception as e:
        print(f"❌ 提取勺头特征时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def extract_elongated_features(point_cloud):
    """
    使用PCA提取细长物体（如勺子）的几何特征
    
    参数:
        point_cloud: 物体的点云数据
    
    返回:
        center: 物体中心位置 (x, y, z)
        main_axis: 主轴方向（细长方向）
        secondary_axis: 次要轴方向
        length: 主轴长度
    """
    try:
        points = np.asarray(point_cloud.points)
        
        if len(points) < 10:
            print("⚠️ 点云数据太少")
            return None, None, None, None
        
        # 1. 先计算质心（用于PCA分析）
        centroid = np.mean(points, axis=0)
        
        # 2. 中心化点云
        centered_points = points - centroid
        
        # 3. 计算协方差矩阵
        cov_matrix = np.cov(centered_points.T)
        
        # 4. 特征值分解（PCA）
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 5. 按特征值排序（从大到小）
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 6. 提取主轴（最大特征值对应的特征向量）
        main_axis = eigenvectors[:, 0]  # 第一主成分（细长方向）
        secondary_axis = eigenvectors[:, 1]  # 第二主成分
        
        # 7. 确定主轴方向（解决方向歧义问题）
        # 策略：让主轴指向更宽的一端（通常是勺头）
        # 使用多维度宽度测量，更鲁棒
        projections = centered_points @ main_axis
        
        # 将点分为两组：正投影和负投影
        positive_mask = projections > 0
        negative_mask = projections < 0
        
        if np.sum(positive_mask) > 0 and np.sum(negative_mask) > 0:
            # 计算两端的点云
            positive_points = centered_points[positive_mask]
            negative_points = centered_points[negative_mask]
            
            # 方法1: 在次要轴方向上的散布（标准差）
            positive_width_secondary = np.std(positive_points @ secondary_axis)
            negative_width_secondary = np.std(negative_points @ secondary_axis)
            
            # 方法2: 在第三轴方向上的散布
            third_axis = eigenvectors[:, 2]
            positive_width_third = np.std(positive_points @ third_axis)
            negative_width_third = np.std(negative_points @ third_axis)
            
            # 方法3: 整体横截面积估计（垂直于主轴的总散布）
            positive_cross_section = np.sqrt(
                np.var(positive_points @ secondary_axis) + 
                np.var(positive_points @ third_axis)
            )
            negative_cross_section = np.sqrt(
                np.var(negative_points @ secondary_axis) + 
                np.var(negative_points @ third_axis)
            )
            
            # 综合评分：多维度宽度平均（不使用密度，避免距离影响）
            positive_score = (positive_width_secondary + positive_width_third + 
                            positive_cross_section * 0.5) / 2.5
            negative_score = (negative_width_secondary + negative_width_third + 
                            negative_cross_section * 0.5) / 2.5
            
            print(f"   方向判断（基于几何宽度，不受距离影响）:")
            print(f"     正向端 - 次轴宽度: {positive_width_secondary:.4f}, "
                  f"第三轴宽度: {positive_width_third:.4f}, "
                  f"横截面: {positive_cross_section:.4f}, 综合分数: {positive_score:.4f}")
            print(f"     负向端 - 次轴宽度: {negative_width_secondary:.4f}, "
                  f"第三轴宽度: {negative_width_third:.4f}, "
                  f"横截面: {negative_cross_section:.4f}, 综合分数: {negative_score:.4f}")
            
            # 如果负向端分数更高，翻转主轴方向
            if negative_score > positive_score:
                main_axis = -main_axis
                print(f"   ✓ 主轴翻转，指向更宽的一端（勺头）")
            else:
                print(f"   ✓ 主轴方向保持，已指向更宽的一端（勺头）")
        
        # 8. 估计物体长度（沿主轴的范围）
        projections = centered_points @ main_axis  # 重新计算投影
        length = np.max(projections) - np.min(projections)
        
        # 9. 计算真正的中心点：沿主轴的几何中点（不是质心）
        # 这样中心点在勺子的中部，更方便抓取
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        mid_proj = (min_proj + max_proj) / 2.0  # 主轴上的中点投影值
        
        # 将中点投影值转换回3D空间坐标
        center = centroid + mid_proj * main_axis
        
        # 10. 计算特征值比率（用于判断是否真的是细长物体）
        ratio_1_2 = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 0
        ratio_1_3 = eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 0 else 0
        
        print(f"🔍 细长物体特征提取 (PCA):")
        print(f"   质心位置: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}] (用于PCA)")
        print(f"   中心位置: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] (主轴几何中点，抓取点)")
        print(f"   主轴方向: [{main_axis[0]:.3f}, {main_axis[1]:.3f}, {main_axis[2]:.3f}]")
        print(f"   估计长度: {length:.3f}m ({length*100:.1f}cm)")
        print(f"   特征值比率: {ratio_1_2:.2f} (主/次), {ratio_1_3:.2f} (主/第三)")
        
        # 判断是否是细长物体（主特征值明显大于其他）
        if ratio_1_2 < 2.0:
            print("⚠️ 警告: 物体可能不是细长形状")
        
        return center, main_axis, secondary_axis, length
        
    except Exception as e:
        print(f"❌ 提取细长物体特征时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def calculate_cup_pose(center, normal):
    """
    根据杯子中心和法向量计算位姿变换矩阵（平面检测法）
    
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


def calculate_cup_pose_from_trapezoid_matrix(center_3d, normal_3d, trapezoid_points):
    """
    基于梯形几何特征计算杯子的完整位姿变换矩阵
    
    参数:
        center_3d: 杯子中心3D坐标 (x, y, z)
        normal_3d: 杯子向上方向向量 (nx, ny, nz)
        trapezoid_points: 梯形四个顶点 (4, 2)
    
    返回:
        4x4变换矩阵，其中：
        - Z轴：杯子向上方向（normal_3d）
        - X轴：杯子水平方向（基于梯形上边）
        - Y轴：通过叉乘得到
    """
    try:
        # Z轴：杯子向上方向
        z_axis = normal_3d / np.linalg.norm(normal_3d)
        
        # X轴：基于梯形上边的水平方向
        top_edge = trapezoid_points[1] - trapezoid_points[0]  # 上边向量
        top_edge = top_edge / np.linalg.norm(top_edge)
        
        # 将2D上边方向转换为3D
        x_axis = np.array([top_edge[0], top_edge[1], 0])
        
        # 确保X轴与Z轴正交
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y轴：通过叉乘得到
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3, 0] = x_axis
        T[:3, 1] = y_axis
        T[:3, 2] = z_axis
        T[:3, 3] = center_3d
        
        print(f"   变换矩阵构建:")
        print(f"     X轴 (水平): [{x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}]")
        print(f"     Y轴 (侧向): [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}]")
        print(f"     Z轴 (向上): [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
        
        return T
        
    except Exception as e:
        print(f"❌ 构建变换矩阵失败: {e}")
        return None


def calculate_elongated_pose(center, main_axis, secondary_axis):
    """
    根据主轴方向计算细长物体的位姿变换矩阵（PCA法）
    
    参数:
        center: 物体中心位置 (x, y, z)
        main_axis: 主轴方向（细长方向）
        secondary_axis: 次要轴方向
    
    返回:
        4x4变换矩阵
    """
    # Y轴：主轴方向（勺子的长度方向）
    y_axis = main_axis / np.linalg.norm(main_axis)
    
    # X轴：次要轴方向
    x_axis = secondary_axis / np.linalg.norm(secondary_axis)
    
    # 确保X轴与Y轴正交
    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Z轴：通过叉乘得到（垂直于XY平面）
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
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


def mask2pose(mask, depth_image, color_image, intrinsics, T_cam2base=None, object_class="cup"):
    """
    从掩码、深度图和彩色图中估计物体的位姿
    根据物体类别自动选择合适的估计方法
    
    参数:
        mask (numpy.ndarray): 物体掩码，形状为(H, W)，值为0或1
        depth_image (numpy.ndarray): 深度图像，形状为(H, W)，单位为米
        color_image (numpy.ndarray): 彩色图像，形状为(H, W, 3)
        intrinsics (numpy.ndarray): 3x3相机内参矩阵
        T_cam2base (numpy.ndarray): 4x4相机到基坐标系的变换矩阵（可选）
        object_class (str): 物体类别，用于选择合适的估计方法
    
    返回:
        pose: [x, y, z, roll, pitch, yaw] 在基坐标系中的位姿
        T: 4x4变换矩阵
    # 得到pose基于相机的座标
    """
    try:
        # 0. 确保mask尺寸与图像匹配(双保险)
        h, w = color_image.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        
        # 保存掩码用于调试
        import os
        from datetime import datetime
        os.makedirs("result", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_path = f"result/mask_debug_{timestamp}.png"
        cv2.imwrite(mask_path, mask*255)
        print(f"🔍 掩码已保存用于调试: {mask_path}")
        
        
        if (mask_h, mask_w) != (h, w):
            print(f"  ⚠️ [mask2pose] 调整mask尺寸: ({mask_h}, {mask_w}) -> ({h}, {w})")
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)         
        
        # 1. 根据掩码提取点云
        # 确保mask是2D的 (H, W)
        mask_2d = mask[:, :, 0] if len(mask.shape) == 3 else mask
        # 确保depth是2D的 (H, W)
        depth_2d = depth_image[:, :, 0] if len(depth_image.shape) == 3 else depth_image
        

        
        # 应用掩码
        color_masked = color_image * mask_2d[:, :, np.newaxis]
        depth_masked = depth_2d * mask_2d
        
        # 检查掩码区域的深度值
        masked_depth_values = depth_masked[mask_2d > 0]
        if len(masked_depth_values) > 0:
            print(f"  掩码区域深度值: [{masked_depth_values.min():.3f}, {masked_depth_values.max():.3f}]")
            print(f"  掩码区域平均深度: {masked_depth_values.mean():.3f}m")
            print(f"  掩码区域有效深度点数: {np.sum((masked_depth_values > 0.1) & (masked_depth_values < 3.5))}")
        else:
            print(f"  ❌ 警告: 掩码区域没有像素!")


        
        # 保存深度掩码图(用于调试)
        if depth_masked.max() > 0:
            depth_vis = cv2.normalize(depth_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
  
        point_cloud = create_point_cloud(depth_masked, intrinsics, color_masked)
        
        # o3d.visualization.draw_geometries([point_cloud], window_name="Masked Point Cloud")
        
        if len(point_cloud.points) < 50:
            print("❌ 点云数据太少，无法估计位姿")
            return None, None
        
        # 2. 如果提供了相机到基坐标系的变换，先转换到基坐标系
        if T_cam2base is not None:
            point_cloud.transform(T_cam2base)
        
        # 3. 根据物体类别选择合适的方法
        # 细长物体列表：勺子、刀、叉、筷子、笔等
        elongated_objects = ['spoon', 'knife', 'fork', 'chopstick', 'pen', 'pencil', 
                            'ruler', 'screwdriver', 'brush', 'toothbrush']
        
        # 平面物体列表：杯子、碗、盒子、瓶子等
        planar_objects = ['cup', 'bowl', 'box', 'bottle', 'mug', 'glass', 
                         'container', 'plate', 'dish']
        
        if object_class.lower() in elongated_objects:
            print(f"📏 检测到细长物体 '{object_class}'，使用PCA主轴法")
            # 使用PCA方法
            center, main_axis, secondary_axis, length = extract_elongated_features(point_cloud)
            
            if center is None:
                return None, None
            
            # 如果是勺子，额外提取勺头中心和勺柄姿态
            extra_info = {}
            if object_class.lower() == 'spoon':
                # 获取质心用于勺头提取
                points = np.asarray(point_cloud.points)
                centroid = np.mean(points, axis=0)
                
                head_center, head_radius, handle_direction, handle_pose = extract_spoon_head_center(
                    point_cloud, main_axis, centroid, head_ratio=0.30
                )
                
                if head_center is not None:
                    extra_info['spoon_head_center'] = head_center
                    extra_info['spoon_head_radius'] = head_radius
                    extra_info['handle_direction'] = handle_direction
                    extra_info['handle_pose'] = handle_pose  # [roll, pitch, yaw]
            
            # 计算位姿变换矩阵
            T = calculate_elongated_pose(center, main_axis, secondary_axis)
            
            # 转换为位置和欧拉角
            pose = transform_matrix_to_pos_euler(T)
            
            print(f"✅ {object_class}位姿估计成功:")
            print(f"   位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
            print(f"   姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
            
            # 如果有勺头信息，添加到pose中
            if extra_info:
                pose = list(pose)  # 转换为列表以便添加额外信息
                pose.append(extra_info)  # 将额外信息作为第7个元素
            
        else:
            print(f"🔲 检测到平面物体 '{object_class}'，使用梯形轮廓法")
            
            # 优先使用新的梯形轮廓方法
            if object_class.lower() in ['cup', 'mug', 'glass']:
                print("   尝试使用梯形轮廓方法...")
                trapezoid_points, center_3d, normal_3d, success = extract_cup_side_contour(
                    mask, color_image, depth_2d, intrinsics
                )
                
                if success and center_3d is not None:
                    print("✅ 梯形轮廓方法成功")
                    # 计算位姿变换矩阵
                    T = calculate_cup_pose_from_trapezoid_matrix(center_3d, normal_3d, trapezoid_points)
                    
                    if T is not None:
                        # 转换为位置和欧拉角
                        pose = transform_matrix_to_pos_euler(T)
                        
                        print(f"✅ {object_class}位姿估计成功 (梯形轮廓法):")
                        print(f"   位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
                        print(f"   姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
                        
                        return pose, T
                    else:
                        print("⚠️ 梯形轮廓法计算变换矩阵失败，回退到平面检测法")
                else:
                    print("⚠️ 梯形轮廓法失败，回退到平面检测法")
            
            # 回退到传统的平面检测方法
            print("   使用传统平面检测法...")
            center, normal, radius = extract_cup_features(point_cloud)
            
            if center is None:
                return None, None
            
            # 计算位姿变换矩阵
            T = calculate_cup_pose(center, normal)
            
            # 转换为位置和欧拉角
            pose = transform_matrix_to_pos_euler(T)
            
            print(f"✅ {object_class}位姿估计成功 (平面检测法):")
            print(f"   位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
            print(f"   姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
        
        return pose, T
        
    except Exception as e:
        print(f"❌ 位姿估计失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def draw_pose_axes(image, intrinsics, pose_matrix, axis_length=0.05):
    # 提取旋转和平移
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]
    # 转为OpenCV格式
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    # 绘制坐标轴
    cv2.drawFrameAxes(image, intrinsics, np.zeros(5), rvec, tvec, axis_length)
    cv2.imshow("Pose Visualization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def visualize_result(color_image, depth_image, T_cam2base, intrinsics, pose):
    """
    可视化检测结果
    
    参数:
        color_image: 彩色图像
        depth_image: 深度图像
        T_cam2base: 相机到基坐标系的变换
        intrinsics: 相机内参
        pose: 物体位姿 [x, y, z, roll, pitch, yaw] 或包含额外信息的列表
    """
    try:
        # 创建完整点云
        pcd = create_point_cloud(depth_image, intrinsics, color_image)
        if T_cam2base is not None:
            pcd.transform(T_cam2base)
        
        # 创建坐标系
        pose_matrix = np.eye(4)
        pose_matrix[:3, 3] = pose[:3]
        # 只使用前6个元素的后3个（roll, pitch, yaw）
        r = R.from_euler('xyz', pose[3:6], degrees=True)
        pose_matrix[:3, :3] = r.as_matrix()
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coordinate_frame.transform(pose_matrix)
        
        geometries = [pcd, coordinate_frame]
        
        # 检查是否有勺头中心信息
        if isinstance(pose, list) and len(pose) > 6 and isinstance(pose[6], dict):
            extra_info = pose[6]
            if 'spoon_head_center' in extra_info:
                head_center = extra_info['spoon_head_center']
                head_radius = extra_info['spoon_head_radius']
                
                # 创建球体标记勺头中心（橙色）
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=head_radius * 0.5)
                sphere.translate(head_center)
                sphere.paint_uniform_color([1.0, 0.5, 0.0])  # 橙色
                geometries.append(sphere)
                
                print(f"🥄 勺头中心标记: 橙色球体")
        
        # 可视化
        o3d.visualization.draw_geometries(geometries)
        
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
            {'class': str, 'pose': [x, y, z, roll, pitch, yaw], 'confidence': float, 'extra_info': dict},
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
            # 只使用前6个元素的后3个（roll, pitch, yaw）
            r = R.from_euler('xyz', pose[3:6], degrees=True)
            pose_matrix[:3, :3] = r.as_matrix()
            
            # 创建坐标系（大小根据物体索引略有变化）
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.08 + idx * 0.02
            )
            coordinate_frame.transform(pose_matrix)
            
            # 添加到几何体列表
            geometries.append(coordinate_frame)
            
            print(f"  {idx+1}. {obj_class}: 坐标系大小 {0.08 + idx * 0.02:.2f}m")
            
            # 如果有勺头中心信息，创建橙色球体标记
            if 'extra_info' in pose_info and pose_info['extra_info']:
                extra = pose_info['extra_info']
                if 'spoon_head_center' in extra:
                    head_center = extra['spoon_head_center']
                    head_radius = extra['spoon_head_radius']
                    
                    # 创建球体标记勺头中心（橙色）
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=head_radius * 0.5)
                    sphere.translate(head_center)
                    sphere.paint_uniform_color([1.0, 0.5, 0.0])  # 橙色
                    geometries.append(sphere)
                    
                    print(f"       -> 勺头中心标记: 橙色球体")
        
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
