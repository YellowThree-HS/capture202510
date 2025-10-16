"""
物体位姿估计模块
从掩码和深度图中提取物体的位姿信息

主要功能：
1. 梯形轮廓法 - 专门用于杯子的侧边轮廓分析
2. PCA主轴法 - 适用于勺子、刀叉、笔等细长物体

核心函数：
- mask2pose(): 主要接口，根据物体类别自动选择估计方法
- create_point_cloud(): 从深度图创建点云
- extract_elongated_features(): 细长物体特征提取（PCA法）
- extract_cup_side_contour(): 杯子侧边轮廓提取（梯形轮廓法）
- calculate_cup_pose_from_trapezoid(): 基于梯形计算杯子3D位姿
- calculate_cup_pose_from_trapezoid_matrix(): 构建杯子位姿变换矩阵
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import cv2
import os
from datetime import datetime

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
    valid_depth[depth_image > 10.0] = 0
    valid_depth[depth_image < 0.01] = 0
    
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
        # 1. 预处理掩码
        mask_2d = mask[:, :, 0] if len(mask.shape) == 3 else mask
        mask_2d = mask_2d.astype(np.uint8)
        
        # 形态学操作：开运算去除噪点
        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_2d, cv2.MORPH_OPEN, kernel)
        
        # 只保留最大的连通区域（杯子主体）
        mask_cleaned = keep_largest_connected_component(mask_cleaned)
        
        # 2. 查找轮廓
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, None, None, False
        
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 3. 轮廓近似
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 4. 梯形拟合
        trapezoid_points = fit_trapezoid(approx_contour)
        
        if trapezoid_points is None:
            return None, None, None, False
        
        # 4.5. 可视化梯形拟合结果
        visualize_trapezoid_fitting(color_image, mask_cleaned, largest_contour, 
                                  approx_contour, trapezoid_points)
        
        # 5. 计算3D位姿
        center_3d, normal_3d = calculate_cup_pose_from_trapezoid(
            trapezoid_points, depth_image, intrinsics, color_image
        )
        
        if center_3d is None:
            return None, None, None, False
        
        return trapezoid_points, center_3d, normal_3d, True
        
    except Exception as e:
        return None, None, None, False




def visualize_trapezoid_fitting(color_image, mask_cleaned, original_contour, 
                               approx_contour, trapezoid_points):
    """
    可视化梯形拟合过程
    
    参数:
        color_image: 原始彩色图像
        mask_cleaned: 清理后的掩码
        original_contour: 原始轮廓
        approx_contour: 近似轮廓
        trapezoid_points: 梯形四个顶点
    """
    try:
        # 创建可视化图像
        vis_image = color_image.copy()
        
        # 1. 绘制清理后的掩码轮廓（绿色）
        cv2.drawContours(vis_image, [original_contour], -1, (0, 255, 0), 2)
        
        # 2. 绘制近似轮廓（蓝色）
        cv2.drawContours(vis_image, [approx_contour], -1, (255, 0, 0), 2)
        
        # 3. 绘制梯形（红色）
        if trapezoid_points is not None:
            # 将梯形顶点转换为整数
            trapezoid_int = np.array(trapezoid_points, dtype=np.int32)
            
            # 绘制梯形边
            for i in range(4):
                pt1 = tuple(trapezoid_int[i])
                pt2 = tuple(trapezoid_int[(i + 1) % 4])
                cv2.line(vis_image, pt1, pt2, (0, 0, 255), 3)
            
            # 绘制梯形顶点
            for i, point in enumerate(trapezoid_int):
                cv2.circle(vis_image, tuple(point), 5, (0, 0, 255), -1)
                # 标注顶点编号
                cv2.putText(vis_image, str(i), 
                           (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 4. 添加图例
        legend_y = 30
        cv2.putText(vis_image, "Green: Original Contour", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_image, "Blue: Approximated Contour", (10, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(vis_image, "Red: Fitted Trapezoid", (10, legend_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 5. 保存可视化结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        filename = os.path.join(result_dir, f"trapezoid_fitting_{timestamp}.jpg")
        cv2.imwrite(filename, vis_image)
        
        print(f"梯形拟合可视化结果已保存到: {filename}")
        
        # 6. 打印梯形顶点坐标
        if trapezoid_points is not None:
            print("梯形顶点坐标:")
            vertex_names = ["左上", "右上", "右下", "左下"]
            for i, (name, point) in enumerate(zip(vertex_names, trapezoid_points)):
                print(f"  {name}: ({point[0]:.1f}, {point[1]:.1f})")
        
    except Exception as e:
        print(f"可视化梯形拟合时出错: {e}")


def keep_largest_connected_component(mask):
    """
    只保留掩码中最大的连通区域，去除小的离群点
    
    参数:
        mask: 二值掩码图像 (H, W)
    
    返回:
        cleaned_mask: 只包含最大连通区域的掩码
    """
    try:
        # 确保掩码是二值的
        mask_binary = (mask > 0).astype(np.uint8)
        
        # 查找所有连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        
        if num_labels <= 1:  # 没有前景区域或只有一个区域
            return mask
        
        # 找到面积最大的区域（排除背景，索引0是背景）
        areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
        largest_component_idx = np.argmax(areas) + 1  # +1因为索引0是背景
        
        # 创建只包含最大连通区域的掩码
        cleaned_mask = (labels == largest_component_idx).astype(np.uint8)
        
        return cleaned_mask
        
    except Exception as e:
        return mask




def fit_trapezoid(contour):
    """
    将轮廓拟合成梯形 - 改进版本，上边取最左和最右的点
    
    参数:
        contour: 近似轮廓点
    
    返回:
        trapezoid_points: 梯形四个顶点 (4, 2)，顺序为[左上, 右上, 右下, 左下]
    """
    try:
        if len(contour) < 4:
            return None
        
        # 将轮廓点转换为numpy数组
        contour_points = contour.reshape(-1, 2)
        
        # 计算轮廓的y坐标范围
        y_min, y_max = np.min(contour_points[:, 1]), np.max(contour_points[:, 1])
        height = y_max - y_min
        
        # 定义上边和下边的区域（各占高度的30%）
        top_threshold = y_min + height * 0.3
        bottom_threshold = y_max - height * 0.3
        
        # 提取上边区域的点
        top_mask = contour_points[:, 1] <= top_threshold
        top_points = contour_points[top_mask]
        
        # 提取下边区域的点
        bottom_mask = contour_points[:, 1] >= bottom_threshold
        bottom_points = contour_points[bottom_mask]
        
        if len(top_points) == 0 or len(bottom_points) == 0:
            # 如果无法分为上下两部分，使用原来的方法作为备用
            return fit_trapezoid_fallback(contour)
        
        # 上边：取最左边和最右边的点
        top_left = top_points[np.argmin(top_points[:, 0])]
        top_right = top_points[np.argmax(top_points[:, 0])]
        
        # 下边：取最左边和最右边的点
        bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
        bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]
        
        # 构造梯形顶点 [左上, 右上, 右下, 左下]
        trapezoid_points = np.array([
            top_left,      # 左上
            top_right,     # 右上
            bottom_right,  # 右下
            bottom_left    # 左下
        ])
        
        return trapezoid_points
        
    except Exception as e:
        # 如果新方法失败，使用原来的方法作为备用
        return fit_trapezoid_fallback(contour)


def fit_trapezoid_fallback(contour):
    """
    梯形拟合的备用方法（原来的方法）
    """
    try:
        if len(contour) < 4:
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
        
        return trapezoid_points
        
    except Exception as e:
        return None




def calculate_cup_pose_from_trapezoid(trapezoid_points, depth_image, intrinsics, color_image=None):
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
        
        # 2. 使用多点采样获取更鲁棒的深度值
        center_x, center_y = int(center_2d[0]), int(center_2d[1])
        
        # 确保坐标在图像范围内
        h, w = depth_image.shape[:2]
        center_x = max(0, min(w-1, center_x))
        center_y = max(0, min(h-1, center_y))
        
        # 在中心点周围生成均匀分布的点（网格采样）
        sample_points = []
        sample_radius = 8
        
        # 创建3x3网格采样点
        for dx in [-sample_radius, 0, sample_radius]:
            for dy in [-sample_radius, 0, sample_radius]:
                if dx == 0 and dy == 0:
                    continue
                
                sample_x = center_x + dx
                sample_y = center_y + dy
                
                # 确保采样点在图像范围内
                sample_x = max(0, min(w-1, sample_x))
                sample_y = max(0, min(h-1, sample_y))
                
                sample_points.append((sample_x, sample_y))
        
        # 添加几何中心点
        sample_points.append((center_x, center_y))
        
        # 添加更多采样点以提高鲁棒性
        additional_offsets = [
            (sample_radius//2, sample_radius//2),
            (-sample_radius//2, -sample_radius//2),
            (sample_radius//2, -sample_radius//2),
            (-sample_radius//2, sample_radius//2),
            (sample_radius//2, 0),
            (-sample_radius//2, 0),
            (0, sample_radius//2),
            (0, -sample_radius//2)
        ]
        for dx, dy in additional_offsets:
            sample_x = center_x + dx
            sample_y = center_y + dy
            sample_x = max(0, min(w-1, sample_x))
            sample_y = max(0, min(h-1, sample_y))
            sample_points.append((sample_x, sample_y))
        
        # 收集所有有效深度值
        valid_depths = []
        for px, py in sample_points:
            px = max(0, min(w-1, px))
            py = max(0, min(h-1, py))
            d = depth_image[py, px]
            if d > 0:
                valid_depths.append(d)
        
        # 如果采样点不够，扩大搜索范围
        if len(valid_depths) < 2:
            search_radius = 10
            for dy in range(-search_radius, search_radius+1):
                for dx in range(-search_radius, search_radius+1):
                    ny, nx = center_y + dy, center_x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        d = depth_image[ny, nx]
                        if d > 0:
                            valid_depths.append(d)
        
        if len(valid_depths) == 0:
            return None, None
        
        # 使用均值计算深度值
        depth_value = np.mean(valid_depths)
        
        # 3. 将2D中心转换为3D坐标
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_3d = (center_x - cx) * depth_value / fx
        y_3d = (center_y - cy) * depth_value / fy
        z_3d = depth_value
        
        center_3d = np.array([x_3d, y_3d, z_3d])
        
        # 4. 计算杯子的向上方向向量（Z轴）
        top_mid = (trapezoid_points[0] + trapezoid_points[1]) / 2  # 上边中点
        bottom_mid = (trapezoid_points[2] + trapezoid_points[3]) / 2  # 下边中点
        
        # 对称轴方向（从下到上，即杯子向上方向）
        symmetry_axis_2d = top_mid - bottom_mid
        symmetry_axis_2d = symmetry_axis_2d / np.linalg.norm(symmetry_axis_2d)
        
        # 将2D方向向量转换为3D方向向量
        normal_3d = np.array([symmetry_axis_2d[0], symmetry_axis_2d[1], 0.5])
        normal_3d = normal_3d / np.linalg.norm(normal_3d)
        
        return center_3d, normal_3d
        
    except Exception as e:
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
            return None, None, None, None
        
        # 5. 计算勺头中心（勺头区域点云的质心）
        head_center = np.mean(head_points, axis=0)
        
        # 6. 估计勺头半径（在垂直于主轴的平面上）
        head_centered = head_points - head_center
        perpendicular_components = head_centered - (head_centered @ main_axis)[:, np.newaxis] * main_axis
        perpendicular_distances = np.linalg.norm(perpendicular_components, axis=1)
        head_radius = np.mean(perpendicular_distances)
        
        # 7. 计算勺柄方向和姿态
        handle_direction = main_axis
        handle_roll, handle_pitch, handle_yaw = vector_to_euler(handle_direction)
        handle_pose = [handle_roll, handle_pitch, handle_yaw]
        
        return head_center, head_radius, handle_direction, handle_pose
        
    except Exception as e:
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
            
            # 如果负向端分数更高，翻转主轴方向
            if negative_score > positive_score:
                main_axis = -main_axis
        
        # 8. 估计物体长度（沿主轴的范围）
        projections = centered_points @ main_axis  # 重新计算投影
        length = np.max(projections) - np.min(projections)
        
        # 9. 计算真正的中心点：沿主轴的几何中点（不是质心）
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        mid_proj = (min_proj + max_proj) / 2.0  # 主轴上的中点投影值
        
        # 将中点投影值转换回3D空间坐标
        center = centroid + mid_proj * main_axis
        
        return center, main_axis, secondary_axis, length
        
    except Exception as e:
        return None, None, None, None




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
        # Z轴：杯子向上方向（已经计算好的normal_3d）
        z_axis = normal_3d / np.linalg.norm(normal_3d)
        
        # X轴：基于梯形上边的水平方向，指向左方
        top_edge = trapezoid_points[0] - trapezoid_points[1]  # 从左到右的方向
        top_edge = top_edge / np.linalg.norm(top_edge)
        
        # 将2D上边方向转换为3D
        x_axis = np.array([top_edge[0], top_edge[1], 0])
        
        # 确保X轴与Z轴正交
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y轴：通过叉乘得到（右手坐标系）
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3, 0] = x_axis
        T[:3, 1] = y_axis
        T[:3, 2] = z_axis
        T[:3, 3] = center_3d
        
        return T
        
    except Exception as e:
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
    
    支持的物体类别：
    - 杯子类 ('cup', 'mug', 'glass'): 使用梯形轮廓法
    - 细长物体 ('spoon', 'knife', 'fork'等): 使用PCA主轴法
    
    参数:
        mask (numpy.ndarray): 物体掩码，形状为(H, W)，值为0或1
        depth_image (numpy.ndarray): 深度图像，形状为(H, W)，单位为米
        color_image (numpy.ndarray): 彩色图像，形状为(H, W, 3)
        intrinsics (numpy.ndarray): 3x3相机内参矩阵
        T_cam2base (numpy.ndarray): 4x4相机到基坐标系的变换矩阵（可选）
        object_class (str): 物体类别，用于选择合适的估计方法
    
    返回:
        pose: [x, y, z, roll, pitch, yaw] 在基坐标系中的位姿，失败时返回None
        T: 4x4变换矩阵，失败时返回None
    """
    try:
        # 0. 确保mask尺寸与图像匹配
        h, w = color_image.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        
        if (mask_h, mask_w) != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)         
        
        # 1. 根据掩码提取点云
        # 确保mask是2D的 (H, W)
        mask_2d = mask[:, :, 0] if len(mask.shape) == 3 else mask
        # 确保depth是2D的 (H, W)
        depth_2d = depth_image[:, :, 0] if len(depth_image.shape) == 3 else depth_image
        
        # 应用掩码
        color_masked = color_image * mask_2d[:, :, np.newaxis]
        depth_masked = depth_2d * mask_2d
        
        point_cloud = create_point_cloud(depth_masked, intrinsics, color_masked)
        
        if len(point_cloud.points) < 50:
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
            
            # 如果有勺头信息，添加到pose中
            if extra_info:
                pose = list(pose)  # 转换为列表以便添加额外信息
                pose.append(extra_info)  # 将额外信息作为第7个元素
            
        else:
            # 使用梯形轮廓方法
            if object_class.lower() in ['cup', 'mug', 'glass']:
                # 注意：extract_cup_side_contour内部会进行掩码清理，所以传入原始mask即可
                trapezoid_points, center_3d, normal_3d, success = extract_cup_side_contour(
                    mask, color_image, depth_2d, intrinsics
                )
                
                if success and center_3d is not None:
                    # 计算位姿变换矩阵
                    T = calculate_cup_pose_from_trapezoid_matrix(center_3d, normal_3d, trapezoid_points)
                    
                    if T is not None:
                        # 转换为位置和欧拉角
                        pose = transform_matrix_to_pos_euler(T)
                        return pose, T
            
            # 如果不是杯子类物体或梯形轮廓法失败，返回None
            return None, None
        
        return pose, T
        
    except Exception as e:
        return None, None

