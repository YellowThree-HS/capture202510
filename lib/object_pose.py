import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

def pos_euler_to_transform_matrix(pos_euler):
    x, y, z, rx, ry, rz = pos_euler
    
    # 将欧拉角转换为旋转矩阵（需明确旋转顺序）
    euler_angles = [rx, ry, rz]
    r = R.from_euler('xyz', euler_angles, degrees=True)
    rotation_matrix = r.as_matrix()
    
    # 构建齐次变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix  # 填充旋转部分
    transform_matrix[:3, 3] = [x, y, z]          # 填充平移部分
    
    return transform_matrix

def result_show(color_data, depth_data,T_cam2base,intr_matrix,pose_list):
    pcd = create_point_cloud(depth_data, intr_matrix, color_data)
    pcd.transform(T_cam2base)  # 转换到基坐标系
    grasp_coordinate_frame_lsit=[]
    for i in pose_list:
        i=pos_euler_to_transform_matrix(i)
        grasp_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        grasp_coordinate_frame.transform(i)
        grasp_coordinate_frame_lsit.append(grasp_coordinate_frame)
    o3d.visualization.draw_geometries([pcd]+grasp_coordinate_frame_lsit)

def transform_matrix_to_pos_euler(T):
    x, y, z = T[:3, 3]
    # 提取旋转矩阵
    rotation_matrix = T[:3, :3]

    # 将旋转矩阵转换为欧拉角
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=True)  # 'xyz' 表示使用的是绕x, y, z轴的旋转顺序

    return [x, y, z] + list(euler_angles)

"""
计算给定底部角和底面法向量的盒子的变换矩阵。

参数:
- bottom_corners: 一个 4x3 矩阵，每行代表盒子底部一个角的 (x, y, z) 坐标。
- bottom_plane: 一个 1x4 向量，表示平面方程，形式为 [a, b, c, d]，其中 ax + by + cz + d = 0。

返回:
- 一个 4x4 变换矩阵。
"""
def calculate_transformation_matrix(bottom_corners, plane_normal):

    # 计算盒子底部两条边的长度
    length_edge = np.linalg.norm(bottom_corners[1] - bottom_corners[0])  # 边1：从角0到角1
    width_edge = np.linalg.norm(bottom_corners[3] - bottom_corners[0])  # 边2：从角0到角3
    # 根据底部边的方向确定 y 轴和 z 轴
    if length_edge > width_edge:
        y_direction = bottom_corners[1] - bottom_corners[0]
    else:
        y_direction = bottom_corners[3] - bottom_corners[0]


    z_direction = plane_normal / np.linalg.norm(plane_normal)
    # z_direction = z_direction if np.dot(z_direction, np.array([0, 0, -1])) > 0 else -z_direction

    y_direction = y_direction / np.linalg.norm(y_direction)

    x_direction = np.cross(y_direction, z_direction)
    x_direction = x_direction / np.linalg.norm(x_direction)

    # 构建旋转矩阵
    rotation_matrix = np.column_stack((x_direction, y_direction, z_direction))

    # 构建变换矩阵
    transform_box_to_base = np.eye(4)
    transform_box_to_base[:3, :3] = rotation_matrix
    transform_box_to_base[:3, 3] = np.mean(bottom_corners, axis=0)  # 以底部中心为原点

    return transform_box_to_base

# 左上角，右上角，右下角，左下角
def get_min_area_rect_points(pcd,plane_model):
    point_cloud = np.asarray(pcd.points)
    # 平面法向量
    n = plane_model[:-1]
    # 计算旋转矩阵
    z_axis = np.array([0, 0, 1])
    k = np.cross(n, z_axis)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n, z_axis))
    R = np.array([
        [np.cos(theta) + k[0] ** 2 * (1 - np.cos(theta)), k[0] * k[1] * (1 - np.cos(theta)) - k[2] * np.sin(theta),
        k[0] * k[2] * (1 - np.cos(theta)) + k[1] * np.sin(theta)],
        [k[1] * k[0] * (1 - np.cos(theta)) + k[2] * np.sin(theta), np.cos(theta) + k[1] ** 2 * (1 - np.cos(theta)),
        k[1] * k[2] * (1 - np.cos(theta)) - k[0] * np.sin(theta)],
        [k[2] * k[0] * (1 - np.cos(theta)) - k[1] * np.sin(theta),
        k[2] * k[1] * (1 - np.cos(theta)) + k[0] * np.sin(theta), np.cos(theta) + k[2] ** 2 * (1 - np.cos(theta))]
    ])
    # 应用旋转
    rotated_points = R.dot(point_cloud.T).T
    points_xy = rotated_points[:, :2]  # 提取X和Y坐标
    rect = cv2.minAreaRect((points_xy*10000).astype(np.int32))

    # 步骤5: 获取最小外接矩形的四个顶点
    box = cv2.boxPoints(rect)/10000
    box_3d = np.hstack((box, rotated_points[0,2] * np.ones((4, 1))))
    R_inv = np.linalg.inv(R)
    box_3d_rotated_back = R_inv.dot(box_3d.T).T

    return box_3d_rotated_back

"""
将点云数据投影到给定的平面模型上。

参数：
plane_model (tuple): 平面模型参数，格式为 (a, b, c, d)，其中 a, b, c 是平面法向量的分量，d 是平面方程的常数项。
point_cloud (numpy.ndarray): 输入的点云数据，形状为 (N, 3)，其中 N 是点的数量。

返回：
projected_points (numpy.ndarray): 投影到平面上的点云数据，形状与输入点云相同。
"""
def project_points_to_plane(plane_model, point_cloud):
    # 提取平面模型参数
    a, b, c, d = plane_model

    # 计算每个点到平面的距离D
    D = (a * point_cloud[:, 0] + b * point_cloud[:, 1] + c * point_cloud[:, 2] + d) / (a ** 2 + b ** 2 + c ** 2)

    # 计算平面法向量
    normal_vector = np.array([a, b, c])

    # 计算投影点云
    projected_points = point_cloud - D[:, np.newaxis] * normal_vector
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
    return projected_pcd

def _process_plane(point_cloud):
    """封装平面分割到几何特征提取的全流程"""
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.005, ransac_n=3,
                                                                num_iterations=1000)
    filtered = point_cloud.select_by_index(inliers)
    projected = project_points_to_plane(plane_model, np.asarray(filtered.points))
    cleaned, _ = projected.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
    corners = get_min_area_rect_points(cleaned, plane_model)
    normal = -plane_model[:-1] / np.linalg.norm(plane_model[:-1])
    return corners, normal

# 左上角，右上角，右下角，左下角
def get_box_size(point_cloud):
    """
    计算纸箱的实际尺寸
    
    参数:
        point_cloud: 纸箱的点云数据
        
    返回:
        tuple: (长度, 宽度, 高度) 单位：米
    """
    try:
        # 1. 使用RANSAC分割平面，获取纸箱顶面
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=0.005, 
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) < 10:  # 确保有足够的点
            print("⚠️ 无法检测到足够的平面点，使用默认高度")
            return 0.2, 0.15, 0.1
        
        # 2. 获取顶面点云
        filtered = point_cloud.select_by_index(inliers)
        
        # 3. 获取纸箱底部角点
        corners, normal = _process_plane(point_cloud)
        
        if corners is None or len(corners) < 4:
            print("⚠️ 无法获取角点信息，使用默认尺寸")
            return 0.2, 0.15, 0.1
        
        # 4. 计算长度和宽度
        length_edge = np.linalg.norm(corners[1] - corners[0])  # 上边
        width_edge = np.linalg.norm(corners[3] - corners[0])   # 左边
        
        # 5. 计算纸箱高度
        # 方法1: 使用点云Z轴范围计算高度
        all_points = np.asarray(point_cloud.points)
        if len(all_points) > 0:
            z_min = np.min(all_points[:, 2])
            z_max = np.max(all_points[:, 2])
            height = z_max - z_min
        else:
            height = 0.1  # 默认高度
        
        # 6. 确保高度合理（如果计算出的高度太小，使用默认值）
        if height < 0.02:  # 小于2cm认为不合理
            height = 0.1
            print("⚠️ 计算高度过小，使用默认高度")
        
        # 7. 确定长度和宽度（确保长度 >= 宽度）
        if length_edge >= width_edge:
            box_length = length_edge
            box_width = width_edge
        else:
            box_length = width_edge
            box_width = length_edge
        
        print(f"📏 纸箱尺寸计算:")
        print(f"   长度: {box_length:.3f}m ({box_length*100:.1f}cm)")
        print(f"   宽度: {box_width:.3f}m ({box_width*100:.1f}cm)")
        print(f"   高度: {height:.3f}m ({height*100:.1f}cm)")
        
        return box_length, box_width, height
        
    except Exception as e:
        print(f"❌ 计算纸箱尺寸时出错: {e}")
        print("⚠️ 使用默认尺寸")
        return 0.2, 0.15, 0.1



"""
根据深度图像和相机内参，创建点云数据，并结合彩色图像为点云上色。

参数:
depth_image (numpy.ndarray): 深度图像数据。
intrinsics (list or numpy.ndarray): 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]。
color_image (numpy.ndarray): 彩色图像数据。
返回:
pcd (open3d.geometry.PointCloud): 创建的点云对象。
"""
def create_point_cloud(depth_image, intrinsics, color_image):
    # 验证输入图像尺寸是否匹配
    if depth_image.shape[:2] != color_image.shape[:2]:
        raise ValueError("Depth and color images must have the same dimensions.")

    # 初始化网格坐标，确保与深度图尺寸一致
    height, width = depth_image.shape[:2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # 将深度值超出合理范围的部分设为无效（例如设置为0）
    valid_depth = depth_image.copy().astype(float)
    valid_depth[depth_image > 3.5] = 0

    # 计算点云坐标，注意有效深度值需要乘以比例因子
    z = valid_depth
    x = (u - intrinsics[0][2]) * z / intrinsics[0][0]
    y = (v - intrinsics[1][2]) * z / intrinsics[1][1]

    # 组合点云坐标和颜色信息
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image[..., ::-1].reshape(-1, 3) / 255.0  # RGB转BGR再转到0-1范围

    # 过滤掉无效点（深度为0的点）
    valid_points_mask = (z.reshape(-1) > 0)
    points = points[valid_points_mask]
    colors = colors[valid_points_mask]

    # 创建Open3D点云对象并赋值
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def visualize_point_cloud(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])

def _process_mask_data(color_data, depth_data, carton_mask,intr_matrix):
    
    color_masked = color_data * carton_mask[:, :, np.newaxis]
    depth_masked = depth_data * carton_mask
    return create_point_cloud(depth_masked, intr_matrix, color_masked)


def get_pose(color_data, depth_data, box_mask, T_cam2base, intr_matrix):
    """
    根据颜色和深度数据计算物体在机器人基坐标系中的位姿
    
    参数:
        color_data (numpy.ndarray): RGB彩色图像数据，形状为(H, W, 3)
        depth_data (numpy.ndarray): 深度图像数据，形状为(H, W)，单位为米
        box_mask (numpy.ndarray): 物体检测框的掩码，形状为(H, W)，值为0或1
        T_cam2base (numpy.ndarray): 4x4齐次变换矩阵，表示相机坐标系到机器人基坐标系的变换
        intr_matrix (numpy.ndarray): 3x3相机内参矩阵
    
    返回:
        list: 包含物体位姿的列表 [x, y, z, roll, pitch, yaw]
            x, y, z: 物体在机器人基坐标系中的位置（单位：米）
            roll, pitch, yaw: 物体在机器人基坐标系中的欧拉角（单位：弧度）
    
    处理流程:
        1. 根据掩码提取点云数据
        2. 将点云从相机坐标系转换到机器人基坐标系
        3. 处理点云平面特征
        4. 计算物体位姿的变换矩阵
        5. 将变换矩阵转换为位置和欧拉角表示
        6. 可视化结果（可选）
    """
    
    # 1. 根据掩码提取点云数据
    # 输入：颜色图像、深度图像、掩码、相机内参
    # 输出：点云对象（包含物体表面的3D点）
    point_cloud = _process_mask_data(color_data, depth_data, box_mask, intr_matrix)
    
    # 2. 将点云从相机坐标系转换到机器人基坐标系
    # 使用齐次变换矩阵 T_cam2base 进行坐标变换
    point_cloud.transform(T_cam2base)
    
    # 3. 处理点云平面特征
    # 输入：变换后的点云
    # 输出：包含平面特征的对象（如法向量、中心点等）
    features = _process_plane(point_cloud)
    
    # 4. 计算物体位姿的变换矩阵
    # 输入：平面特征（如法向量和中心点）
    # 输出：4x4齐次变换矩阵，表示物体坐标系到机器人基坐标系的变换
    T = calculate_transformation_matrix(features[0], features[1])
    
    # 5. 将变换矩阵转换为位置和欧拉角表示
    # 输入：4x4变换矩阵
    # 输出：位置和欧拉角列表 [x, y, z, roll, pitch, yaw]
    pos_euler = transform_matrix_to_pos_euler(T)
    
    # 6. 可视化结果（可选）
    # 在原始图像上绘制检测结果和位姿信息
    # result_show(color_data, depth_data, T_cam2base, intr_matrix, [pos_euler])
    
    return T, pos_euler

def pixel_in_camera(u, v, z, intr_matrix):
    """将像素坐标转换为世界坐标"""
    # intr_matrix = np.array([[camera_params.fx, 0, camera_params.cx],
    #                     [0, camera_params.fy, camera_params.cy],
    #                     [0, 0, 1]], dtype=np.float64)
    # 相机坐标系

    if z <= 0:  # 无效深度
        return None

    # 从CameraInfo获取内参
    fx = intr_matrix[0][0]
    fy = intr_matrix[1][1]
    cx = intr_matrix[0][2]
    cy = intr_matrix[1][2]

    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z

    # 构造齐次坐标
    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    return point_cam

def point_clouds_filter(x,y,z):    
    # 1. 组织点云数据
    points = np.array([x, y, z]).T

    # 2. KDTree 最近邻距离计算
    kdtree = KDTree(points)
    distances, _ = kdtree.query(points, k=5)  # 计算每个点的最近邻距离
    mean_distances = np.mean(distances, axis=1)
    threshold = np.percentile(mean_distances, 90)  # 设定阈值（如90%分位数）
    filtered_indices = mean_distances < threshold
    points = points[filtered_indices]

    # 3. DBSCAN 聚类
    dbscan = DBSCAN(eps=0.05, min_samples=10)  # 可调整超参数
    labels = dbscan.fit_predict(points)

    unique_labels = set(labels) - {-1}  # 去除噪声点
    if len(unique_labels) == 0:
        return [], [], []

    # 4. 计算每个聚类的中心坐标和点数
    cluster_sizes = {}
    cluster_centers = {}
    for label in unique_labels:
        cluster_points = points[labels == label]
        cluster_sizes[label] = len(cluster_points)
        cluster_centers[label] = np.mean(cluster_points, axis=0)  # 计算质心

    # 5. 计算聚类之间的最小距离
    cluster_labels = list(cluster_centers.keys())
    cluster_distances = np.zeros((len(cluster_labels), len(cluster_labels)))

    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            dist = np.linalg.norm(
                cluster_centers[cluster_labels[i]]
                - cluster_centers[cluster_labels[j]]
            )
            cluster_distances[i, j] = dist
            cluster_distances[j, i] = dist

    # 6. 检查是否有多个聚类中心且最近距离 > 0.3m
    if (
        len(cluster_labels) > 1
        and np.min(cluster_distances[np.nonzero(cluster_distances)]) > 0.3
    ):

        # 选择最大的聚类
        largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
        valid_indices = labels == largest_cluster_label
    else:
        valid_indices = labels != -1  # 保留所有有效点（非噪声）

    # 7. 过滤点云数据
    filtered_x = points[valid_indices, 0].tolist()
    filtered_y = points[valid_indices, 1].tolist()
    filtered_z = points[valid_indices, 2].tolist()
    return filtered_x, filtered_y, filtered_z

def extract_dual_peaks_height(z_list, bin_width=0.005, min_peak_distance=0.02):
    """
    从z_list中提取双峰分布的高度
    
    参数:
        z_list (list): z坐标列表
        bin_width (float): 直方图bin宽度，默认5mm
        min_peak_distance (float): 两个峰值之间的最小距离，默认5cm
    
    返回:
        float: 高度值（两个峰值之间的差值）
    """
    
    if len(z_list) < 10:
        print("⚠️ z_list数据点太少，无法进行峰值检测")
        return 0.1  # 返回默认高度
    
    # 转换为numpy数组
    z_array = np.array(z_list)
    
    # 创建直方图
    z_min, z_max = np.min(z_array), np.max(z_array)
    bins = np.arange(z_min, z_max + bin_width, bin_width)
    hist, bin_edges = np.histogram(z_array, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 寻找峰值
    # 使用prominence参数确保找到明显的峰值
    peaks, properties = find_peaks(hist, 
                                 prominence=np.max(hist) * 0.1,  # 峰值至少是最大值的10%
                                 distance=int(min_peak_distance / bin_width))  # 峰值间最小距离
    
    if len(peaks) < 2:
        print("⚠️ 未找到足够的峰值，使用默认高度")
        return 0.1
    
    # 获取两个最显著的峰值
    peak_heights = hist[peaks]
    # 按峰值高度排序，取前两个
    top_peaks_idx = np.argsort(peak_heights)[-2:]
    top_peaks = peaks[top_peaks_idx]
    
    # 获取峰值对应的z坐标
    peak_z_values = bin_centers[top_peaks]
    peak_z_values = np.sort(peak_z_values)  # 从小到大排序
    
    # 计算高度
    height = peak_z_values[1] - peak_z_values[0]
    
    print(f"🔍 双峰检测结果:")
    print(f"   峰值1 z坐标: {peak_z_values[0]:.3f}m")
    print(f"   峰值2 z坐标: {peak_z_values[1]:.3f}m")
    print(f"   计算高度: {height:.3f}m ({height*100:.1f}cm)")
    print(f"   检测到的峰值数量: {len(peaks)}")
    
    # 验证高度是否合理
    if height < 0.01 or height > 0.5:  # 高度在1cm到50cm之间
        print("⚠️ 计算出的高度不合理，使用默认高度")
        return 0.1
    
    return height

def get_box_size_v2(point_cloud, x_list, y_list, z_list):
    """
    计算纸箱的实际尺寸
    
    参数:
        point_cloud: 纸箱的点云数据
        table_height: 桌子高度，单位：米
        x_list, y_list, z_list: 带有背景的bbox点云（雨田)
    返回:
        tuple: (长度, 宽度, 高度) 单位：米
    """
    try:
        # 1. 使用RANSAC分割平面，获取纸箱顶面
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=0.005, 
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) < 10:  # 确保有足够的点
            print("⚠️ 无法检测到足够的平面点，使用默认高度")
            return 0.2, 0.15, 0.1
        
        # 2. 获取顶面点云
        filtered = point_cloud.select_by_index(inliers)
        
        # 3. 获取纸箱底部角点
        corners, normal = _process_plane(point_cloud)
        
        if corners is None or len(corners) < 4:
            print("⚠️ 无法获取角点信息，使用默认尺寸")
            return 0.2, 0.15, 0.1
    
        # 4. 计算长度和宽度
        length_edge = np.linalg.norm(corners[1] - corners[0])  # 上边
        width_edge = np.linalg.norm(corners[3] - corners[0])   # 左边
        if length_edge >= width_edge:
            box_length = length_edge
            box_width = width_edge
        else:
            box_length = width_edge
            box_width = length_edge
        
        # 5. 计算纸箱高度
        # 方法1: 使用双峰检测方法（推荐）
        print("🔍 使用双峰检测方法计算高度...")
        height = extract_dual_peaks_height(z_list, bin_width=0.005, min_peak_distance=0.05)
        
        # 方法2: 备用方法 - 对点云进行聚类过滤
        if height == 0.1:  # 如果双峰检测失败，使用备用方法
            print("🔄 双峰检测失败，使用备用聚类方法...")
            filtered_x, filtered_y, filtered_z = point_clouds_filter(x_list, y_list, z_list)
            
            # 检查过滤后的数据是否有效
            if len(filtered_z) == 0:
                print("⚠️ 过滤后没有有效的点云数据，使用默认高度")
                height = 0.1
            else:
                height = np.max(filtered_z) - np.min(filtered_z)
                
            # 确保高度合理
            if height < 0.02:  # 小于2cm认为不合理
                height = 0.1
                print("⚠️ 计算高度过小，使用默认高度")
        
        print(f"📏 纸箱尺寸计算 (v2):")
        print(f"   长度: {box_length:.3f}m ({box_length*100:.1f}cm)")
        print(f"   宽度: {box_width:.3f}m ({box_width*100:.1f}cm)")
        print(f"   高度: {height:.3f}m ({height*100:.1f}cm)")
        print(f"   过滤后点云数量: {len(filtered_z)}")
        
        return box_length, box_width, height
        
    except Exception as e:
        print(f"计算纸箱尺寸时出错: {e}")
        return None


def get_box_length_width(point_cloud):
    """
    计算纸箱的实际尺寸
    
    参数:
        point_cloud: 纸箱的点云数据
        
    返回:
        tuple: (长度, 宽度, 高度) 单位：米
    """
    try:
        # 1. 使用RANSAC分割平面，获取纸箱顶面
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=0.005, 
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) < 10:  # 确保有足够的点
            print("⚠️ 无法检测到足够的平面点，使用默认高度")
            return 0.2, 0.15, 0.1
        
        # 2. 获取顶面点云
        filtered = point_cloud.select_by_index(inliers)
        
        # 3. 获取纸箱底部角点
        corners, normal = _process_plane(point_cloud)
        
        if corners is None or len(corners) < 4:
            print("⚠️ 无法获取角点信息，使用默认尺寸")
            return 0.2, 0.15, 0.1
        
        # 4. 计算长度和宽度
        length_edge = np.linalg.norm(corners[1] - corners[0])  # 上边
        width_edge = np.linalg.norm(corners[3] - corners[0])   # 左边
        
        # 7. 确定长度和宽度（确保长度 >= 宽度）
        if length_edge >= width_edge:
            box_length = length_edge
            box_width = width_edge
        else:
            box_length = width_edge
            box_width = length_edge
        
        print(f"📏 纸箱尺寸计算:")
        print(f"   长度: {box_length:.3f}m ({box_length*100:.1f}cm)")
        print(f"   宽度: {box_width:.3f}m ({box_width*100:.1f}cm)")
        
        return box_length, box_width
        
    except Exception as e:
        print(f"❌ 计算纸箱尺寸时出错: {e}")
        print("⚠️ 使用默认尺寸")
        return 0.2, 0.15, 0.1

if __name__ == "__main__":
    point_cloud = o3d.io.read_point_cloud("pointcloud.pcd")
    point_cloud_with_background = o3d.io.read_point_cloud("pointcloud.pcd")
    # visualize_point_cloud(point_cloud, point_cloud_with_background)
    print(get_box_size_v2(point_cloud, point_cloud_with_background))
