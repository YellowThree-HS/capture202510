import numpy as np
import cv2
from lib.dobot import DobotRobot
from lib.camera import Camera
from ultralytics import YOLO
import open3d as o3d
def load_image(cam):
    color_image = cv2.imread("lid/color.png")
    depth_image = cv2.imread("lid/depth.png", cv2.IMREAD_UNCHANGED)
    color_image, depth_image = cam.get_frames()
    return color_image, depth_image

def detect_with_yolo(color_image, model_path="weights/all50.pt", conf_threshold=0.5, target_class="lid"):
    model = YOLO(model_path)
    results = model.predict(
        source=color_image,
        save=False,
        conf=conf_threshold,
        iou=0.7,
        verbose=False
    )
    result = results[0]
    class_names = result.names
    
    target_class_id = None
    for class_id, class_name in class_names.items():
        if class_name == target_class:
            target_class_id = class_id
            break
    
    
    all_bboxes = result.boxes.xyxy.cpu().numpy()
    all_masks = result.masks.data.cpu().numpy()
    all_confs = result.boxes.conf.cpu().numpy()
    all_classes = result.boxes.cls.cpu().numpy()
    
    target_indices = np.where(all_classes == target_class_id)[0]
    
    filtered_bboxes = all_bboxes[target_indices]
    filtered_masks = all_masks[target_indices]
    filtered_confs = all_confs[target_indices]
    
    print(f"✅ 检测到 {len(filtered_bboxes)} 个 '{target_class}'")
    
    return {
        'bboxes': filtered_bboxes,
        'masks': filtered_masks,
        'confidences': filtered_confs,
        'class_name': target_class
    }

def refine_masks_with_morphology(masks, erode_kernel_size=5, close_kernel_size=3, iterations=1):

    refined_masks = []
    for i, mask in enumerate(masks):
        # 将掩码转换为二值图像
        h, w = mask.shape
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 1. 先进行闭运算（填充掩码内部的小孔）
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # 2. 腐蚀操作（让掩码集中，去除边缘毛刺）
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
        eroded = cv2.erode(closed, erode_kernel, iterations=iterations)
        
        # 3. 转换回浮点型 [0, 1]
        refined_mask = eroded.astype(np.float32) / 255.0
        
        # 计算优化前后的面积变化
        original_area = np.sum(mask > 0.5)
        refined_area = np.sum(refined_mask > 0.5)
        
        if refined_area > 0:  # 确保掩码没有被完全腐蚀掉
            refined_masks.append(refined_mask)
        else:
            refined_masks.append(mask)
    return np.array(refined_masks)

def depth_to_pointcloud(depth_image, mask, color_image=None, camera_intrinsics=None, depth_scale=0.0001):
    """将深度图像和掩码转换为点云"""
    if mask.shape != depth_image.shape[:2]:
        mask = cv2.resize(mask.astype(np.float32), 
                         (depth_image.shape[1], depth_image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    if camera_intrinsics is None:
        h, w = depth_image.shape[:2]
        camera_intrinsics = {
            'fx': w * 0.8,
            'fy': h * 0.8,
            'ppx': w / 2.0,
            'ppy': h / 2.0
        }
    
    binary_mask = (mask > 0.5).astype(np.uint8)
    y_coords, x_coords = np.where(binary_mask == 1)
    
    if len(x_coords) == 0:
        return np.array([]), np.array([])
    
    depth_values = depth_image[y_coords, x_coords]
    if depth_values.ndim > 1:
        depth_values = depth_values.flatten()
    
    valid_depth_mask = (depth_values > 0) & (depth_values < 10000)
    if np.sum(valid_depth_mask) == 0:
        return np.array([]), np.array([])
    
    valid_x = x_coords[valid_depth_mask]
    valid_y = y_coords[valid_depth_mask]
    valid_depth = depth_values[valid_depth_mask]
    
    depth_meters = valid_depth * depth_scale
    
    fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
    ppx, ppy = camera_intrinsics['ppx'], camera_intrinsics['ppy']
    
    x_3d = (valid_x - ppx) / fx * depth_meters
    y_3d = (valid_y - ppy) / fy * depth_meters
    z_3d = depth_meters
    
    points = np.column_stack([x_3d, y_3d, z_3d])
    
    if color_image is not None:
        if color_image.shape[:2] != depth_image.shape[:2]:
            color_image_resized = cv2.resize(color_image, 
                                           (depth_image.shape[1], depth_image.shape[0]))
        else:
            color_image_resized = color_image
        
        color_values = color_image_resized[valid_y, valid_x]
        colors = color_values[:, [2, 1, 0]]  # BGR -> RGB
    else:
        colors = np.column_stack([
            valid_depth / 1000.0 * 255,
            valid_depth / 1000.0 * 255,
            valid_depth / 1000.0 * 255
        ]).astype(np.uint8)
    
    print(f"   点云: {len(points)} 点")
    return points, colors

def transform_points_to_robot_base(points, robot_pose_matrix, hand_eye_matrix, cam_intrinsics):
    # 转换到机械臂基坐标系：pose_matrix = robot_pose_matrix @ hand_eye_matrix @ T_object2cam
    if len(points) == 0:
        return points
    
    # 将点云转换为齐次坐标 (N, 4)
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    
    # 计算完整的变换矩阵: T_base = T_robot @ T_hand_eye
    # 这里假设 points 已经在相机坐标系中，需要转换到基坐标系
    transform_matrix = robot_pose_matrix @ hand_eye_matrix
    
    # 应用变换矩阵
    transformed_points_homogeneous = (transform_matrix @ points_homogeneous.T).T
    
    # 转换回3D坐标
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points

def extract_top_height_points(points, colors, height_percentage=0.1):
    if len(points) == 0:
        print("❌ 点云为空")
        return np.array([]), np.array([]), None
    
    # 获取Z坐标（高度）
    heights = points[:, 2]
    
    # 计算要保留的点数
    num_keep = max(int(len(points) * height_percentage), 1)
    
    # 获取高度最高的点的索引
    top_height_indices = np.argsort(heights)[-num_keep:]
    
    # 提取高度前percentage%的点
    top_points = points[top_height_indices]
    top_colors = colors[top_height_indices]
    
    # 计算边界框
    min_coords = top_points.min(axis=0)
    max_coords = top_points.max(axis=0)
    bbox = np.concatenate([min_coords, max_coords])  # [min_x, min_y, min_z, max_x, max_y, max_z]
    
    return top_points, top_colors, bbox


def visualize_bbox_selection(points, colors, bbox, window_name="矩形框选点云", point_size=4.0):
    if len(points) == 0:
        print("❌ 点云为空")
        return
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900)
    vis.add_geometry(pcd)
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)
    
    # 创建边界框可视化
    if bbox is not None:
        bbox_min = bbox[:3]
        bbox_max = bbox[3:]
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min
        
        # 创建边界框线框
        bbox_mesh = o3d.geometry.TriangleMesh.create_box(
            width=bbox_size[0], height=bbox_size[1], depth=bbox_size[2]
        )
        bbox_mesh.translate(bbox_center - bbox_size/2)
        
        # 设置边界框为线框模式
        bbox_mesh.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色边界框
        bbox_lines = o3d.geometry.LineSet.create_from_triangle_mesh(bbox_mesh)
        bbox_lines.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色线条
        
        vis.add_geometry(bbox_lines)
    
    # 渲染设置
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    render_option.show_coordinate_frame = True
    
    # 相机设置
    view_control = vis.get_view_control()
    view_control.set_front([0.0, 0.0, -1.0])
    view_control.set_up([0.0, -1.0, 0.0])
    view_control.set_lookat(points.mean(axis=0))
    view_control.set_zoom(0.6)
    
    vis.run()
    vis.destroy_window()

def extract_lid_center_from_bbox(points, bbox, num_points=10):
    if len(points) == 0 or bbox is None:
        print("❌ 点云或边界框为空")
        return np.array([0, 0, 0]), np.eye(3)
    
    # 获取边界框信息
    bbox_min = bbox[:3]
    bbox_max = bbox[3:]
    bbox_center = (bbox_min + bbox_max) / 2
    
    # 计算边界框的长边（X和Y方向）
    bbox_size = bbox_max - bbox_min
    x_size = bbox_size[0]
    y_size = bbox_size[1]
    
    # 确定长边方向
    if x_size >= y_size:
        # X方向是长边
        long_axis = 0  # X轴
        short_axis = 1  # Y轴
        long_size = x_size
        short_size = y_size
    else:
        # Y方向是长边
        long_axis = 1  # Y轴
        short_axis = 0  # X轴
        long_size = y_size
        short_size = x_size
    
    
    # 在长边中心附近选择点
    if long_axis == 0:  # X方向是长边
        # 计算长边的中心位置
        long_edge_center_x = (bbox_min[0] + bbox_max[0]) / 2
        
        # 在长边中心附近选择点（Y和Z方向稍微内缩，X方向在中心附近）
        long_edge_mask = (
            (points[:, 0] >= long_edge_center_x - long_size * 0.2) &  # X方向：中心±20%范围
            (points[:, 0] <= long_edge_center_x + long_size * 0.2) &
            (points[:, 1] >= bbox_min[1] + short_size * 0.1) &  # Y方向稍微内缩
            (points[:, 1] <= bbox_max[1] - short_size * 0.1) &
            (points[:, 2] >= bbox_min[2] + bbox_size[2] * 0.1) &  # Z方向稍微内缩
            (points[:, 2] <= bbox_max[2] - bbox_size[2] * 0.1)
        )
        long_edge_points = points[long_edge_mask]
        
        if len(long_edge_points) > 0:
            # 按距离长边中心的距离排序，选择最近的num_points个点
            distances_to_center = np.abs(long_edge_points[:, 0] - long_edge_center_x)
            sorted_indices = np.argsort(distances_to_center)
            selected_points = long_edge_points[sorted_indices[:min(num_points, len(long_edge_points))]]
        else:
            # 如果没有找到合适的点，使用边界框中心
            selected_points = np.array([bbox_center])
    else:  # Y方向是长边
        # 计算长边的中心位置
        long_edge_center_y = (bbox_min[1] + bbox_max[1]) / 2
        
        # 在长边中心附近选择点（X和Z方向稍微内缩，Y方向在中心附近）
        long_edge_mask = (
            (points[:, 0] >= bbox_min[0] + short_size * 0.1) &  # X方向稍微内缩
            (points[:, 0] <= bbox_max[0] - short_size * 0.1) &
            (points[:, 1] >= long_edge_center_y - long_size * 0.2) &  # Y方向：中心±20%范围
            (points[:, 1] <= long_edge_center_y + long_size * 0.2) &
            (points[:, 2] >= bbox_min[2] + bbox_size[2] * 0.1) &  # Z方向稍微内缩
            (points[:, 2] <= bbox_max[2] - bbox_size[2] * 0.1)
        )
        long_edge_points = points[long_edge_mask]
        
        if len(long_edge_points) > 0:
            # 按距离长边中心的距离排序，选择最近的num_points个点
            distances_to_center = np.abs(long_edge_points[:, 1] - long_edge_center_y)
            sorted_indices = np.argsort(distances_to_center)
            selected_points = long_edge_points[sorted_indices[:min(num_points, len(long_edge_points))]]
        else:
            # 如果没有找到合适的点，使用边界框中心
            selected_points = np.array([bbox_center])
    
    # 计算盖子中心点
    lid_center = np.mean(selected_points, axis=0)
    
    
    print(f"   盖子中心: [{lid_center[0]:.3f}, {lid_center[1]:.3f}, {lid_center[2]:.3f}]")
    
    # 建立局部坐标系
    # Z轴：垂直桌面向上 (0, 0, 1)
    z_axis = np.array([0, 0, 1])
    
    # X轴：与长边平行
    if long_axis == 0:  # X方向是长边
        x_axis = np.array([1, 0, 0])  # 与X轴平行
    else:  # Y方向是长边
        x_axis = np.array([0, 1, 0])  # 与Y轴平行
    
    # Y轴：垂直于长边向外（右手坐标系）
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)  # 归一化
    
    # 重新计算X轴确保正交
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)  # 归一化
    
    # 构建局部坐标系矩阵
    local_coordinate_system = np.column_stack([x_axis, y_axis, z_axis])

    return lid_center, local_coordinate_system

def visualize_lid_center(points, colors, lid_center, local_coordinate_system, window_name="盖子中心点可视化", point_size=3.0):
    if len(points) == 0:
        print("❌ 点云为空")
        return
    
    # 创建原始点云（使用完整RGB颜色）
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(points)
    original_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
    
    # 创建盖子中心点（红色大球）
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0001)
    center_sphere.translate(lid_center)
    center_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    
    # 创建局部坐标系
    x_axis, y_axis, z_axis = local_coordinate_system[:, 0], local_coordinate_system[:, 1], local_coordinate_system[:, 2]
    axis_length = 0.05  # 坐标系轴长度
    
    # X轴（红色）
    x_axis_end = lid_center + x_axis * axis_length
    x_axis_line = o3d.geometry.LineSet()
    x_axis_line.points = o3d.utility.Vector3dVector([lid_center, x_axis_end])
    x_axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    x_axis_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # 红色
    
    # Y轴（绿色）
    y_axis_end = lid_center + y_axis * axis_length
    y_axis_line = o3d.geometry.LineSet()
    y_axis_line.points = o3d.utility.Vector3dVector([lid_center, y_axis_end])
    y_axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    y_axis_line.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]])  # 绿色
    
    # Z轴（蓝色）
    z_axis_end = lid_center + z_axis * axis_length
    z_axis_line = o3d.geometry.LineSet()
    z_axis_line.points = o3d.utility.Vector3dVector([lid_center, z_axis_end])
    z_axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    z_axis_line.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 1.0]])  # 蓝色
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900)
    
    # 添加几何体
    vis.add_geometry(original_pcd)
    vis.add_geometry(center_sphere)
    vis.add_geometry(x_axis_line)
    vis.add_geometry(y_axis_line)
    vis.add_geometry(z_axis_line)
    
    # 添加全局坐标系
    global_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(global_coord)
    
    # 渲染设置
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    
    # 相机设置
    view_control = vis.get_view_control()
    view_control.set_front([0.0, 0.0, -1.0])
    view_control.set_up([0.0, -1.0, 0.0])
    view_control.set_lookat(lid_center)
    view_control.set_zoom(0.6)
    
    
    vis.run()
    vis.destroy_window()


def detect_lid_pose(cam, robot_matrix):
    # 相机内参
    cam_intrinsics = np.array([
        [652.76428223,   0.,         650.07250977],
        [  0.,         651.92443848, 366.10205078],
        [  0.,           0.,           1.        ]
    ])
    # 手眼标定
    hand_eye_matrix = np.array([
        [ 0.01230037,  0.99763761,  0.06758625,  0.08419052],
        [-0.99992251,  0.01240196, -0.00108365,  0.00995925],
        [-0.00191929, -0.06756769,  0.99771285, -0.15882536],
        [ 0.0,         0.0,         0.0,         1.0        ]
    ])
    robot_matrix = np.array([
        [-0.99058,  -0.016664,  -0.13595,  -0.48475 ],
        [-0.013119,  0.99955,   -0.02693,   0.10345 ],
        [ 0.13634,  -0.024892,  -0.99035,   0.33156 ],
        [ 0,         0,          0,         1       ]
    ], dtype=np.float64)
    color_image, depth_image = load_image(cam)
    detection_result = detect_with_yolo(color_image)
    masks = detection_result['masks']
    bboxes = detection_result['bboxes']
    confidences = detection_result['confidences']

    masks = refine_masks_with_morphology(
        masks, 
        erode_kernel_size=7,    # 腐蚀核大小（可调整）
        close_kernel_size=5,    # 闭运算核大小
        iterations=2            # 腐蚀迭代次数（可调整）
    )
    for mask in masks:
        points, colors = depth_to_pointcloud(depth_image, mask, color_image)
        original_points = points.copy()
        transformed_points = transform_points_to_robot_base(points, robot_matrix, hand_eye_matrix, cam_intrinsics)
        top_points, top_colors, bbox = extract_top_height_points(transformed_points, colors, height_percentage=0.1)
        visualize_bbox_selection(top_points, top_colors, bbox, "矩形框选点云")
        lid_center, local_coordinate_system = extract_lid_center_from_bbox(top_points, bbox, num_points=10)
        
        visualize_lid_center(top_points, top_colors, lid_center, local_coordinate_system, "盖子中心点可视化")
    

def main():
    cam = Camera(camera_model='D405')
    robot=DobotRobot(robot_ip='192.168.5.2',no_gripper=True)
    robot_matrix = robot.get_pose_matrix()
    result = detect_lid_pose(cam, robot_matrix=robot_matrix)
    print(result)
    cam.release()



if __name__ == "__main__":
    main()
