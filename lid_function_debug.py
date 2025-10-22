#!/usr/bin/env python3
"""
使用YOLO检测得到掩码，然后保存掩码里的点云
"""

import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
import pyrealsense2 as rs
from datetime import datetime
import open3d as o3d

def transform_points_to_robot_base(points, robot_pose_matrix, hand_eye_matrix, cam_intrinsics):
    """
    将点云从相机坐标系转换到机械臂基坐标系
    
    参数:
        points: 点云坐标 (N, 3)，在相机坐标系中
        robot_pose_matrix: 机械臂末端位姿矩阵 (4x4)
        hand_eye_matrix: 手眼标定矩阵 (4x4)，相机到机械臂末端的变换
        cam_intrinsics: 相机内参矩阵 (3x3)
    
    返回:
        transformed_points: 转换后的点云坐标 (N, 3)，在机械臂基坐标系中
    """
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
    
    print(f"✅ 点云坐标转换完成: {len(transformed_points)} 个点")
    print(f"   原始范围: [{points.min():.3f}, {points.max():.3f}]")
    print(f"   转换后范围: [{transformed_points.min():.3f}, {transformed_points.max():.3f}]")
    
    return transformed_points

def visualize_coordinate_transformation(original_points, transformed_points, window_name="坐标变换对比"):
    """
    可视化坐标变换前后的点云对比
    
    参数:
        original_points: 原始点云 (相机坐标系)
        transformed_points: 变换后点云 (机械臂基坐标系)
        window_name: 窗口名称
    """
    if len(original_points) == 0 or len(transformed_points) == 0:
        print("❌ 点云为空，无法可视化")
        return
    
    # 创建原始点云（红色）
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    original_colors = np.tile([1.0, 0.0, 0.0], (len(original_points), 1))  # 红色
    original_pcd.colors = o3d.utility.Vector3dVector(original_colors)
    
    # 创建变换后点云（绿色）
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_colors = np.tile([0.0, 1.0, 0.0], (len(transformed_points), 1))  # 绿色
    transformed_pcd.colors = o3d.utility.Vector3dVector(transformed_colors)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=800)
    
    # 添加点云
    vis.add_geometry(original_pcd)
    vis.add_geometry(transformed_pcd)
    
    # 添加坐标系
    # 相机坐标系（红色）
    cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(cam_coord)
    
    # 机械臂基坐标系（绿色）
    base_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=transformed_points.mean(axis=0))
    vis.add_geometry(base_coord)
    
    # 渲染设置
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    
    # 相机设置
    view_control = vis.get_view_control()
    view_control.set_front([0.0, 0.0, -1.0])
    view_control.set_up([0.0, -1.0, 0.0])
    view_control.set_lookat(transformed_points.mean(axis=0))
    view_control.set_zoom(0.6)
    
    print(f"🔍 可视化坐标变换:")
    print(f"   红色点云: 相机坐标系 ({len(original_points)} 点)")
    print(f"   绿色点云: 机械臂基坐标系 ({len(transformed_points)} 点)")
    
    vis.run()
    vis.destroy_window()

def load_image_from_path(color_path = "lid/color.png", depth_path = "lid/depth.png"):
    color_image = cv2.imread(color_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    return color_image, depth_image

def detect_with_yolo(color_image, model_path="weights/all.pt", conf_threshold=0.3, target_class="lid"):
    """使用YOLO进行检测，只返回指定类别的结果"""
    
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
    """
    使用形态学操作优化掩码质量
    
    参数:
        masks: numpy数组，shape (N, H, W)，YOLO分割掩码
        erode_kernel_size: 腐蚀核大小（越大腐蚀越强）
        close_kernel_size: 闭运算核大小（用于填充小孔）
        iterations: 腐蚀迭代次数
    
    返回:
        refined_masks: 优化后的掩码
    """
    
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
            print(f"   Mask {i+1}: {original_area} -> {refined_area} 像素 (保留 {refined_area/original_area*100:.1f}%)")
        else:
            # 如果腐蚀过度，使用原掩码
            print(f"   Mask {i+1}: 腐蚀过度，保留原掩码")
            refined_masks.append(mask)
    
    return np.array(refined_masks)

def depth_to_pointcloud(depth_image, mask, color_image=None, camera_intrinsics=None, depth_scale=0.0001):
    """将深度图像和掩码转换为点云"""
    if mask.shape != depth_image.shape[:2]:
        mask = cv2.resize(mask.astype(np.float32), 
                         (depth_image.shape[1], depth_image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    
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

def dbscan_nearest_points(points, colors, keep_ratio=0.1, eps=0.02, min_samples=10, visualize=True):
    """
    提取距离最近的点并进行DBSCAN聚类
    
    参数:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)
        keep_ratio: 保留最近点的比例
        eps: DBSCAN邻域半径
        min_samples: DBSCAN最小样本数
        visualize: 是否可视化对比结果
    
    返回:
        main_cluster_points: 主要簇的点云坐标
        main_cluster_colors: 主要簇的点云颜色
    """
    from sklearn.cluster import DBSCAN
    
    if len(points) == 0:
        print("❌ 输入点云为空")
        return np.array([]), np.array([])
    
    # 1. 计算每个点到相机原点的距离
    distances = np.linalg.norm(points, axis=1)
    
    # 2. 提取距离最近的 keep_ratio 的点
    num_keep = max(int(len(points) * keep_ratio), min_samples)
    nearest_indices = np.argsort(distances)[:num_keep]
    
    nearest_points = points[nearest_indices]
    nearest_colors = colors[nearest_indices]
    
    print(f"   保留最近 {keep_ratio*100:.1f}% 的点: {len(nearest_points)}/{len(points)}")
    print(f"   距离范围: [{distances[nearest_indices].min():.3f}m, {distances[nearest_indices].max():.3f}m]")
    
    # 3. DBSCAN 聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(nearest_points)
    labels = clustering.labels_
    
    # 统计聚类结果
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"   DBSCAN: {n_clusters} 个簇, {n_noise} 个噪声点")
    
    if n_clusters == 0:
        print("❌ 未找到有效簇")
        return np.array([]), np.array([])
    
    # 4. 找到最大的簇（主要簇）
    cluster_sizes = []
    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        cluster_sizes.append(np.sum(cluster_mask))
    
    main_cluster_id = np.argmax(cluster_sizes)
    main_cluster_mask = (labels == main_cluster_id)
    
    main_cluster_points = nearest_points[main_cluster_mask]
    main_cluster_colors = nearest_colors[main_cluster_mask]
    
    print(f"   主要簇: 簇{main_cluster_id}, {len(main_cluster_points)} 点")
    
    # 5. 可视化对比：原始点云 vs 滤出的主要簇
    if visualize:
        # 创建原始点云（灰色半透明）
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(points)
        original_colors_gray = np.ones((len(points), 3)) * 0.3  # 深灰色
        original_pcd.colors = o3d.utility.Vector3dVector(original_colors_gray)
        
        # 创建主要簇点云（使用原始颜色）
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(main_cluster_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(main_cluster_colors.astype(np.float32) / 255.0)
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="原始点云 vs 滤出点云", width=1200, height=800)
        
        # 添加几何体
        vis.add_geometry(original_pcd)
        vis.add_geometry(filtered_pcd)
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis.add_geometry(coordinate_frame)
        
        # 渲染设置
        render_option = vis.get_render_option()
        render_option.point_size = 4.0
        render_option.background_color = np.array([0.05, 0.05, 0.05])
        
        # 相机设置
        view_control = vis.get_view_control()
        view_control.set_front([0.0, 0.0, -1.0])
        view_control.set_up([0.0, -1.0, 0.0])
        view_control.set_lookat(points.mean(axis=0))
        view_control.set_zoom(0.7)
        
        vis.run()
        vis.destroy_window()
    
    return main_cluster_points, main_cluster_colors

def visualize_pointcloud_o3d(points, colors, window_name="Point Cloud", point_size=3.0):
    """使用 Open3D 可视化点云"""
    if len(points) == 0:
        print("❌ 点云为空")
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1200, height=800)
    vis.add_geometry(pcd)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)
    
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    view_control = vis.get_view_control()
    view_control.set_front([0.0, 0.0, -1.0])
    view_control.set_up([0.0, -1.0, 0.0])
    view_control.set_lookat([0.0, 0.0, 0.5])
    view_control.set_zoom(0.8)
    
    vis.run()
    vis.destroy_window()

def visualize_det_seg_results(color_image, bboxes, masks, confidences, window_name="检测结果", show_mask=True, show_bbox=True):
    """可视化YOLO检测和分割结果"""
    vis_image = color_image.copy()
    h, w = vis_image.shape[:2]
    mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
    
    colors_palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]
    
    for i, (bbox, mask, conf) in enumerate(zip(bboxes, masks, confidences)):
        color = colors_palette[i % len(colors_palette)]
        
        if show_mask:
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask.astype(np.float32), (w, h), 
                                         interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask
            
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(vis_image)
            colored_mask[binary_mask == 1] = color
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 1.0, 0)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color, 2)
        
        if show_bbox:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            label = f"Lid {i+1}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(vis_image, (x1, y1 - text_h - baseline - 5),
                         (x1 + text_w, y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - baseline - 2),
                       font, font_scale, (255, 255, 255), thickness)
    
    if show_mask:
        alpha = 0.4
        vis_image = cv2.addWeighted(vis_image, 1.0, mask_overlay, alpha, 0)
    
    info_text = f"检测到 {len(bboxes)} 个物体"
    cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 255, 0), 2)
    
    # 安全地创建和显示窗口
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 900)
        cv2.imshow(window_name, vis_image)
        cv2.waitKey(0)
    except Exception as e:
        print(f"❌ 显示窗口时出错: {e}")
    finally:
        # 安全地销毁窗口
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(window_name)
        except cv2.error:
            pass
    
    return vis_image

def visualize_distance_pcd(points, distances, window_name="距离可视化", point_size=3.0):
    """使用Open3D可视化点云，根据距离着色（近红远蓝）"""
    if len(points) == 0:
        print("❌ 点云为空")
        return
    
    print(f"   距离: [{distances.min():.3f}m, {distances.max():.3f}m], 均值={distances.mean():.3f}m")
    
    dist_min = distances.min()
    dist_max = distances.max()
    
    if dist_max - dist_min < 1e-6:
        normalized_distances = np.ones_like(distances) * 0.5
    else:
        normalized_distances = (distances - dist_min) / (dist_max - dist_min)
    
    colors = np.zeros((len(points), 3))
    for i, norm_dist in enumerate(normalized_distances):
        inv_dist = 1.0 - norm_dist
        if inv_dist < 0.5:
            ratio = inv_dist * 2.0
            colors[i] = [0, ratio, 1.0 - ratio]
        else:
            ratio = (inv_dist - 0.5) * 2.0
            colors[i] = [ratio, 1.0 - ratio, 0]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1200, height=800)
    vis.add_geometry(pcd)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)
    
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    
    view_control = vis.get_view_control()
    view_control.set_front([0.0, 0.0, -1.0])
    view_control.set_up([0.0, -1.0, 0.0])
    view_control.set_lookat(points.mean(axis=0))
    view_control.set_zoom(0.7)
    
    vis.run()
    vis.destroy_window()

def visualize_height_gradient_pcd(points, window_name="高度渐变色可视化", point_size=4.0):
    """
    使用Open3D可视化点云，根据高度（Z坐标）着色
    低处：蓝色 -> 中间：绿色 -> 高处：红色
    
    参数:
        points: 点云坐标 (N, 3)
        window_name: 窗口名称
        point_size: 点云大小
    """
    if len(points) == 0:
        print("❌ 点云为空")
        return
    
    # 获取Z坐标（高度）
    heights = points[:, 2]
    height_min = heights.min()
    height_max = heights.max()
    height_mean = heights.mean()
    
    print(f"🎨 高度渐变色可视化:")
    print(f"   高度范围: [{height_min:.3f}m, {height_max:.3f}m], 均值={height_mean:.3f}m")
    
    # 归一化高度到 [0, 1]
    if height_max - height_min < 1e-6:
        normalized_heights = np.ones_like(heights) * 0.5
    else:
        normalized_heights = (heights - height_min) / (height_max - height_min)
    
    # 创建渐变色：蓝色(低) -> 绿色(中) -> 红色(高)
    colors = np.zeros((len(points), 3))
    for i, norm_height in enumerate(normalized_heights):
        if norm_height < 0.5:
            # 蓝色到绿色：0.0-0.5
            ratio = norm_height * 2.0  # 0.0 -> 1.0
            colors[i] = [0, ratio, 1.0 - ratio]  # [0,0,1] -> [0,1,0]
        else:
            # 绿色到红色：0.5-1.0
            ratio = (norm_height - 0.5) * 2.0  # 0.0 -> 1.0
            colors[i] = [ratio, 1.0 - ratio, 0]  # [0,1,0] -> [1,0,0]
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900)
    vis.add_geometry(pcd)
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)
    
    # 渲染设置
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    
    # 相机设置
    view_control = vis.get_view_control()
    view_control.set_front([0.0, 0.0, -1.0])
    view_control.set_up([0.0, -1.0, 0.0])
    view_control.set_lookat(points.mean(axis=0))
    view_control.set_zoom(0.6)
    
    print(f"   蓝色: 最低点 ({height_min:.3f}m)")
    print(f"   绿色: 中间高度 ({height_mean:.3f}m)")
    print(f"   红色: 最高点 ({height_max:.3f}m)")
    
    vis.run()
    vis.destroy_window()

def extract_top_height_points(points, colors, height_percentage=0.1):
    """
    提取高度前height_percentage%的点云
    
    参数:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)
        height_percentage: 保留的高度百分比 (0.0-1.0)
    
    返回:
        top_points: 高度前percentage%的点云坐标
        top_colors: 高度前percentage%的点云颜色
        bbox: 边界框 [min_x, min_y, min_z, max_x, max_y, max_z]
    """
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
    
    print(f"📦 提取高度前 {height_percentage*100:.1f}% 点云:")
    print(f"   原始点数: {len(points)}")
    print(f"   提取点数: {len(top_points)}")
    print(f"   高度范围: [{heights[top_height_indices].min():.3f}m, {heights[top_height_indices].max():.3f}m]")
    print(f"   边界框: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}] -> [{bbox[3]:.3f}, {bbox[4]:.3f}, {bbox[5]:.3f}]")
    
    return top_points, top_colors, bbox

def visualize_bbox_selection(points, colors, bbox, window_name="矩形框选点云", point_size=4.0):
    """
    可视化矩形框选的点云，用RGB着色
    
    参数:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3) - RGB格式
        bbox: 边界框 [min_x, min_y, min_z, max_x, max_y, max_z]
        window_name: 窗口名称
        point_size: 点云大小
    """
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
    
    print(f"🎨 RGB着色可视化:")
    print(f"   点云数量: {len(points)}")
    print(f"   颜色范围: RGB [0-255]")
    if bbox is not None:
        print(f"   边界框尺寸: [{bbox[3]-bbox[0]:.3f}, {bbox[4]-bbox[1]:.3f}, {bbox[5]-bbox[2]:.3f}]")
    
    vis.run()
    vis.destroy_window()

def extract_lid_center_from_bbox(points, bbox, num_points=10):
    """
    从矩形框提取盖子中心点
    
    参数:
        points: 点云坐标 (N, 3)
        bbox: 边界框 [min_x, min_y, min_z, max_x, max_y, max_z]
        num_points: 取平均的点数
    
    返回:
        lid_center: 盖子中心点坐标 [x, y, z]
        local_coordinate_system: 局部坐标系矩阵 (3x3) [X_axis, Y_axis, Z_axis]
    """
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
    
    print(f"📏 边界框分析:")
    print(f"   X方向尺寸: {x_size:.3f}m")
    print(f"   Y方向尺寸: {y_size:.3f}m")
    print(f"   长边方向: {'X' if x_size >= y_size else 'Y'}")
    
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
    
    print(f"🎯 盖子中心点计算:")
    print(f"   长边中心: {'X=' + str(long_edge_center_x) if long_axis == 0 else 'Y=' + str(long_edge_center_y)}")
    print(f"   候选点数: {len(long_edge_points)}")
    print(f"   选择点数: {len(selected_points)}")
    
    # 打印选中的点的XYZ坐标
    print(f"   选中的点云坐标:")
    for i, point in enumerate(selected_points):
        print(f"     点{i+1}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
    
    print(f"   盖子中心: [{lid_center[0]:.3f}, {lid_center[1]:.3f}, {lid_center[2]:.3f}]")
    
    # 验证中心点是否在长边中心附近
    if long_axis == 0:
        center_offset = abs(lid_center[0] - long_edge_center_x)
        print(f"   X方向偏移: {center_offset:.3f}m (长边中心: {long_edge_center_x:.3f})")
    else:
        center_offset = abs(lid_center[1] - long_edge_center_y)
        print(f"   Y方向偏移: {center_offset:.3f}m (长边中心: {long_edge_center_y:.3f})")
    
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
    
    print(f"📐 局部坐标系:")
    print(f"   X轴 (长边方向): [{x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}]")
    print(f"   Y轴 (垂直长边): [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}]")
    print(f"   Z轴 (垂直向上): [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
    
    return lid_center, local_coordinate_system

def visualize_lid_center(points, colors, lid_center, local_coordinate_system, window_name="盖子中心点可视化", point_size=3.0):
    """
    可视化盖子中心点和局部坐标系
    
    参数:
        points: 原始点云坐标 (N, 3)
        colors: 原始点云颜色 (N, 3) - RGB格式
        lid_center: 盖子中心点坐标 [x, y, z]
        local_coordinate_system: 局部坐标系矩阵 (3x3)
        window_name: 窗口名称
        point_size: 点云大小
    """
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
    
    print(f"🎨 盖子中心点可视化:")
    print(f"   中心点: [{lid_center[0]:.3f}, {lid_center[1]:.3f}, {lid_center[2]:.3f}]")
    print(f"   红色球: 盖子中心点")
    print(f"   红色轴: X轴 (长边方向)")
    print(f"   绿色轴: Y轴 (垂直长边)")
    print(f"   蓝色轴: Z轴 (垂直向上)")
    
    vis.run()
    vis.destroy_window()

def cleanup_all_windows():
    """清理所有OpenCV窗口"""
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"清理窗口时出错: {e}")

def main():
    """主函数"""
    try:
        # 相机内参
        cam_intrinsics = np.array([
            [652.76428223,   0.,         650.07250977],
            [  0.,         651.92443848, 366.10205078],
            [  0.,           0.,           1.        ]
        ])
        # 手眼标定
        hand_eye_matrix = np.array([
            [ 0.01949938,  0.99822277,  0.05631227,  0.0758227 ],
            [-0.99977,     0.01997063, -0.00781785,  0.05666132],
            [-0.00892854, -0.05614688,  0.9983826,  -0.10319311],
            [ 0.,          0.,          0.,          1.        ]
        ])
        # 机器人4x4齐次变换矩阵
        robot_matrix = np.array([
            [-0.92798,  0.24683,  -0.27916,  -0.50775],
            [ 0.28297,  0.95422,  -0.096935, -0.12591],
            [ 0.24245, -0.16895,  -0.95534,   0.42253],
            [ 0.0,      0.0,       0.0,       1.0]
        ], dtype=np.float64)
        
        # 打印变换矩阵信息s
        print_transformation_info(robot_matrix, hand_eye_matrix)
        color_image, depth_image = load_image_from_path("lid/color_20251022_170325.png", "lid/depth_20251022_170325.png")
        
        detection_result = detect_with_yolo(color_image, "weights/best1021.pt", 0.7, target_class="lid")
        
        # 不要进行深度修复！

        masks = detection_result['masks']
        bboxes = detection_result['bboxes']
        confidences = detection_result['confidences']

        # masks = refine_masks_with_morphology(
        #     masks, 
        #     erode_kernel_size=7,    # 腐蚀核大小（可调整）
        #     close_kernel_size=5,    # 闭运算核大小
        #     iterations=2            # 腐蚀迭代次数（可调整）
        # )
        
        for mask in masks:
            # 提取前30%的点
            points, colors = depth_to_pointcloud(depth_image, mask, color_image)
            # points = points[:int(len(points) * 0.9)]
            # colors = colors[:int(len(colors) * 0.9)]

            # 保存原始点云（相机坐标系）
            original_points = points.copy()
            
            # 转换到机械臂基坐标系：pose_matrix = robot_pose_matrix @ hand_eye_matrix @ T_object2cam
            transformed_points = transform_points_to_robot_base(points, robot_matrix, hand_eye_matrix, cam_intrinsics)
            
            # 可视化坐标变换对比
            # visualize_coordinate_transformation(original_points, transformed_points, "相机坐标系 vs 机械臂基坐标系")
            
            # 使用高度渐变色可视化转换后的点云
            # visualize_height_gradient_pcd(transformed_points, "机械臂基坐标系 - 高度渐变色")
            
            # 提取高度前10%的点云并用矩形框框选
            top_points, top_colors, bbox = extract_top_height_points(transformed_points, colors, height_percentage=0.5)
            
            # 可视化矩形框选的点云（RGB着色）
            if len(top_points) > 0:
                # visualize_bbox_selection(top_points, top_colors, bbox, "高度前10%点云 - 矩形框选RGB着色")
                
                # 提取盖子中心点和局部坐标系
                lid_center, local_coordinate_system = extract_lid_center_from_bbox(top_points, bbox, num_points=10)
                
                # 可视化盖子中心点和局部坐标系（使用完整RGB点云）
                visualize_lid_center(transformed_points, colors, lid_center, local_coordinate_system, "盖子中心点与局部坐标系")
                
                # 将机械臂基坐标系中的位姿转换回相机坐标系用于可视化
                # 计算逆变换：T_cam = T_hand_eye^(-1) @ T_robot^(-1) @ T_base
                robot_inv = np.linalg.inv(robot_matrix)
                hand_eye_inv = np.linalg.inv(hand_eye_matrix)
                
                # 创建机械臂基坐标系中的位姿矩阵
                base_pose_matrix = np.eye(4)
                base_pose_matrix[:3, :3] = local_coordinate_system
                base_pose_matrix[:3, 3] = lid_center
                
                # 转换到相机坐标系
                camera_pose_matrix = hand_eye_inv @ robot_inv @ base_pose_matrix
                
                # 在图像上绘制位姿轴
                image_with_axes = draw_pose_axes(color_image, camera_pose_matrix, cam_intrinsics, axis_length=0.05, thickness=3)
                
                # 安全地显示带位姿轴的图像
                try:
                    cv2.namedWindow("位姿轴可视化", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("位姿轴可视化", 1200, 900)
                    cv2.imshow("位姿轴可视化", image_with_axes)
                    cv2.waitKey(0)
                except Exception as e:
                    print(f"❌ 显示位姿轴窗口时出错: {e}")
                finally:
                    # 安全地销毁窗口
                    try:
                        if cv2.getWindowProperty("位姿轴可视化", cv2.WND_PROP_VISIBLE) >= 1:
                            cv2.destroyWindow("位姿轴可视化")
                    except cv2.error:
                        pass
            
            # 保存转换后的点云
            save_transformed_pointcloud(transformed_points, colors, "transformed_pointcloud.ply")
            
            # 计算物体中心在机械臂基坐标系中的位置
            object_center = calculate_object_center(transformed_points)
            print(f"\n🎯 物体中心位置 (机械臂基坐标系):")
            print(f"   X: {object_center[0]:.3f} m")
            print(f"   Y: {object_center[1]:.3f} m") 
            print(f"   Z: {object_center[2]:.3f} m")
            
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理所有窗口
        cleanup_all_windows()
        print("✅ 程序结束，所有窗口已清理")

def save_transformed_pointcloud(points, colors, filename):
    """
    保存转换后的点云到文件
    
    参数:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)
        filename: 保存文件名
    """
    if len(points) == 0:
        print("❌ 点云为空，无法保存")
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
    
    o3d.io.write_point_cloud(filename, pcd)
    print(f"✅ 转换后点云已保存: {filename}")

def calculate_object_center(points):
    """
    计算物体中心位置
    
    参数:
        points: 点云坐标 (N, 3)
    
    返回:
        center: 物体中心坐标 [x, y, z]
    """
    if len(points) == 0:
        return np.array([0, 0, 0])
    
    center = np.mean(points, axis=0)
    return center

def draw_pose_axes(image, pose_matrix, camera_intrinsics, axis_length=0.05, thickness=3):
    """
    在图像上绘制位姿轴（X红、Y绿、Z蓝）
    
    参数:
        image: 输入图像 (H, W, 3)
        pose_matrix: 4x4位姿矩阵
        camera_intrinsics: 相机内参矩阵 (3x3)
        axis_length: 轴长度（米）
        thickness: 线条粗细
    
    返回:
        image_with_axes: 绘制了坐标轴的图像
    """
    # 获取位姿矩阵的旋转和平移部分
    rotation_matrix = pose_matrix[:3, :3]
    translation = pose_matrix[:3, 3]
    
    print(f"🔍 位姿轴调试信息:")
    print(f"   位姿矩阵位置: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
    print(f"   轴长度: {axis_length}m")
    
    # 定义坐标轴方向（在物体坐标系中）
    axes_3d = np.array([
        [0, 0, 0],           # 原点
        [axis_length, 0, 0], # X轴
        [0, axis_length, 0], # Y轴
        [0, 0, axis_length]  # Z轴
    ])
    
    # 将坐标轴转换到相机坐标系
    axes_camera = (rotation_matrix @ axes_3d.T).T + translation
    
    print(f"   相机坐标系中的轴:")
    for i, axis in enumerate(['原点', 'X轴', 'Y轴', 'Z轴']):
        print(f"     {axis}: [{axes_camera[i][0]:.3f}, {axes_camera[i][1]:.3f}, {axes_camera[i][2]:.3f}]")
    
    # 投影到图像平面
    axes_2d = []
    for i, point_3d in enumerate(axes_camera):
        if point_3d[2] > 0:  # 确保Z坐标为正（在相机前方）
            # 使用相机内参投影
            x_2d = (camera_intrinsics[0, 0] * point_3d[0] / point_3d[2] + camera_intrinsics[0, 2])
            y_2d = (camera_intrinsics[1, 1] * point_3d[1] / point_3d[2] + camera_intrinsics[1, 2])
            axes_2d.append([int(x_2d), int(y_2d)])
            print(f"     投影点{i}: ({int(x_2d)}, {int(y_2d)})")
        else:
            axes_2d.append([-1, -1])  # 标记为无效点
            print(f"     投影点{i}: 无效 (Z<0)")
    
    # 绘制坐标轴
    image_with_axes = image.copy()
    h, w = image.shape[:2]
    
    if len(axes_2d) >= 4:
        origin = axes_2d[0]
        x_end = axes_2d[1]
        y_end = axes_2d[2]
        z_end = axes_2d[3]
        
        print(f"   图像尺寸: {w}x{h}")
        print(f"   投影点: 原点{origin}, X{x_end}, Y{y_end}, Z{z_end}")
        
        # 检查所有点是否在图像范围内
        valid_points = all(0 <= p[0] < w and 0 <= p[1] < h and p[0] != -1 and p[1] != -1 
                          for p in [origin, x_end, y_end, z_end])
        
        if valid_points:
            # X轴（红色）
            cv2.arrowedLine(image_with_axes, tuple(origin), tuple(x_end), (0, 0, 255), thickness)
            cv2.putText(image_with_axes, 'X', tuple(x_end), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Y轴（绿色）
            cv2.arrowedLine(image_with_axes, tuple(origin), tuple(y_end), (0, 255, 0), thickness)
            cv2.putText(image_with_axes, 'Y', tuple(y_end), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Z轴（蓝色）
            cv2.arrowedLine(image_with_axes, tuple(origin), tuple(z_end), (255, 0, 0), thickness)
            cv2.putText(image_with_axes, 'Z', tuple(z_end), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            print(f"✅ 位姿轴绘制完成:")
            print(f"   原点: ({origin[0]}, {origin[1]})")
            print(f"   X轴: ({x_end[0]}, {x_end[1]}) - 红色")
            print(f"   Y轴: ({y_end[0]}, {y_end[1]}) - 绿色")
            print(f"   Z轴: ({z_end[0]}, {z_end[1]}) - 蓝色")
        else:
            print("❌ 位姿轴超出图像范围，无法绘制")
            # 显示超出范围的点
            for i, (name, point) in enumerate([("原点", origin), ("X轴", x_end), ("Y轴", y_end), ("Z轴", z_end)]):
                if point[0] == -1 or point[1] == -1:
                    print(f"   {name}: 无效点")
                elif not (0 <= point[0] < w and 0 <= point[1] < h):
                    print(f"   {name}: ({point[0]}, {point[1]}) 超出范围 [0-{w}, 0-{h}]")
    
    return image_with_axes

def print_transformation_info(robot_matrix, hand_eye_matrix):
    """
    打印变换矩阵信息
    
    参数:
        robot_matrix: 机械臂位姿矩阵
        hand_eye_matrix: 手眼标定矩阵
    """
    print(f"\n📊 变换矩阵信息:")
    print(f"   机械臂位姿矩阵:")
    print(f"     位置: [{robot_matrix[0,3]:.3f}, {robot_matrix[1,3]:.3f}, {robot_matrix[2,3]:.3f}]")
    print(f"   手眼标定矩阵:")
    print(f"     位置: [{hand_eye_matrix[0,3]:.3f}, {hand_eye_matrix[1,3]:.3f}, {hand_eye_matrix[2,3]:.3f}]")
    
    # 计算组合变换矩阵
    combined_matrix = robot_matrix @ hand_eye_matrix
    print(f"   组合变换矩阵:")
    print(f"     位置: [{combined_matrix[0,3]:.3f}, {combined_matrix[1,3]:.3f}, {combined_matrix[2,3]:.3f}]")


        
if __name__ == "__main__":
    
    # 运行主程序
    main()
