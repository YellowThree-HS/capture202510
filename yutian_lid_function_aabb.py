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

def load_image_from_path(color_path = "lid/color.png", depth_path = "lid/depth.png"):
    if not os.path.exists(color_path) or not os.path.exists(depth_path):
        print(f"❌ 图像文件不存在")
        return None, None
    
    color_image = cv2.imread(color_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if color_image is None or depth_image is None:
        print(f"❌ 无法读取图像")
        return None, None
    
    print(f"✅ 加载图像: {color_image.shape}, {depth_image.shape}")
    return color_image, depth_image

def detect_with_yolo(color_image, model_path="weights/all.pt", conf_threshold=0.3, target_class="lid"):
    """使用YOLO进行检测，只返回指定类别的结果"""
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    model = YOLO(model_path)
    results = model.predict(
        source=color_image,
        save=False,
        conf=conf_threshold,
        iou=0.7,
        verbose=False
    )
    
    if len(results) == 0 or results[0].masks is None:
        print("❌ 未检测到任何物体")
        return None
    
    result = results[0]
    class_names = result.names
    
    target_class_id = None
    for class_id, class_name in class_names.items():
        if class_name == target_class:
            target_class_id = class_id
            break
    
    if target_class_id is None:
        print(f"❌ 未找到目标类别 '{target_class}'")
        return None
    
    all_bboxes = result.boxes.xyxy.cpu().numpy()
    all_masks = result.masks.data.cpu().numpy()
    all_confs = result.boxes.conf.cpu().numpy()
    all_classes = result.boxes.cls.cpu().numpy()
    
    target_indices = np.where(all_classes == target_class_id)[0]
    
    if len(target_indices) == 0:
        print(f"❌ 未检测到 '{target_class}'")
        return None
    
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
    if masks is None or len(masks) == 0:
        return masks
    
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

def refine_depth_with_promptda(color_path, depth_path, model_path="yutian_submodules/PromptDA/weights/vitl.ckpt", device='cuda'):
    """
    使用 PromptDA 对深度图进行修复
    
    参数:
        color_path: 彩色图像路径
        depth_path: 深度图像路径
        model_path: PromptDA 模型权重路径
        device: 运行设备 ('cuda' 或 'cpu')
    
    返回:
        refined_depth: numpy数组，修复后的深度图，单位米，shape (H, W)
    """
    import torch
    import torch.nn.functional as F
    from promptda.promptda import PromptDA
    from promptda.utils.io_wrapper import load_image, load_depth
    
    if not os.path.exists(color_path) or not os.path.exists(depth_path):
        print(f"❌ 图像文件不存在")
        return None
    
    # 加载图像
    image = load_image(color_path).to(device)
    prompt_depth = load_depth(depth_path).to(device)
    
    # 记录原始尺寸
    _, _, H, W = image.shape
    
    # 调整到 14 的倍数
    patch_size = 14
    H_new = ((H + patch_size - 1) // patch_size) * patch_size
    W_new = ((W + patch_size - 1) // patch_size) * patch_size
    
    # 调整尺寸
    if H != H_new or W != W_new:
        image = F.interpolate(image, size=(H_new, W_new), mode='bilinear', align_corners=False)
        prompt_depth = F.interpolate(prompt_depth, size=(H_new, W_new), mode='nearest')
    
    # 加载模型并预测
    model = PromptDA.from_pretrained(model_path).to(device).eval()
    
    with torch.no_grad():
        depth = model.predict(image, prompt_depth)
    
    # 恢复到原始尺寸
    if H != H_new or W != W_new:
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(0)
        
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)
    
    # 转换为 numpy 数组 (单位: 米)
    if depth.dim() == 4:
        depth_np = depth.squeeze(0).squeeze(0).cpu().numpy()
    elif depth.dim() == 3:
        depth_np = depth.squeeze(0).cpu().numpy()
    else:
        depth_np = depth.cpu().numpy()
    
    print(f"✅ 深度修复完成: {depth_np.shape}, 范围=[{depth_np.min():.3f}, {depth_np.max():.3f}]m")
    
    return depth_np

def depth_to_pointcloud(depth_image, mask, color_image=None, camera_intrinsics=None, depth_scale=0.001):
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
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 900)
    cv2.imshow(window_name, vis_image)
    cv2.waitKey(0)
    
    # 安全地销毁窗口，检查窗口是否存在
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(window_name)
    except cv2.error:
        # 如果窗口不存在或已经被销毁，忽略错误
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

def main():
    """主函数"""
    color_image, depth_image = load_image_from_path("lid/color.png", "lid/depth.png")
    if color_image is None or depth_image is None:
        return
    
    detection_result = detect_with_yolo(color_image, "weights/all500.pt", 0.7, target_class="lid")
    if detection_result is None:
        return
    
    # 不要进行深度修复！

    masks = detection_result['masks']
    bboxes = detection_result['bboxes']
    confidences = detection_result['confidences']

    masks = refine_masks_with_morphology(
        masks, 
        erode_kernel_size=7,    # 腐蚀核大小（可调整）
        close_kernel_size=5,    # 闭运算核大小
        iterations=2            # 腐蚀迭代次数（可调整）
    )

    
    # for i, mask in enumerate(masks):
    #     print(f"\n处理 Lid {i+1}/{len(masks)}")
        
    #     points, colors = depth_to_pointcloud(depth_image, mask, color_image)
    #     if len(points) == 0:
    #         print(f"❌ 点云为空，跳过")
    #         continue

    #     distances = np.linalg.norm(points, axis=1)
    #     visualize_distance_pcd(points, distances, window_name=f"Lid {i+1} 距离")
        
    #     # 提取并可视化关注点云（会显示对比）
    #     points_filtered, colors_filtered = dbscan_nearest_points(
    #         points, colors, 
    #         keep_ratio=0.2, 
    #         eps=0.02, 
    #         min_samples=10,
    #         visualize=True  # 显示原始 vs 滤出对比
    #     )
        
if __name__ == "__main__":
    main()
