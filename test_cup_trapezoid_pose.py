#!/usr/bin/env python3
"""
使用test文件夹下的RGB和深度图测试新的梯形轮廓位姿估计方法
显示完整的中间过程和最终结果
"""

import cv2
import numpy as np
import sys
import os
import open3d as o3d

# 添加lib目录到路径
sys.path.append('lib')

from lib.camera import Camera
from lib.yolo_and_sam import YOLOSegmentator
from lib.mask2pose import (
    mask2pose, 
    extract_cup_side_contour, 
    fit_trapezoid, 
    calculate_cup_pose_from_trapezoid,
    create_point_cloud,
    draw_pose_axes
)

def visualize_contour_extraction(mask, color_image, output_dir="result"):
    """可视化轮廓提取过程"""
    
    print("📊 可视化轮廓提取过程...")
    
    # 1. 预处理掩码
    mask_2d = mask[:, :, 0] if len(mask.shape) == 3 else mask
    mask_2d = mask_2d.astype(np.uint8)
    
    # 检查并调整掩码尺寸
    h, w = color_image.shape[:2]
    mask_h, mask_w = mask_2d.shape[:2]
    
    if (mask_h, mask_w) != (h, w):
        print(f"   ⚠️ 调整掩码尺寸以匹配彩色图像: ({mask_h}, {mask_w}) -> ({h}, {w})")
        mask_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 形态学操作：开运算去除噪点
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_2d, cv2.MORPH_OPEN, kernel)
    
    # 2. 查找轮廓
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("❌ 未找到轮廓")
        return None
    
    # 选择最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 3. 轮廓近似
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 4. 梯形拟合
    trapezoid_points = fit_trapezoid(approx_contour)
    
    # 5. 创建可视化图像
    vis_img = color_image.copy()
    
    # 绘制原始轮廓
    cv2.drawContours(vis_img, [largest_contour], -1, (0, 255, 0), 2)
    
    # 绘制近似轮廓
    cv2.drawContours(vis_img, [approx_contour], -1, (255, 0, 0), 2)
    
    # 绘制梯形
    if trapezoid_points is not None:
        # 绘制梯形边框（粗红线）
        cv2.polylines(vis_img, [trapezoid_points.astype(int)], True, (0, 0, 255), 4)
        
        # 填充梯形区域（半透明红色）
        trapezoid_filled = vis_img.copy()
        cv2.fillPoly(trapezoid_filled, [trapezoid_points.astype(int)], (0, 0, 255))
        vis_img = cv2.addWeighted(vis_img, 0.7, trapezoid_filled, 0.3, 0)
        
        # 标记梯形顶点（黄色圆圈 + 编号）
        vertex_names = ['左上', '右上', '右下', '左下']
        for i, point in enumerate(trapezoid_points):
            cv2.circle(vis_img, tuple(point.astype(int)), 12, (0, 255, 255), -1)
            cv2.circle(vis_img, tuple(point.astype(int)), 12, (0, 0, 0), 2)
            
            # 显示顶点编号和名称
            text = f"{i}: {vertex_names[i]}"
            cv2.putText(vis_img, text, (int(point[0])+15, int(point[1])-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_img, text, (int(point[0])+15, int(point[1])-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 绘制对称轴（紫色线）
        top_mid = (trapezoid_points[0] + trapezoid_points[1]) / 2
        bottom_mid = (trapezoid_points[2] + trapezoid_points[3]) / 2
        cv2.line(vis_img, tuple(top_mid.astype(int)), tuple(bottom_mid.astype(int)), (255, 0, 255), 4)
        
        # 绘制对称轴标签
        axis_center = (top_mid + bottom_mid) / 2
        cv2.putText(vis_img, "对称轴", (int(axis_center[0])+10, int(axis_center[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, "对称轴", (int(axis_center[0])+10, int(axis_center[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 绘制中心点（青色圆圈）
        center = np.mean(trapezoid_points, axis=0)
        cv2.circle(vis_img, tuple(center.astype(int)), 15, (255, 255, 0), -1)
        cv2.circle(vis_img, tuple(center.astype(int)), 15, (0, 0, 0), 2)
        cv2.putText(vis_img, "中心", (int(center[0])+20, int(center[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, "中心", (int(center[0])+20, int(center[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 绘制上边和下边（绿色和蓝色）
        cv2.line(vis_img, tuple(trapezoid_points[0].astype(int)), tuple(trapezoid_points[1].astype(int)), (0, 255, 0), 3)
        cv2.line(vis_img, tuple(trapezoid_points[2].astype(int)), tuple(trapezoid_points[3].astype(int)), (255, 0, 0), 3)
        
        # 添加梯形尺寸信息
        top_width = np.linalg.norm(trapezoid_points[1] - trapezoid_points[0])
        bottom_width = np.linalg.norm(trapezoid_points[2] - trapezoid_points[3])
        height = np.linalg.norm(bottom_mid - top_mid)
        
        info_text = f"上宽: {top_width:.1f}px"
        cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        info_text = f"下宽: {bottom_width:.1f}px"
        cv2.putText(vis_img, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis_img, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        info_text = f"高度: {height:.1f}px"
        cv2.putText(vis_img, info_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(vis_img, info_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # 判断梯形形状
        if top_width < bottom_width:
            shape_text = "✓ 梯形形状正确 (上窄下宽)"
            color = (0, 255, 0)
        else:
            shape_text = "⚠ 可能不是杯子 (上宽下窄)"
            color = (0, 0, 255)
        
        cv2.putText(vis_img, shape_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis_img, shape_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # 保存可视化结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "contour_extraction_visualization.jpg")
    cv2.imwrite(output_path, vis_img)
    print(f"✅ 轮廓提取可视化已保存: {output_path}")
    
    return vis_img, trapezoid_points

def visualize_point_cloud_extraction(mask, depth_image, color_image, intrinsics, output_dir="result"):
    """可视化点云提取过程"""
    
    print("📊 可视化点云提取过程...")
    
    # 确保mask和depth是2D的
    mask_2d = mask[:, :, 0] if len(mask.shape) == 3 else mask
    depth_2d = depth_image[:, :, 0] if len(depth_image.shape) == 3 else depth_image
    
    # 检查并调整图像尺寸
    h, w = color_image.shape[:2]
    mask_h, mask_w = mask_2d.shape[:2]
    depth_h, depth_w = depth_2d.shape[:2]
    
    print(f"   图像尺寸检查:")
    print(f"     彩色图像: {h}x{w}")
    print(f"     掩码: {mask_h}x{mask_w}")
    print(f"     深度图: {depth_h}x{depth_w}")
    
    # 调整掩码尺寸以匹配彩色图像
    if (mask_h, mask_w) != (h, w):
        print(f"   ⚠️ 调整掩码尺寸: ({mask_h}, {mask_w}) -> ({h}, {w})")
        mask_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 调整深度图尺寸以匹配彩色图像
    if (depth_h, depth_w) != (h, w):
        print(f"   ⚠️ 调整深度图尺寸: ({depth_h}, {depth_w}) -> ({h}, {w})")
        depth_2d = cv2.resize(depth_2d, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 应用掩码
    color_masked = color_image * mask_2d[:, :, np.newaxis]
    depth_masked = depth_2d * mask_2d
    
    # 创建点云
    point_cloud = create_point_cloud(depth_masked, intrinsics, color_masked)
    
    if len(point_cloud.points) < 10:
        print("❌ 点云数据太少")
        return None
    
    print(f"✅ 成功提取 {len(point_cloud.points)} 个点云点")
    
    # 保存点云
    os.makedirs(output_dir, exist_ok=True)
    point_cloud_path = os.path.join(output_dir, "cup_point_cloud.ply")
    o3d.io.write_point_cloud(point_cloud_path, point_cloud)
    print(f"✅ 杯子点云已保存: {point_cloud_path}")
    
    # 创建简单的点云可视化（使用matplotlib）
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制点云
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=1, alpha=0.6)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('杯子点云 (3D视图)')
        
        # 保存3D可视化
        plt_path = os.path.join(output_dir, "cup_point_cloud_3d.png")
        plt.savefig(plt_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 点云3D可视化已保存: {plt_path}")
        
    except ImportError:
        print("⚠️ matplotlib未安装，跳过3D可视化")
    
    return point_cloud

def main():
    """主函数"""
    
    print("🚀 开始测试梯形轮廓位姿估计方法")
    print("=" * 60)
    
    # 1. 初始化相机和检测器
    print("1. 初始化系统...")
    cam = Camera(camera_model='D405')
    segmentator = YOLOSegmentator()
    
    # 2. 读取测试图像
    print("\n2. 读取测试图像...")
    color_image_path = 'test/color.png'
    depth_image_path = 'test/depth.png'
    
    if not os.path.exists(color_image_path) or not os.path.exists(depth_image_path):
        print(f"❌ 测试图像文件不存在:")
        print(f"   {color_image_path}: {os.path.exists(color_image_path)}")
        print(f"   {depth_image_path}: {os.path.exists(depth_image_path)}")
        return
    
    color_image = cv2.imread(color_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    
    print(f"✅ 成功读取图像:")
    print(f"   彩色图像: {color_image.shape}")
    print(f"   深度图像: {depth_image.shape}")
    
    # 3. 处理深度图
    print("\n3. 处理深度图...")
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]
    
    # RealSense D405的深度比例
    depth_scale = 0.0001
    depth_image = depth_image.astype(np.float32) * depth_scale
    
    print(f"   深度范围: [{depth_image.min():.3f}, {depth_image.max():.3f}] 米")
    
    # 4. 检测和分割杯子
    print("\n4. 检测和分割杯子...")
    categories_to_find = ['cup']
    
    result = segmentator.detect_and_segment_all(
        image=color_image,
        categories=categories_to_find,
        save_result=True
    )
    
    if not result['success'] or len(result['objects']) == 0:
        print("❌ 未检测到杯子")
        return
    
    cup_obj = result['objects'][0]
    print(f"✅ 检测到杯子:")
    print(f"   类别: {cup_obj['class']}")
    print(f"   置信度: {cup_obj['confidence']:.2f}")
    print(f"   边界框: {cup_obj['bbox_xyxy']}")
    print(f"   掩码尺寸: {cup_obj['mask'].shape}")
    
    # 5. 可视化轮廓提取过程
    print("\n5. 可视化轮廓提取过程...")
    vis_img, trapezoid_points = visualize_contour_extraction(
        cup_obj['mask'], color_image
    )
    
    # 6. 可视化点云提取过程
    print("\n6. 可视化点云提取过程...")
    intrinsics = cam.get_camera_matrix()
    point_cloud = visualize_point_cloud_extraction(
        cup_obj['mask'], depth_image, color_image, intrinsics
    )
    
    # 7. 测试新的梯形轮廓位姿估计方法
    print("\n7. 测试梯形轮廓位姿估计...")
    
    # 使用新的梯形轮廓方法
    trapezoid_points_new, center_3d, normal_3d, success = extract_cup_side_contour(
        cup_obj['mask'], color_image, depth_image, intrinsics
    )
    
    if success:
        print("✅ 梯形轮廓方法成功!")
        print(f"   梯形顶点: {trapezoid_points_new}")
        print(f"   3D中心: [{center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f}]")
        print(f"   向上向量: [{normal_3d[0]:.3f}, {normal_3d[1]:.3f}, {normal_3d[2]:.3f}]")
    else:
        print("❌ 梯形轮廓方法失败")
    
    # 8. 使用完整的位姿估计方法
    print("\n8. 完整位姿估计...")
    pose, T_object2cam = mask2pose(
        mask=cup_obj['mask'],
        depth_image=depth_image,
        color_image=color_image,
        intrinsics=intrinsics,
        T_cam2base=None,  # 相机坐标系
        object_class=cup_obj['class']
    )
    
    if pose is not None:
        print("✅ 位姿估计成功!")
        print(f"   位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
        print(f"   姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
        
        # 9. 在RGB图上显示坐标轴
        print("\n9. 在RGB图上显示坐标轴...")
        
        # 创建带坐标轴的图像
        axes_img = color_image.copy()
        
        # 绘制坐标轴和梯形
        try:
            axes_img = draw_pose_axes(axes_img, intrinsics, T_object2cam)
            
            # 在坐标轴图像上叠加梯形信息
            axes_img = add_trapezoid_to_image(axes_img, trapezoid_points_new, cup_obj['mask'])
            
            # 保存带坐标轴和梯形的图像
            axes_output_path = "result/cup_with_coordinate_axes_and_trapezoid.jpg"
            cv2.imwrite(axes_output_path, axes_img)
            print(f"✅ 带坐标轴和梯形的图像已保存: {axes_output_path}")
            
        except Exception as e:
            print(f"⚠️ 绘制坐标轴时出错: {e}")
            # 手动绘制简化的坐标轴
            axes_img = draw_simple_coordinate_axes(axes_img, T_object2cam, intrinsics)
            
            # 添加梯形信息
            if trapezoid_points_new is not None:
                axes_img = add_trapezoid_to_image(axes_img, trapezoid_points_new, cup_obj['mask'])
            
            axes_output_path = "result/cup_with_simple_axes_and_trapezoid.jpg"
            cv2.imwrite(axes_output_path, axes_img)
            print(f"✅ 简化坐标轴和梯形图像已保存: {axes_output_path}")
        
    else:
        print("❌ 位姿估计失败")
    
    print("\n🎉 测试完成！")
    print("📁 所有结果已保存到 result/ 文件夹")

def add_trapezoid_to_image(image, trapezoid_points, mask):
    """在图像上添加梯形信息"""
    
    if trapezoid_points is None:
        return image
    
    # 调整梯形点坐标以匹配图像尺寸
    h, w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    
    if (mask_h, mask_w) != (h, w):
        # 计算缩放比例
        scale_x = w / mask_w
        scale_y = h / mask_h
        trapezoid_points = trapezoid_points * np.array([scale_x, scale_y])
    
    # 绘制梯形边框（粗红线）
    cv2.polylines(image, [trapezoid_points.astype(int)], True, (0, 0, 255), 4)
    
    # 标记梯形顶点（黄色圆圈）
    vertex_names = ['左上', '右上', '右下', '左下']
    for i, point in enumerate(trapezoid_points):
        cv2.circle(image, tuple(point.astype(int)), 10, (0, 255, 255), -1)
        cv2.circle(image, tuple(point.astype(int)), 10, (0, 0, 0), 2)
        
        # 显示顶点编号
        text = f"{i}"
        cv2.putText(image, text, (int(point[0])+12, int(point[1])-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, text, (int(point[0])+12, int(point[1])-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 绘制对称轴（紫色线）
    top_mid = (trapezoid_points[0] + trapezoid_points[1]) / 2
    bottom_mid = (trapezoid_points[2] + trapezoid_points[3]) / 2
    cv2.line(image, tuple(top_mid.astype(int)), tuple(bottom_mid.astype(int)), (255, 0, 255), 3)
    
    # 绘制中心点（青色圆圈）
    center = np.mean(trapezoid_points, axis=0)
    cv2.circle(image, tuple(center.astype(int)), 12, (255, 255, 0), -1)
    cv2.circle(image, tuple(center.astype(int)), 12, (0, 0, 0), 2)
    
    # 添加梯形尺寸信息到图像右上角
    top_width = np.linalg.norm(trapezoid_points[1] - trapezoid_points[0])
    bottom_width = np.linalg.norm(trapezoid_points[2] - trapezoid_points[3])
    height = np.linalg.norm(bottom_mid - top_mid)
    
    info_y_start = 30
    cv2.putText(image, f"梯形信息:", (w-200, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(image, f"上宽: {top_width:.1f}px", (w-200, info_y_start+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, f"下宽: {bottom_width:.1f}px", (w-200, info_y_start+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(image, f"高度: {height:.1f}px", (w-200, info_y_start+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # 判断梯形形状
    if top_width < bottom_width:
        shape_text = "梯形正确 (上窄下宽)"
        color = (0, 255, 0)
    else:
        shape_text = "可能不是杯子"
        color = (0, 0, 255)
    
    cv2.putText(image, shape_text, (w-200, info_y_start+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def draw_simple_coordinate_axes(image, pose_matrix, intrinsics, axis_length=0.05):
    """绘制简化的坐标轴"""
    
    # 提取位置和旋转
    position = pose_matrix[:3, 3]
    rotation_matrix = pose_matrix[:3, :3]
    
    # 投影3D点到2D
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 原点
    if position[2] > 0:
        origin_2d = (
            int((position[0] / position[2]) * fx + cx),
            int((position[1] / position[2]) * fy + cy)
        )
        
        # X轴 (红色)
        x_end = position + rotation_matrix[:, 0] * axis_length
        if x_end[2] > 0:
            x_end_2d = (
                int((x_end[0] / x_end[2]) * fx + cx),
                int((x_end[1] / x_end[2]) * fy + cy)
            )
            cv2.arrowedLine(image, origin_2d, x_end_2d, (0, 0, 255), 3)
            cv2.putText(image, "X", (x_end_2d[0]+5, x_end_2d[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Y轴 (绿色)
        y_end = position + rotation_matrix[:, 1] * axis_length
        if y_end[2] > 0:
            y_end_2d = (
                int((y_end[0] / y_end[2]) * fx + cx),
                int((y_end[1] / y_end[2]) * fy + cy)
            )
            cv2.arrowedLine(image, origin_2d, y_end_2d, (0, 255, 0), 3)
            cv2.putText(image, "Y", (y_end_2d[0]+5, y_end_2d[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Z轴 (蓝色)
        z_end = position + rotation_matrix[:, 2] * axis_length
        if z_end[2] > 0:
            z_end_2d = (
                int((z_end[0] / z_end[2]) * fx + cx),
                int((z_end[1] / z_end[2]) * fy + cy)
            )
            cv2.arrowedLine(image, origin_2d, z_end_2d, (255, 0, 0), 3)
            cv2.putText(image, "Z", (z_end_2d[0]+5, z_end_2d[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return image

if __name__ == "__main__":
    main()
