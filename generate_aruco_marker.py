#!/usr/bin/env python3
"""
ArUco标记生成器

生成8x8cm的ArUco标记，用于手眼标定验证
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os


def generate_aruco_marker(marker_id=0, marker_size_cm=8, output_dir="aruco_markers"):
    """
    生成单个ArUco标记
    
    参数:
        marker_id: 标记ID
        marker_size_cm: 标记大小（厘米）
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取ArUco字典
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
    
    # 生成标记图像
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, 200)  # 200x200像素
    
    # 保存标记
    marker_filename = os.path.join(output_dir, f"aruco_marker_{marker_id:03d}.png")
    cv2.imwrite(marker_filename, marker_image)
    
    print(f"✓ 生成ArUco标记 {marker_id}: {marker_filename}")
    print(f"  实际大小: {marker_size_cm}cm x {marker_size_cm}cm")
    print(f"  打印时请确保尺寸准确")
    
    return marker_filename


def generate_multiple_markers(num_markers=16, marker_size_cm=8, output_dir="aruco_markers"):
    """
    生成多个ArUco标记
    
    参数:
        num_markers: 标记数量
        marker_size_cm: 标记大小（厘米）
        output_dir: 输出目录
    """
    print(f"正在生成 {num_markers} 个ArUco标记...")
    print(f"标记大小: {marker_size_cm}cm x {marker_size_cm}cm")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    generated_files = []
    
    for i in range(num_markers):
        filename = generate_aruco_marker(i, marker_size_cm, output_dir)
        generated_files.append(filename)
    
    print("-" * 50)
    print(f"✓ 成功生成 {num_markers} 个ArUco标记")
    print("\n使用说明:")
    print("1. 打印这些标记，确保每个标记的实际尺寸为 8cm x 8cm")
    print("2. 将标记粘贴到硬纸板或塑料板上")
    print("3. 运行 aruco_verification.py 进行手眼标定验证")
    print("4. 按空格键检测标记并查看位姿信息")
    
    return generated_files


def main():
    """主函数"""
    print("ArUco标记生成器")
    print("=" * 50)
    
    # 生成一个标记用于测试
    print("生成单个ArUco标记 (ID=0, 8cm x 8cm)...")
    generate_aruco_marker(0, 8)
    
    print("\n" + "="*50)
    print("使用说明:")
    print("1. 打印生成的标记，确保实际尺寸为 8cm x 8cm")
    print("2. 将标记粘贴到硬纸板上")
    print("3. 运行 python aruco_verification.py")
    print("4. 按空格键检测标记并查看位姿信息")
    print("5. 验证手眼标定矩阵的准确性")


if __name__ == "__main__":
    main()

