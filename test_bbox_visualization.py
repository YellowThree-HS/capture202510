#!/usr/bin/env python3
"""
测试检测框扩展可视化功能
"""

import cv2
import numpy as np
import sys
import os

# 添加lib目录到路径
sys.path.append('lib')

from lib.yolo_and_sam import YOLOSegmentator

def test_bbox_visualization():
    """测试检测框扩展可视化"""
    
    print("🧪 测试检测框扩展可视化功能")
    print("=" * 50)
    
    # 测试图像路径
    test_image_path = 'test/color.png'
    
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return
    
    print(f"1. 测试图像: {test_image_path}")
    
    # 初始化检测器
    print("\n2. 初始化检测器...")
    segmentator = YOLOSegmentator(
        yolo_weights="weights/yolov8s-world.pt",
        sam_weights="weights/FastSAM-s.pt",
        use_sam2=False
    )
    
    # 执行检测和分割
    print("\n3. 执行检测和分割...")
    result = segmentator.detect_and_segment_all(
        image=test_image_path,
        categories=['cup'],
        save_result=True
    )
    
    if result['success']:
        print("✅ 检测和分割成功!")
        print(f"   检测到 {len(result['objects'])} 个物体")
        
        for i, obj in enumerate(result['objects']):
            print(f"\n   物体 {i+1}: {obj['class']} (置信度: {obj['confidence']:.2f})")
            print(f"   原始边界框: {obj['bbox_xyxy']}")
            if 'bbox_xyxy_expanded' in obj:
                print(f"   扩展边界框: {obj['bbox_xyxy_expanded']}")
                
                # 计算扩展量
                orig = obj['bbox_xyxy']
                exp = obj['bbox_xyxy_expanded']
                expand_x = (exp[0] - orig[0], exp[2] - orig[2])  # 左扩展, 右扩展
                expand_y = (exp[1] - orig[1], exp[3] - orig[3])  # 上扩展, 下扩展
                print(f"   扩展量: 左{expand_x[0]:.0f}px, 右{expand_x[1]:.0f}px, 上{expand_y[0]:.0f}px, 下{expand_y[1]:.0f}px")
            else:
                print("   ⚠️ 未找到扩展边界框")
                
            if obj['mask'] is not None:
                print(f"   掩码尺寸: {obj['mask'].shape}")
            else:
                print("   ⚠️ 未获取到掩码")
        
        # 检查合并可视化文件
        if 'combined_path' in result and result['combined_path']:
            print(f"\n📁 合并可视化已保存: {result['combined_path']}")
            print("🔍 请查看这个文件，应该能看到:")
            print("   - 蓝色实线框: 原始检测框")
            print("   - 黄色虚线框: 扩展后的检测框")
        else:
            print("\n⚠️ 未生成合并可视化文件")
    else:
        print("❌ 检测和分割失败")
    
    print("\n📁 所有结果已保存到 result/ 文件夹")

if __name__ == "__main__":
    test_bbox_visualization()
