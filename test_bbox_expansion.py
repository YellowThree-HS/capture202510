#!/usr/bin/env python3
"""
测试检测框扩展功能
"""

import cv2
import numpy as np
import sys
import os

# 添加lib目录到路径
sys.path.append('lib')

from lib.yolo_and_sam import YOLOSegmentator

def test_bbox_expansion():
    """测试检测框扩展功能"""
    
    print("🧪 测试检测框扩展功能")
    print("=" * 50)
    
    # 初始化检测器
    print("1. 初始化检测器...")
    segmentator = YOLOSegmentator()
    
    # 测试图像路径
    test_image_path = 'test/color.png'
    
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return
    
    print(f"2. 处理测试图像: {test_image_path}")
    
    # 执行检测和分割
    result = segmentator.detect_and_segment_all(
        image=test_image_path,
        categories=['cup'],
        save_result=True
    )
    
    if result['success']:
        print("✅ 检测和分割成功!")
        print(f"   检测到 {len(result['objects'])} 个物体")
        
        for i, obj in enumerate(result['objects']):
            print(f"   物体 {i+1}: {obj['class']} (置信度: {obj['confidence']:.2f})")
            print(f"   原始边界框: {obj['bbox_xyxy']}")
            if 'bbox_xyxy_expanded' in obj:
                print(f"   扩展边界框: {obj['bbox_xyxy_expanded']}")
            if obj['mask'] is not None:
                print(f"   掩码尺寸: {obj['mask'].shape}")
            else:
                print("   ⚠️ 未获取到掩码")
    else:
        print("❌ 检测和分割失败")
    
    print("\n📁 结果已保存到 result/ 文件夹")
    print("🔍 请查看以下文件:")
    print("   - det_*.png: 检测结果")
    print("   - seg_*.png: 分割结果") 
    print("   - combined_*.jpg: 合并可视化")
    print("   - mask_*.png: 掩码调试文件")

if __name__ == "__main__":
    test_bbox_expansion()
