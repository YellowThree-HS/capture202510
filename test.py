

import cv2
import os
import time
import numpy as np
from datetime import datetime
from lib.camera import Camera
from lib.yolo_and_sam import YOLOSegmentator
from lib.mask2pose import mask2pose, visualize_result


def main():



    cam = Camera(camera_model='D435')  # 初始化相机

    T_cam2base = np.eye(4)  # 假设相机位姿已知，这里使用单位矩阵作为示例

    temp_image_path = "images/cup1.jpg"  # 替换为你的测试图像路径
    categories_to_find = ['spoon','cup']
    segmentator = YOLOSegmentator()
    # result 
    result = segmentator.detect_and_segment_all(
        image_path=temp_image_path,
        categories=categories_to_find,
        save_result=True
    )
    print("Detection and segmentation result:", result)




    # result[''objects']是一个字典的list，每个字典代表一个物体包含类别、置信度、边界框和掩码
    for idx, obj in enumerate(result['objects']):
        print(f"   置信度: {obj['confidence']:.2f}")
        print(f"   边界框: {obj['bbox_xyxy']}")
        
        # 如果有掩码，计算位姿
        if obj['mask'] is not None:
            pose_start = time.time()
            pose, T = mask2pose(
                mask=obj['mask'],
                depth_image=depth_image_meters,
                color_image=color_image,
                intrinsics=cam.intrinsics,
                T_cam2base=T_cam2base,
                object_class=obj['class']  # 传入物体类别
            )
            pose_end = time.time()
            
            if pose is not None:
                print(f"   位姿估计耗时: {pose_end - pose_start:.2f}s")
                print(f"   📍 位置 (x, y, z): [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
                print(f"   📐 姿态 (roll, pitch, yaw): [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
                all_poses.append({
                    'class': obj['class'],
                    'pose': pose,
                    'confidence': obj['confidence']
                })
            else:
                print(f"   ❌ 位姿估计失败")
        else:
            print(f"   ⚠️ 未获取到掩码")




if __name__ == "__main__":
    main()
