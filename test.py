

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
    color_image = cam.get_color_image()
    print(type(color_image))
    depth_image = cam.get_depth_image()
    depth_image_meters = depth_image.astype(np.float32) / 1000.0  # 转换为米

    # color_image = 'images/cup1.jpg'
    T_cam2base = np.eye(4)  # 假设相机位姿已知，这里使用单位矩阵作为示例
    

    categories_to_find = ['spoon','cup']
    segmentator = YOLOSegmentator()
    # result 
    result = segmentator.detect_and_segment_all(
        image=color_image,
        categories=categories_to_find,
        save_result=True
    )
    print("Detection and segmentation result:", result)



    # result[''objects']是一个字典的list，每个字典代表一个物体包含类别、置信度、边界框和掩码
    for idx, obj in enumerate(result['objects']):
        print(f"置信度: {obj['confidence']:.2f}")
        print(f"边界框: {obj['bbox_xyxy']}")
        
        # 如果有掩码，计算位姿
        if obj['mask'] is not None:

            pose, T = mask2pose(
                mask=obj['mask'],
                depth_image=depth_image_meters,
                color_image=color_image,
                intrinsics=cam.get_camera_matrix(),
                T_cam2base=T_cam2base,
                object_class=obj['class']  # 传入物体类别
            )
            
            if pose is not None:
                print(f"   📍 位置 (x, y, z): [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
                print(f"   📐 姿态 (roll, pitch, yaw): [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
            else:
                print(f"   ❌ 位姿估计失败")
        else:
            print(f"   ⚠️ 未获取到掩码")




if __name__ == "__main__":
    main()
