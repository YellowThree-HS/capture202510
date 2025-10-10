

import cv2
import os
import time
import numpy as np
from datetime import datetime
from lib.camera import Camera
from lib.yolo_and_sam import YOLOSegmentator
# from lib.mask2pose import mask2pose, visualize_result


def main():
    temp_image_path = "images/cup1.jpg"  # 替换为你的测试图像路径
    categories_to_find = ['spoon','cup']
    segmentator = YOLOSegmentator()
    result = segmentator.detect_and_segment_all(
        image_path=temp_image_path,
        categories=categories_to_find,
        save_result=True
    )
    print("Detection and segmentation result:", result)


if __name__ == "__main__":
    main()
