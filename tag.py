import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
from lib.camera import Camera


cam = Camera(camera_model='D435')  # 初始化相机
# --- 2. ArUco 和相机参数准备 ---

# ArUco 字典
# 你需要知道你的 Tag 属于哪个字典。DICT_6X6_250 是一个非常常见的选择。
# 如果不确定，可以尝试不同的字典，比如 DICT_4X4_50, DICT_5X5_100 等。
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
parameters = aruco.DetectorParameters()
camera_matrix = cam.get_camera_matrix()
dist_coeffs = cam.get_distortion_coeffs()


# --- 3. 主循环 ---
try:
    while True:
        color_image, depth_image = cam.get_frames()

        # 将彩色图像转为灰度图（ArUco检测在灰度图上进行）
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # 如果检测到 marker，就在图像上绘制出来
            aruco.drawDetectedMarkers(color_image, corners, ids)
            
            # 估计位姿
            # rvecs: 旋转向量 (Rotation vectors)
            # tvecs: 平移向量 (Translation vectors)
            # markerLength: 你的 ArUco 标记的实际物理尺寸（单位：米）。!! 这个值必须准确 !!
            markerLength = 0.05  # 示例：假设标记的边长为 5 厘米 (0.05米)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs)

            # 遍历所有检测到的 marker
            for i in range(len(ids)):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                
                # 在图像上绘制坐标轴来可视化位姿
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.05) # 轴的长度为 5cm

                # 打印 Tag 的信息
                tag_id = ids[i][0]
                # tvec 是标记中心相对于相机坐标系的位置向量 (x, y, z)
                # tvec[2] 就是 z 轴上的距离，即标记到相机的直线距离
                distance = np.linalg.norm(tvec) # 计算到相机的欧式距离

                print(f"--- Detected Tag ---")
                print(f"  ID: {tag_id}") # 这就是你需要的 Tag 标号
                print(f"  Distance to camera: {distance:.3f} meters")
                print(f"  Translation Vector (tvec): {tvec.flatten()}") # (x, y, z) in meters
                print(f"  Rotation Vector (rvec): {rvec.flatten()}")
                print(f"----------------------\n")

        # 显示结果图像
        cv2.imshow('RealSense ArUco Detection', color_image)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(e)
