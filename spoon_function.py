import cv2
import numpy as np
import os
from lib.camera import Camera
from lib.mask2pose import mask2pose, draw_pose_axes
from ultralytics import YOLO
from lib.dobot import DobotRobot

def detect_spoon_pose(cam, conf_threshold=0.6, robot_matrix=None):
    color_image, depth_image = cam.get_frames()

    color_filename = f'spoon/color.png'
    depth_filename = f'spoon/depth.png'

    cv2.imwrite(color_filename, color_image)
    cv2.imwrite(depth_filename, depth_image)
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]
    # RealSense D405的深度比例是0.0001
    depth_scale = 0.0001
    depth_image = depth_image.astype(np.float32) * depth_scale
    model_path = 'weights/best.pt'  # 使用detect2.py中的模型
    # 1. 目标检测
    model = YOLO(model_path)
    results = model.predict(
        source=color_image,
        save=False,
        conf=conf_threshold,
        iou=0.7,
        show_labels=False,
        show_conf=False,
        verbose=False
    )

    result = results[0]
    num_detections = len(result.masks) 
    spoon_detected = False
    spoon_obj = None

    for i in range(num_detections):
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]
        
        # 只处理spoon类别
        if class_name.lower() == 'spoon':
            mask = result.masks.data[i].cpu().numpy()  # 获取掩码
            confidence = float(result.boxes.conf[i])
            bbox = result.boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            
            spoon_obj = {
                'class': class_name,
                'confidence': confidence,
                'bbox_xyxy': bbox,
                'mask': mask
            }
            spoon_detected = True
            break  # 找到第一个spoon就停止

    intrinsics = cam.get_camera_matrix()

    # 2. 使用mask2pose进行位姿估计
    pose, T_object2cam = mask2pose(
        mask=spoon_obj['mask'],
        depth_image=depth_image,
        color_image=color_image,
        intrinsics=intrinsics,
        T_cam2base=None,
        object_class='spoon'
    )
    # 3. 转换到机械臂基坐标系
    hand_eye_matrix = np.array([
            [ 0.01230037,  0.99763761,  0.06758625,  0.08419052],
            [-0.99992251,  0.01240196, -0.00108365,  0.00995925],
            [-0.00191929, -0.06756769,  0.99771285, -0.15882536],
            [ 0.0,         0.0,         0.0,         1.0        ]
        ])

    if robot_matrix is not None:
        robot_pose_matrix = robot_matrix
    else:
        robot_pose_matrix = np.eye(4)
    # 勺子中心的位姿矩阵
    pose_matrix = robot_pose_matrix @ hand_eye_matrix @ T_object2cam

    # 提取勺子额外信息
    spoon_info = {}
    if len(pose) > 6:  # 如果有额外信息（勺头中心等）
        spoon_info = pose[6]  # 第7个元素是额外信息
        pose = pose[:6]  # 只保留前6个元素作为基本位姿

    # 1. 勺子整体中心的位姿矩阵（用于抓取勺子的几何中心）
    spoon_center_pose_matrix = pose_matrix

    # 2. 勺柄中心的位姿矩阵（用于抓取勺柄）
    # 使用与勺子整体中心相同的旋转矩阵，只改变位置
    spoon_handle_center = spoon_info['spoon_handle_center']
    spoon_handle_center_matrix = T_object2cam.copy()  # 复制整体位姿矩阵
    spoon_handle_center_matrix[:3, 3] = spoon_handle_center  # 只更新位置
    spoon_handle_center_pose_matrix = robot_pose_matrix @ hand_eye_matrix @ spoon_handle_center_matrix

    # 3. 勺头中心的位姿矩阵（用于抓取勺头）
    # 使用与勺子整体中心相同的旋转矩阵，只改变位置
    spoon_head_center = spoon_info['spoon_head_center']
    spoon_head_center_matrix = T_object2cam.copy()  # 复制整体位姿矩阵
    spoon_head_center_matrix[:3, 3] = spoon_head_center  # 只更新位置
    spoon_head_center_pose_matrix = robot_pose_matrix @ hand_eye_matrix @ spoon_head_center_matrix

    # 可视化位姿、勺柄中心和勺头中心
    try:
        draw_pose_axes(color_image, intrinsics, T_object2cam)
        
        visualization_added = False
        
        # 添加勺柄中心的橙色点标记
        if spoon_info and 'spoon_handle_center' in spoon_info:
            handle_center_cam = spoon_info['spoon_handle_center']  # 相机坐标系下的勺柄中心
            
            # 将3D点投影到2D图像坐标
            handle_center_3d = np.array([handle_center_cam[0], handle_center_cam[1], handle_center_cam[2]])
            handle_center_2d, _ = cv2.projectPoints(
                handle_center_3d.reshape(-1, 1, 3),
                np.zeros(3), np.zeros(3),
                intrinsics[:3, :3], np.zeros(4)
            )
            handle_center_2d = handle_center_2d[0, 0].astype(int)
            
            # 在图像上绘制橙色圆点标记勺柄中心
            cv2.circle(color_image, tuple(handle_center_2d), 8, (0, 165, 255), -1)  # 橙色实心圆
            cv2.circle(color_image, tuple(handle_center_2d), 12, (0, 100, 200), 2)  # 橙色边框
            
            # 添加文字标签
            cv2.putText(color_image, 'Handle', 
                       (handle_center_2d[0] + 15, handle_center_2d[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            print(f"  勺柄中心2D坐标: [{handle_center_2d[0]}, {handle_center_2d[1]}]")
            visualization_added = True
        
        # 添加勺头中心的蓝色点标记
        if spoon_info and 'spoon_head_center' in spoon_info:
            head_center_cam = spoon_info['spoon_head_center']  # 相机坐标系下的勺头中心
            
            # 将3D点投影到2D图像坐标
            head_center_3d = np.array([head_center_cam[0], head_center_cam[1], head_center_cam[2]])
            head_center_2d, _ = cv2.projectPoints(
                head_center_3d.reshape(-1, 1, 3),
                np.zeros(3), np.zeros(3),
                intrinsics[:3, :3], np.zeros(4)
            )
            head_center_2d = head_center_2d[0, 0].astype(int)
            
            # 在图像上绘制蓝色圆点标记勺头中心
            cv2.circle(color_image, tuple(head_center_2d), 8, (255, 100, 0), -1)  # 蓝色实心圆
            cv2.circle(color_image, tuple(head_center_2d), 12, (200, 50, 0), 2)   # 蓝色边框
            
            # 添加文字标签
            cv2.putText(color_image, 'Head', 
                       (head_center_2d[0] + 15, head_center_2d[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
            
            print(f"  勺头中心2D坐标: [{head_center_2d[0]}, {head_center_2d[1]}]")
            visualization_added = True
        
        # 如果添加了可视化，保存图像
        if visualization_added:
            result_filename = f'spoon/spoon_with_both_centers.png'
            cv2.imwrite(result_filename, color_image)
            print(f"  可视化结果已保存: {result_filename}")
    except Exception as e:
        print(f"位姿可视化失败: {e}")

    return {
        'spoon_center_pose_matrix': spoon_center_pose_matrix,      # 勺子整体中心位姿矩阵
        'spoon_handle_pose_matrix': spoon_handle_center_pose_matrix,  # 勺柄中心位姿矩阵
        'spoon_head_pose_matrix': spoon_head_center_pose_matrix,    # 勺头中心位姿矩阵
    }

def main():
    """
    主函数 - 初始化相机并运行检测
    """
    print("🚀 勺子检测和位姿估计程序")

    cam = Camera(camera_model='D405')

    robot=DobotRobot(robot_ip='192.168.5.2',no_gripper=True)
    robot_matrix = robot.get_pose_matrix()
    result = detect_spoon_pose(cam, conf_threshold=0.5, robot_matrix=robot_matrix)
    print(result)
    cam.release()


if __name__ == "__main__":
    # 运行主程序
    main()