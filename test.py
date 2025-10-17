import cv2
import os
import time
import numpy as np
from datetime import datetime
from lib.camera import Camera
from lib.yolo_and_sam import YOLOSegmentator
from lib.mask2pose import mask2pose, visualize_result
from lib.dobot import DobotRobot
from cv2 import aruco
from pathlib import Path
from ultralytics import YOLO

def detect_aruco_pose(camera,image):
    """
    检测ArUco板子的位姿
    
    参数:
        image: 输入图像
        
    返回:
        success: 是否检测成功
        rvec: 旋转向量
        tvec: 平移向量
        corners: 检测到的角点
        ids: 检测到的ID
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    aruco_params = cv2.aruco.DetectorParameters()
    # ArUco板子参数 (8x8cm)
    marker_size = 0.08  # 8cm = 0.08m
    board_size = (4, 4)  # 4x4的ArUco板子
    
    camera_matrix = camera.get_camera_matrix('color')
    dist_coeffs = camera.get_distortion_coeffs('color')
    
    # 检测ArUco标记
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is None or len(ids) < 1:  # 至少需要1个标记
        return False, None, None, None, None
    
    # 对于单个标记，使用estimatePoseSingleMarkers
    if len(ids) == 1:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs
        )
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]
        success = True


    
    aruco_to_cam = np.eye(4)
    aruco_to_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
    aruco_to_cam[:3, 3] = tvec.flatten()
    
    return aruco_to_cam


def yolo_and_sam_segment(color_image):
    categories_to_find = ['spoon']
    segmentator = YOLOSegmentator()
    
    # 使用调整后的图像进行检测
    result = segmentator.detect_and_segment_all(
        image=color_image,
        categories=categories_to_find,
        save_result=True
    )
    return result

def my_yolo(color_image):
    # 检查文件是否存在
    weights_path = './weights/best.pt'
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        print(f"❌ 错误: 权重文件不存在: {weights_path}")
        return
    
    model = YOLO(str(weights_path))

    
    # 使用调整后的图像进行检测
    result = model.predict(
        source=color_image,
        save=True,              # 保存可视化结果
        save_txt=True,          # 保存标签文件
        save_conf=True,         # 保存置信度
        conf=0.25,              # 置信度阈值
        iou=0.7,                # NMS的IoU阈值
        exist_ok=True,          # 允许覆盖
        show_labels=True,       # 显示标签
        show_conf=True,         # 显示置信度
        line_width=2,           # 边界框线宽
    )
    return result


def main():
    robot1 = DobotRobot(robot_ip='192.168.5.1')
    robot2 = DobotRobot(robot_ip='192.168.5.2')

    print("机器人1状态:", robot1.get_XYZrxryrz_state())
    print("机器人2状态:", robot2.get_XYZrxryrz_state())

    return 0
    cam = Camera(camera_model='D405')  # 初始化相机
    robot = DobotRobot(robot_ip='192.168.5.')  # 初始化机械臂
    intrinsics = cam.get_camera_matrix('color')
    robot.r_inter.StartDrag()
    # return 0
    # print(robot.get_pose_matrix())

    # # 读取原始图像
    # color_image_path = 'test/color.png'
    # depth_image_path = 'test/depth.png'
    # color_image = cv2.imread(color_image_path)
    # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    # 创建结果保存目录
    output_dir = 'yolo_results_realtime'
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 实时YOLO检测启动")
    print("📹 结果将保存到:", output_dir)
    print("⏹️  按 Ctrl+C 停止\n")
    
    frame_count = 0
    
    try:
        while True:
            color_image, depth_image = cam.get_frames()
            
            if color_image is None:
                print("⚠️ 未获取到图像")
                time.sleep(0.1)
                continue

            frame_count += 1
            
            # YOLO检测
            result = my_yolo(color_image)
            
            # 保存结果图像（YOLO已经保存到runs/segment/predict/）
            result_image_path = './runs/segment/predict/image0.jpg'
            
            if os.path.exists(result_image_path):
                result_image = cv2.imread(result_image_path)
                
                # 保存到自定义目录（带时间戳）
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                save_path = os.path.join(output_dir, f'frame_{frame_count:04d}_{timestamp}.jpg')
                cv2.imwrite(save_path, result_image)
                
                # 尝试显示（如果支持GUI）
                try:
                    cv2.imshow("YOLO Detection", result_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC退出
                        print("\n👋 用户按ESC退出")
                        break
                except cv2.error:
                    # 如果不支持GUI，只打印信息
                    if frame_count == 1:
                        print("⚠️ OpenCV GUI不可用，结果将只保存到文件")
                    
                    # 提取检测信息并打印
                    if result and len(result) > 0:
                        boxes = getattr(result[0], 'boxes', None)
                        if boxes is not None and len(boxes) > 0:
                            names = getattr(result[0], 'names', {})
                            cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)
                            conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
                            
                            print(f"帧 {frame_count}: 检测到 {len(boxes)} 个物体")
                            for i in range(len(cls_arr)):
                                class_id = int(cls_arr[i])
                                class_name = names.get(class_id, f"class_{class_id}")
                                confidence = float(conf_arr[i])
                                print(f"  - {class_name}: {confidence:.2%}")
                        else:
                            print(f"帧 {frame_count}: 未检测到物体")
                    
                    # 每10帧打印一次保存信息
                    if frame_count % 10 == 0:
                        print(f"✅ 已保存 {frame_count} 帧到 {output_dir}")
                    
                    # 控制帧率（避免太快）
                    time.sleep(0.1)
            else:
                print(f"⚠️ 未找到结果图像: {result_image_path}")
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  检测已停止")
        print(f"📊 总共处理 {frame_count} 帧")
        print(f"💾 结果保存在: {output_dir}")
    
    cv2.destroyAllWindows()
    return 0

    # 确保深度图是二维的
    if len(depth_image.shape) == 3:
        print(f"  深度图是3通道,取第一个通道")
        depth_image = depth_image[:, :, 0]
    
    # 转换为浮点数并使用正确的深度比例转换为米
    # RealSense D405的深度比例是0.0001 (不是0.001!)
    # 所以要除以10000 (不是1000)
    depth_scale = 0.0001  # 从相机信息中获取
    depth_image = depth_image.astype(np.float32) * depth_scale
    
    
    T_cam2base = np.eye(4)  # 假设相机位姿已知


    result = my_yolo(color_image)
    # result['objects']是一个字典的list，每个字典代表一个物体包含类别、置信度、边界框和掩码
    visualize_yolo_result(color_image, result, wait_key=True)

    return 0


    intrinsics = cam.get_camera_matrix()
    
    for idx, obj in enumerate(result['objects']):
        print(f"\n物体 {idx + 1}:")
        print(f"  类别: {obj['class']}")
        print(f"  置信度: {obj['confidence']:.2f}")
        print(f"  边界框: {obj['bbox_xyxy']}")
        
        # 如果有掩码，计算位姿
        if obj['mask'] is not None:
            print(f"  掩码尺寸: {obj['mask'].shape}")
            print(f"  彩色图像尺寸: {color_image.shape}")
            print(f"  深度图像尺寸: {depth_image.shape}")
            
            pose, T_object2cam = mask2pose(
                mask=obj['mask'],
                depth_image=depth_image,
                color_image=color_image,
                intrinsics=intrinsics,
                T_cam2base=T_cam2base,
                object_class=obj['class']
            )
            
            if pose is not None:
                print(f"   📍 相机坐标系下位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
                print(f"   📐 相机坐标系下姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
                break  # 只处理第一个检测到的物体
            else:
                print(f"   ❌ 位姿估计失败")
        else:
            print(f"   ⚠️ 未获取到掩码")
    
    
    

    calibration_file = 'best_hand_eye_calibration.npy'
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"手眼标定文件不存在: {calibration_file}")
    
    hand_eye_matrix = np.load(calibration_file)
            
    # aruco_to_cam = detect_aruco_pose(cam,color_image)
    
    target_to_cam = T_object2cam
    
    # pose_matrix =  robot.get_pose_matrix() @ hand_eye_matrix @ target_to_cam
    
    # robot.moveL(pose_matrix)
    # print("目标物体相对于机械臂基座的位姿矩阵:\n", pose_matrix)

    

    # draw_pose_axes(color_image, intrinsics, T_object2cam)

    # print(f"ArUco板相对于机械臂基座位置 (mm): X={position[0]*1000:.2f}, Y={position[1]*1000:.2f}, Z={position[2]*1000:.2f}\n")
    # print(f"ArUco板相对于机械臂基座变换矩阵:\n{pose_matrix}")



if __name__ == "__main__":
    main()