"""
电子秤检测和位姿估计函数
功能：传入RGB图和深度图，识别图片检测电子秤并分割，最后返回位姿
包含手眼标定矩阵
"""

import cv2
import numpy as np
import os
from lib.camera import Camera
from lib.mask2pose import mask2pose, draw_pose_axes
from ultralytics import YOLO


def calculate_scales_pose_simple(bbox, depth_image, intrinsics):
    """
    简化的电子秤位姿估计：使用检测框中心点和周围10个采样点的平均深度
    
    参数:
        bbox: 检测框 [x1, y1, x2, y2]
        depth_image: 深度图像 (H, W)，单位为米
        intrinsics: 相机内参矩阵 3x3
    
    返回:
        pose: [x, y, z, roll, pitch, yaw] 位姿，失败时返回None
        T: 4x4变换矩阵，失败时返回None
    """
    try:
        # 1. 计算检测框中心点
        x1, y1, x2, y2 = bbox
        # center_x = int((x1 + x2) / 2)
        center_x = int(x1 + (x2 - x1) * 0.7)
        # center_y = int((y1 + y2) / 2)
        center_y = int(y1 + (y2 - y1) * 0.8)
        
        # 确保坐标在图像范围内
        h, w = depth_image.shape[:2]
        center_x = max(0, min(w-1, center_x))
        center_y = max(0, min(h-1, center_y))
        
        # 2. 在中心点周围生成采样点（包括中心点共11个点）
        sample_radius = 20  # 采样半径
        sample_points = []
        
        # 中心点
        sample_points.append((center_x, center_y))
        
        # 周围10个点：在中心点周围均匀分布
        angles = np.linspace(0, 2*np.pi, 300, endpoint=False)
        for angle in angles:
            dx = int(sample_radius * np.cos(angle))
            dy = int(sample_radius * np.sin(angle))
            
            sample_x = center_x + dx
            sample_y = center_y + dy
            
            # 确保采样点在图像范围内
            sample_x = max(0, min(w-1, sample_x))
            sample_y = max(0, min(h-1, sample_y))
            
            sample_points.append((sample_x, sample_y))
        
        # 3. 收集所有有效深度值
        valid_depths = []
        for px, py in sample_points:
            d = depth_image[py, px]
            # 过滤无效深度值,以及大于0.6米的深度值
            if 0 < d < 0.8:
                # print(f"  采样点 ({px}, {py}) 深度: {d:.3f} 米")
                valid_depths.append(d)
        
        if len(valid_depths) == 0:
            print("  警告：没有找到有效的深度值")
            return None, None
        
        # 4. 计算平均深度
        avg_depth = np.mean(valid_depths)
        # print(f"  采样点数: {len(valid_depths)}")
        # print(f"  平均深度: {avg_depth:.3f} 米")
        
        # 5. 将2D中心转换为3D坐标
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_3d = (center_x - cx) * avg_depth / fx
        y_3d = (center_y - cy) * avg_depth / fy
        z_3d = avg_depth
        
        # 6. 电子秤姿态设为水平（roll=0, pitch=0, yaw=0）
        # 这适用于电子秤平放在桌面上的情况
        roll, pitch, yaw = 0.0, 0.0, 0.0
        
        pose = [x_3d, y_3d, z_3d, roll, pitch, yaw]
        
        # 7. 构建变换矩阵
        from scipy.spatial.transform import Rotation as R
        rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
        
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = [x_3d, y_3d, z_3d]
        print(f"  位姿估计成功: 位置 [{x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}] 米, 姿态 [{roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°]")
        return pose, T
        
    except Exception as e:
        print(f"  位姿估计出错: {e}")
        return None, None


def detect_scales_pose(cam, conf_threshold=0.1, robot_matrix=None):
    """
    检测电子秤并返回位姿
    
    参数:
        cam: 已初始化的相机对象
        conf_threshold: 检测置信度阈值，默认0.1
        robot_matrix: 机械臂位姿矩阵，可选
    
    返回:
        dict: {
            'success': bool,  # 是否成功检测到电子秤
            'pose': [x, y, z, roll, pitch, yaw] or None,  # 电子秤位姿 (毫米, 度)
            'pose_matrix': np.ndarray or None,  # 4x4位姿变换矩阵
            'detection_info': dict or None,  # 检测信息
            'error_message': str or None  # 错误信息
        }
    """
    try:
        color_image, depth_image = cam.get_frames()

        color_filename = f'capture/color2.png'
        depth_filename = f'capture/depth2.png'

        cv2.imwrite(color_filename, color_image)
        cv2.imwrite(depth_filename, depth_image)
        
        # 3. 处理深度图像格式
        if len(depth_image.shape) == 3:
            # print("深度图是3通道，取第一个通道")
            depth_image = depth_image[:, :, 0]
        
        # 转换为浮点数并使用正确的深度比例转换为米
        # RealSense D405的深度比例是0.0001
        depth_scale = 0.0001
        depth_image = depth_image.astype(np.float32) * depth_scale
        
        # 4. 初始化YOLO模型
        model_path = 'weights/all.pt'  # 使用detect2.py中的模型
        if not os.path.exists(model_path):
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': None,
                'error_message': f"模型文件不存在: {model_path}"
            }
        
        # print(f"\n🔍 开始检测电子秤...")
        # print(f"  图像尺寸: {color_image.shape}")
        # print(f"  模型路径: {model_path}")
        # print(f"  置信度阈值: {conf_threshold}")
        
        # 5. 加载模型并进行检测
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
        
        # 6. 检查检测结果
        if len(results) == 0 or results[0].masks is None:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': None,
                'error_message': "没有检测到任何物体或没有分割掩码"
            }
        
        result = results[0]
        num_detections = len(result.masks)
        
        if num_detections == 0:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': None,
                'error_message': "没有检测到任何物体"
            }
        
        # 7. 查找scales类别的检测结果
        scales_detected = False
        scales_obj = None
        
        for i in range(num_detections):
            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]
            
            # 只处理scales类别
            if class_name.lower() == 'scales':
                mask = result.masks.data[i].cpu().numpy()  # 获取掩码
                confidence = float(result.boxes.conf[i])
                bbox = result.boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                
                # print(f"\n⚖️ 检测到电子秤:")
                # print(f"  类别: {class_name} (ID: {class_id})")
                # print(f"  置信度: {confidence:.2f}")
                # print(f"  边界框: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                # print(f"  掩码尺寸: {mask.shape}")
                
                scales_obj = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox_xyxy': bbox,
                    'mask': mask
                }
                scales_detected = True
                break  # 找到第一个scales就停止
        
        # 8. 检查是否找到scales
        if not scales_detected:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': None,
                'error_message': "没有检测到scales类别"
            }
        
        # 9. 获取相机内参
        intrinsics = cam.get_camera_matrix()
        if intrinsics is None:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': scales_obj,
                'error_message': "无法获取相机内参"
            }
        
        # 10. 手眼标定矩阵（从right_eye_calibration.txt读取）
        calibration_file = 'right_calibration.txt'
        if not os.path.exists(calibration_file):
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': scales_obj,
                'error_message': f"手眼标定文件不存在: {calibration_file}"
            }
        
        # 读取手眼标定矩阵
        try:
            # 从文本文件读取矩阵
            with open(calibration_file, 'r') as f:
                lines = f.readlines()
            
            # 解析每行数据
            matrix_data = []
            for line in lines:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    # 移除行首行尾的方括号
                    row_content = line[1:-1]
                    # 清理可能残留的方括号
                    row_content = row_content.replace('[', '').replace(']', '')
                    # 分割数值并转换为浮点数
                    values = [float(x.strip()) for x in row_content.split()]
                    matrix_data.append(values)
            
            hand_eye_matrix = np.array(matrix_data)
            # print(f"  标定矩阵形状: {hand_eye_matrix.shape}")
            
            
            # print(f"  标定矩阵加载成功")
            # print(f"  标定矩阵内容:\n{hand_eye_matrix}")
            
        except Exception as e:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': scales_obj,
                'error_message': f"读取手眼标定矩阵失败: {str(e)}"
            }
        
        # 11. 简化位姿估计 - 使用检测框中心点和周围采样点
        # print(f"\n📍 开始位姿估计...")
        pose, T_object2cam = calculate_scales_pose_simple(
            bbox=scales_obj['bbox_xyxy'],
            depth_image=depth_image,
            intrinsics=intrinsics
        )
        
        if pose is None or T_object2cam is None:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': scales_obj,
                'error_message': "位姿估计失败"
            }
        
        # print(f"  相机坐标系下位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
        # print(f"  相机坐标系下姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
        
        # 12. 转换到机械臂基坐标系
        # 使用传入的机械臂位姿矩阵，如果没有则使用单位矩阵
        if robot_matrix is not None:
            robot_pose_matrix = robot_matrix
        else:
            robot_pose_matrix = np.eye(4)
        pose_matrix = robot_pose_matrix @ hand_eye_matrix @ T_object2cam
        
        # print(f"  机械臂坐标系下位姿矩阵: {pose_matrix}")
        
        # 14. 可视化位姿（可选）
        # try:
        #     draw_pose_axes(color_image, intrinsics, T_object2cam)
        # except Exception as e:
        #     print(f"位姿可视化失败: {e}")
        
        # 15. 返回结果
        return pose_matrix
        
    except Exception as e:
        return {
            'success': False,
            'pose': None,
            'pose_matrix': None,
            'detection_info': None,
            'error_message': f"函数执行出错: {str(e)}"
        }

def main():
    """
    主函数 - 初始化相机并运行检测
    """
    print("🚀 电子秤检测和位姿估计程序")
    print("=" * 50)
    
    # 初始化相机
    print("📷 初始化相机...")
    try:
        cam = Camera(camera_model='D405')
        print("✅ 相机初始化成功")
    except Exception as e:
        print(f"❌ 相机初始化失败: {e}")
        return
    
    try:
        # 调用检测函数
        result = detect_scales_pose(cam, conf_threshold=0.7, robot_matrix=None)
        
        
        print(result)
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        return None
    finally:
        # 释放相机资源
        print("\n🔧 释放相机资源...")
        cam.release()
        print("✅ 程序结束")



if __name__ == "__main__":
    # 运行主程序
    main()
