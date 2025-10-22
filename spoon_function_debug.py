"""
勺子检测和位姿估计函数
功能：传入RGB图和深度图，识别图片检测勺子并分割，最后返回位姿
包含手眼标定矩阵
"""

import cv2
import numpy as np
import os
from lib.camera import Camera
from lib.mask2pose import mask2pose, draw_pose_axes
from ultralytics import YOLO


def detect_spoon_pose(cam, conf_threshold=0.1, robot_matrix=None):
    """
    检测勺子并返回位姿
    
    参数:
        cam: 已初始化的相机对象
        conf_threshold: 检测置信度阈值，默认0.1
        robot_matrix: 机械臂位姿矩阵，可选
    
    返回:
        dict: {
            'success': bool,  # 是否成功检测到勺子
            'pose': [x, y, z, roll, pitch, yaw] or None,  # 勺子位姿 (米, 度)
            'pose_matrix': np.ndarray or None,  # 4x4位姿变换矩阵
            'detection_info': dict or None,  # 检测信息
            'spoon_info': dict or None,  # 勺子额外信息（勺头中心、半径等）
            'error_message': str or None  # 错误信息
        }
    """
    try:
        color_image, depth_image = cam.get_frames()

        color_filename = f'spoon/color.png'
        depth_filename = f'spoon/depth.png'

        cv2.imwrite(color_filename, color_image)
        cv2.imwrite(depth_filename, depth_image)
        
        # 3. 处理深度图像格式
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        
        # 转换为浮点数并使用正确的深度比例转换为米
        # RealSense D405的深度比例是0.0001
        depth_scale = 0.0001
        depth_image = depth_image.astype(np.float32) * depth_scale
        
        # 4. 初始化YOLO模型
        model_path = 'weights/best.pt'  # 使用detect2.py中的模型
        if not os.path.exists(model_path):
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': None,
                'spoon_info': None,
                'error_message': f"模型文件不存在: {model_path}"
            }
        
        print(f"\n🥄 开始检测勺子...")
        print(f"  图像尺寸: {color_image.shape}")
        print(f"  模型路径: {model_path}")
        print(f"  置信度阈值: {conf_threshold}")
        
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
                'spoon_info': None,
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
                'spoon_info': None,
                'error_message': "没有检测到任何物体"
            }
        
        # 7. 查找spoon类别的检测结果
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
                
                print(f"\n🥄 检测到勺子:")
                print(f"  类别: {class_name} (ID: {class_id})")
                print(f"  置信度: {confidence:.2f}")
                print(f"  边界框: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                print(f"  掩码尺寸: {mask.shape}")
                
                spoon_obj = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox_xyxy': bbox,
                    'mask': mask
                }
                spoon_detected = True
                break  # 找到第一个spoon就停止
        
        # 8. 检查是否找到spoon
        if not spoon_detected:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': None,
                'spoon_info': None,
                'error_message': "没有检测到spoon类别"
            }
        
        # 9. 获取相机内参
        intrinsics = cam.get_camera_matrix()
        if intrinsics is None:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': spoon_obj,
                'spoon_info': None,
                'error_message': "无法获取相机内参"
            }
        
        hand_eye_matrix = np.array([
            [ 0.01230037,  0.99763761,  0.06758625,  0.08419052],
            [-0.99992251,  0.01240196, -0.00108365,  0.00995925],
            [-0.00191929, -0.06756769,  0.99771285, -0.15882536],
            [ 0.0,         0.0,         0.0,         1.0        ]
        ])

        
        # 11. 使用mask2pose进行位姿估计
        print(f"\n📍 开始勺子位姿估计...")
        pose, T_object2cam = mask2pose(
            mask=spoon_obj['mask'],
            depth_image=depth_image,
            color_image=color_image,
            intrinsics=intrinsics,
            T_cam2base=None,  # 先不转换坐标系
            object_class="spoon"
        )
        
        if pose is None or T_object2cam is None:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': spoon_obj,
                'spoon_info': None,
                'error_message': "位姿估计失败"
            }
        
        # 12. 提取勺子额外信息
        spoon_info = {}
        if len(pose) > 6:  # 如果有额外信息（勺头中心等）
            spoon_info = pose[6]  # 第7个元素是额外信息
            pose = pose[:6]  # 只保留前6个元素作为基本位姿
        
        print(f"  相机坐标系下位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
        print(f"  相机坐标系下姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
        
        if spoon_info:
            if 'spoon_handle_center' in spoon_info:
                handle_center = spoon_info['spoon_handle_center']
                print(f"  勺柄中心: [{handle_center[0]:.3f}, {handle_center[1]:.3f}, {handle_center[2]:.3f}] 米")
            if 'spoon_handle_radius' in spoon_info:
                print(f"  勺柄半径: {spoon_info['spoon_handle_radius']:.3f} 米")
            if 'spoon_head_center' in spoon_info:
                head_center = spoon_info['spoon_head_center']
                print(f"  勺头中心: [{head_center[0]:.3f}, {head_center[1]:.3f}, {head_center[2]:.3f}] 米")
            if 'spoon_head_radius' in spoon_info:
                print(f"  勺头半径: {spoon_info['spoon_head_radius']:.3f} 米")
            
            # 显示姿态信息
            if 'handle_pose' in spoon_info:
                handle_pose = spoon_info['handle_pose']
                print(f"  勺柄姿态: [{handle_pose[0]:.1f}°, {handle_pose[1]:.1f}°, {handle_pose[2]:.1f}°]")
                print(f"  勺子整体姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
                
                # 验证姿态一致性（主轴方向的欧拉角应该一致）
                pose_diff = np.abs(np.array(pose[3:6]) - np.array(handle_pose))
                max_diff = np.max(pose_diff)
                print(f"  姿态差异: 最大差异 {max_diff:.1f}° ({'一致' if max_diff < 1.0 else '不一致'})")
        
        # 13. 转换到机械臂基坐标系
        # 使用传入的机械臂位姿矩阵，如果没有则使用单位矩阵
        if robot_matrix is not None:
            robot_pose_matrix = robot_matrix
        else:
            robot_pose_matrix = np.eye(4)
        pose_matrix = robot_pose_matrix @ hand_eye_matrix @ T_object2cam
        
        print(f"  机械臂坐标系下位姿矩阵计算完成")
        
        # 14. 可视化位姿、勺柄中心和勺头中心（可选）
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
                result_filename = f'spoon/spoon_with_both_centers_{int(cv2.getTickCount())}.png'
                cv2.imwrite(result_filename, color_image)
                print(f"  可视化结果已保存: {result_filename}")
        except Exception as e:
            print(f"位姿可视化失败: {e}")
        
        # 15. 返回结果
        return {
            'success': True,
            'pose': pose,
            'pose_matrix': pose_matrix,
            'detection_info': spoon_obj,
            'spoon_info': spoon_info,
            'error_message': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'pose': None,
            'pose_matrix': None,
            'detection_info': None,
            'spoon_info': None,
            'error_message': f"函数执行出错: {str(e)}"
        }


def main():
    """
    主函数 - 初始化相机并运行检测
    """
    print("🚀 勺子检测和位姿估计程序")
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
        result = detect_spoon_pose(cam, conf_threshold=0.7, robot_matrix=None)
        
        if result['success']:
            print(f"\n✅ 勺子检测成功!")
            print(f"  位姿: {result['pose']}")
            if result['spoon_info']:
                print(f"  勺子信息: {result['spoon_info']}")
        else:
            print(f"\n❌ 勺子检测失败: {result['error_message']}")
        
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
