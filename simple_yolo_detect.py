"""
杯子检测和位姿估计函数
功能：传入RGB图和深度图，识别图片检测杯子并分割，最后返回位姿
包含手眼标定矩阵
"""

import cv2
import numpy as np
import os
from lib.camera import Camera
from lib.yolo_and_sam import YOLOSegmentator
from lib.mask2pose import mask2pose, draw_pose_axes


def detect_cup_pose(rgb_image, depth_image, camera_model='D405', conf_threshold=0.1, robot_matrix=None):
    """
    检测杯子并返回位姿
    
    参数:
        rgb_image: RGB图像 (numpy.ndarray) 或图像路径 (str)
        depth_image: 深度图像 (numpy.ndarray) 或图像路径 (str)
        camera_model: 相机型号，默认'D405'
        conf_threshold: 检测置信度阈值，默认0.1
    
    返回:
        dict: {
            'success': bool,  # 是否成功检测到杯子
            'pose': [x, y, z, roll, pitch, yaw] or None,  # 杯子位姿 (米, 度)
            'pose_matrix': np.ndarray or None,  # 4x4位姿变换矩阵
            'detection_info': dict or None,  # 检测信息
            'error_message': str or None  # 错误信息
        }
    """
    try:
        # 1. 初始化相机对象（用于获取内参）
        cam = Camera(camera_model=camera_model)
        
        # 2. 处理输入图像
        if isinstance(rgb_image, str):
            color_image = cv2.imread(rgb_image)
            if color_image is None:
                return {
                    'success': False,
                    'pose': None,
                    'pose_matrix': None,
                    'detection_info': None,
                    'error_message': f"无法读取RGB图像: {rgb_image}"
                }
        else:
            color_image = rgb_image.copy()
        
        if isinstance(depth_image, str):
            depth_image_array = cv2.imread(depth_image, cv2.IMREAD_UNCHANGED)
            if depth_image_array is None:
                return {
                    'success': False,
                    'pose': None,
                    'pose_matrix': None,
                    'detection_info': None,
                    'error_message': f"无法读取深度图像: {depth_image}"
                }
        else:
            depth_image_array = depth_image.copy()
        
        # 3. 处理深度图像格式
        if len(depth_image_array.shape) == 3:
            print("深度图是3通道，取第一个通道")
            depth_image_array = depth_image_array[:, :, 0]
        
        # 转换为浮点数并使用正确的深度比例转换为米
        # RealSense D405的深度比例是0.0001
        depth_scale = 0.0001
        depth_image_array = depth_image_array.astype(np.float32) * depth_scale
        
        # 4. 初始化检测器
        categories_to_find = ['cup']
        segmentator = YOLOSegmentator()
        
        # 5. 检测和分割杯子
        print(f"\n🔍 开始检测杯子...")
        print(f"  图像尺寸: {color_image.shape}")
        print(f"  检测类别: {categories_to_find}")
        print(f"  置信度阈值: {conf_threshold}")
        
        result = segmentator.detect_and_segment_all(
            image=color_image,
            categories=categories_to_find,
            save_result=False,
            conf=conf_threshold
        )
        
        # 6. 检查检测结果
        if not result['success']:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': None,
                'error_message': "检测失败：没有检测到任何物体"
            }
        
        if 'objects' not in result or len(result['objects']) == 0:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': None,
                'error_message': "没有检测到任何物体"
            }
        
        # 7. 处理第一个检测到的杯子
        cup_obj = result['objects'][0]
        print(f"\n🍵 检测到杯子:")
        print(f"  类别: {cup_obj['class']}")
        print(f"  置信度: {cup_obj['confidence']:.2f}")
        print(f"  边界框: {cup_obj['bbox_xyxy']}")
        
        # 8. 检查是否有掩码
        if cup_obj['mask'] is None:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': cup_obj,
                'error_message': "未获取到杯子的分割掩码"
            }
        
        print(f"  掩码尺寸: {cup_obj['mask'].shape}")
        
        # 9. 获取相机内参
        intrinsics = cam.get_camera_matrix()
        if intrinsics is None:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': cup_obj,
                'error_message': "无法获取相机内参"
            }
        
        # 10. 手眼标定矩阵（从left_calibration.txt读取）
        calibration_file = 'left_calibration.txt'
        if not os.path.exists(calibration_file):
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': cup_obj,
                'error_message': f"手眼标定文件不存在: {calibration_file}"
            }
        
        # 读取手眼标定矩阵
        with open(calibration_file, 'r') as f:
            content = f.read().strip()
        
        # 解析矩阵数据 - 处理嵌套方括号格式
        # 移除最外层的方括号
        if content.startswith('[[') and content.endswith(']]'):
            content = content[1:-1]  # 移除最外层的方括号
        
        # 按行分割
        lines = content.split('\n')
        matrix_data = []
        for line in lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                # 移除方括号并分割
                line = line[1:-1]
                row = [float(x.strip()) for x in line.split()]
                matrix_data.append(row)
        
        if len(matrix_data) != 4:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': cup_obj,
                'error_message': f"手眼标定矩阵格式错误，期望4x4矩阵，实际{len(matrix_data)}x{len(matrix_data[0]) if matrix_data else 0}"
            }
        
        hand_eye_matrix = np.array(matrix_data)
        
        # 11. 估计杯子位姿
        print(f"\n📍 开始位姿估计...")
        pose, T_object2cam = mask2pose(
            mask=cup_obj['mask'],
            depth_image=depth_image_array,
            color_image=color_image,
            intrinsics=intrinsics,
            T_cam2base=None,  # 在相机坐标系中计算
            object_class=cup_obj['class']
        )
        
        if pose is None or T_object2cam is None:
            return {
                'success': False,
                'pose': None,
                'pose_matrix': None,
                'detection_info': cup_obj,
                'error_message': "位姿估计失败"
            }
        
        print(f"  相机坐标系下位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
        print(f"  相机坐标系下姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
        
        # 12. 转换到机械臂基坐标系
        # 假设机械臂当前位姿为单位矩阵（可以根据实际情况调整）
        robot_pose_matrix = np.eye(4)
        robot_pose_matrix = robot_matrix
        pose_matrix = robot_pose_matrix @ hand_eye_matrix @ T_object2cam
        
        # 13. 转换为机械臂坐标系下的位姿
        from scipy.spatial.transform import Rotation as R
        rx, ry, rz = R.from_matrix(pose_matrix[:3, :3]).as_euler('xyz', degrees=True)
        x, y, z = pose_matrix[:3, 3] * 1000.0  # 转换为毫米
        ry = ry + 90  # 根据实际需要调整
        
        final_pose = [x, y, z, rx, ry, rz]
        
        print(f"\n✅ 位姿估计完成:")
        print(f"  机械臂坐标系下位置: [{x:.1f}, {y:.1f}, {z:.1f}] 毫米")
        print(f"  机械臂坐标系下姿态: [{rx:.1f}°, {ry:.1f}°, {rz:.1f}°]")
        
        # 14. 可视化位姿（可选）
        try:
            draw_pose_axes(color_image, intrinsics, T_object2cam)
        except Exception as e:
            print(f"位姿可视化失败: {e}")
        
        # 15. 返回结果
        return {
            'success': True,
            'pose': final_pose,
            'pose_matrix': pose_matrix,
            'detection_info': {
                'class': cup_obj['class'],
                'confidence': cup_obj['confidence'],
                'bbox_xyxy': cup_obj['bbox_xyxy'],
                'mask_shape': cup_obj['mask'].shape,
                'camera_pose': pose,
                'camera_pose_matrix': T_object2cam
            },
            'error_message': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'pose': None,
            'pose_matrix': None,
            'detection_info': None,
            'error_message': f"函数执行出错: {str(e)}"
        }
    finally:
        # 释放相机资源
        try:
            cam.release()
        except:
            pass


def test_with_test_images():
    """
    使用test文件夹中的图片测试函数
    """
    print("🧪 开始测试杯子检测和位姿估计函数...")
    
    # 测试图片路径
    rgb_path = 'right_pic/color.png'
    depth_path = 'right_pic/depth.png'
    
    # 检查文件是否存在
    if not os.path.exists(rgb_path):
        print(f"❌ RGB图像文件不存在: {rgb_path}")
        return
    
    if not os.path.exists(depth_path):
        print(f"❌ 深度图像文件不存在: {depth_path}")
        return
    
    print(f"📁 使用测试图片:")
    print(f"  RGB: {rgb_path}")
    print(f"  深度: {depth_path}")
    
    # 调用检测函数
    result = detect_cup_pose(rgb_path, depth_path)
    
    # 输出结果
    print(f"\n📊 测试结果:")
    print(f"  成功: {result['success']}")
    
    if result['success']:
        print(f"  杯子位姿: {result['pose']}")
        print(f"  检测信息: {result['detection_info']}")
    else:
        print(f"  错误信息: {result['error_message']}")
    
    return result


if __name__ == "__main__":
    # 运行测试
    test_with_test_images()