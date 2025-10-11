"""
实时D435相机检测程序 - 杯子位姿估计
按 空格键 触发检测和位姿估计
按 'q' 退出程序
按 'v' 可视化位姿（需要Open3D）
"""

import cv2
import os
import time
import numpy as np
from datetime import datetime
from lib.camera import Camera
from lib.yolo_and_sam import YOLOSegmentator
from lib.mask2pose import mask2pose, visualize_result


def main():
    # --- 配置 ---
    # 你想要检测的物体类别（可以检测多个）
    categories_to_find = ["cup", "spoon"]
    
    # 临时图片保存目录
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 相机到基坐标系的变换（如果有的话）
    # 如果没有，设为None，将在相机坐标系中计算位姿
    T_cam2base = None  # 可以设置为你的实际变换矩阵
    
    print("=" * 60)
    print("实时D435相机检测程序 - 多物体位姿估计")
    print("=" * 60)
    print("检测类别: " + ", ".join(categories_to_find))
    print("策略: 每个类别只检测置信度最高的物体")
    print("=" * 60)
    print("控制说明:")
    print("  空格键 - 触发检测和位姿估计")
    print("  'v' 键 - 3D可视化最后一次检测结果")
    print("  'q' 键 - 退出程序")
    print("=" * 60)
    
    # 初始化相机
    try:
        cam = Camera(camera_model='D435')
    except Exception as e:
        print(f"相机初始化失败: {e}")
        print("请确保D435相机已正确连接")
        return
    
    # 初始化检测器（按空格后才加载，避免启动太慢）
    segmentator = None
    
    # 保存最后一次检测的数据，用于可视化
    last_detection_data = None
    
    try:
        print("\n开始实时预览...")
        
        while True:
            # 获取相机画面
            color_image, depth_image = cam.get_frames()
            
            if color_image is None:
                print("无法获取相机画面")
                break
            
            # 转换深度图单位：从毫米转换为米
            if depth_image is not None:
                depth_image_meters = depth_image.astype(float) * cam.depth_scale
            else:
                depth_image_meters = None
            
            # 在画面上显示提示信息
            display_image = color_image.copy()
            cv2.putText(display_image, "SPACE: Detect | V: Visualize | Q: Quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示画面
            cv2.imshow("D435 Camera - Real-time", display_image)
            
            # 检测按键
            key = cv2.waitKey(1) & 0xFF
            
            # 按 'q' 退出
            if key == ord('q'):
                print("\n退出程序...")
                break
            
            # 按 'v' 可视化
            elif key == ord('v'):
                if last_detection_data is not None:
                    print("\n正在打开3D可视化窗口...")
                    # 检查是单个位姿还是多个位姿
                    if 'pose' in last_detection_data:
                        # 单个物体（向后兼容）
                        visualize_result(
                            last_detection_data['color'],
                            last_detection_data['depth'],
                            T_cam2base,
                            cam.get_camera_matrix(),
                            last_detection_data['pose']
                        )
                    elif 'poses' in last_detection_data:
                        # 多个物体
                        from lib.mask2pose import visualize_multi_objects
                        visualize_multi_objects(
                            last_detection_data['color'],
                            last_detection_data['depth'],
                            T_cam2base,
                            cam.get_camera_matrix(),
                            last_detection_data['poses']
                        )
                else:
                    print("\n⚠️ 尚未进行检测，请先按空格键进行检测")
            
            # 按空格键触发检测
            elif key == ord(' '):
                print("\n" + "=" * 60)
                print("触发多物体检测...")
                
                # 延迟加载检测器（节省启动时间）
                if segmentator is None:
                    print("首次检测，正在加载模型...")
                    segmentator = YOLOSegmentator()
                
                # 保存当前帧到临时文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_image_path = os.path.join(temp_dir, f"frame_{timestamp}.jpg")
                cv2.imwrite(temp_image_path, color_image)
                
                # 执行检测和分割（所有物体）
                start_time = time.time()
                result = segmentator.detect_and_segment_all(
                    image=temp_image_path,
                    categories=categories_to_find
                )
                end_time = time.time()
                
                # 显示结果
                if result['success']:
                    num_objects = len(result['objects'])
                    print(f"\n✓ 检测成功! 共检测到 {num_objects} 个物体 (耗时: {end_time - start_time:.2f}s)")
                    print(f"  检测结果: {result['detection_path']}")
                    
                    # 显示检测结果图像
                    det_img = cv2.imread(result['detection_path'])
                    if det_img is not None:
                        cv2.imshow("Detection Result", det_img)
                    
                    # 为每个物体计算位姿
                    if depth_image_meters is not None:
                        all_poses = []
                        
                        print("\n" + "=" * 60)
                        print("开始位姿估计...")
                        print("=" * 60)
                        
                        for idx, obj in enumerate(result['objects']):
                            print(f"\n🔍 物体 {idx+1}/{num_objects}: {obj['class']}")
                            print(f"   置信度: {obj['confidence']:.2f}")
                            print(f"   边界框: {obj['bbox_xyxy']}")
                            
                            # 如果有掩码，计算位姿
                            if obj['mask'] is not None:
                                pose_start = time.time()
                                pose, T = mask2pose(
                                    mask=obj['mask'],
                                    depth_image=depth_image_meters,
                                    color_image=color_image,
                                    intrinsics=cam.get_camera_matrix(),
                                    T_cam2base=T_cam2base,
                                    object_class=obj['class']  # 传入物体类别
                                )
                                pose_end = time.time()
                                
                                if pose is not None:
                                    print(f"   位姿估计耗时: {pose_end - pose_start:.2f}s")
                                    print(f"   📍 位置 (x, y, z): [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
                                    print(f"   📐 姿态 (roll, pitch, yaw): [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
                                    
                                    # 检查是否有勺子的额外信息
                                    extra_info = None
                                    if isinstance(pose, list) and len(pose) > 6 and isinstance(pose[6], dict):
                                        extra_info = pose[6]
                                        if 'spoon_head_center' in extra_info:
                                            head_center = extra_info['spoon_head_center']
                                            head_radius = extra_info['spoon_head_radius']
                                            handle_pose = extra_info['handle_pose']
                                            print(f"   🥄 勺头中心位置: [{head_center[0]:.3f}, {head_center[1]:.3f}, {head_center[2]:.3f}] 米")
                                            print(f"   🥄 勺头半径: {head_radius:.3f}m ({head_radius*100:.1f}cm)")
                                            print(f"   🥄 勺柄姿态: [roll={handle_pose[0]:.1f}°, pitch={handle_pose[1]:.1f}°, yaw={handle_pose[2]:.1f}°]")
                                    
                                    all_poses.append({
                                        'class': obj['class'],
                                        'pose': pose,
                                        'confidence': obj['confidence'],
                                        'extra_info': extra_info
                                    })
                                else:
                                    print(f"   ❌ 位姿估计失败")
                            else:
                                print(f"   ⚠️ 未获取到掩码")
                        
                        # 保存最后一次检测数据（用于可视化）
                        if len(all_poses) > 0:
                            last_detection_data = {
                                'color': color_image.copy(),
                                'depth': depth_image_meters.copy(),
                                'poses': all_poses  # 保存所有位姿
                            }
                            
                            print("\n" + "=" * 60)
                            print(f"✅ 成功估计了 {len(all_poses)} 个物体的位姿")
                            print("=" * 60)
                            
                            # 打印汇总信息
                            print("\n📊 位姿汇总:")
                            for i, pose_info in enumerate(all_poses):
                                pose = pose_info['pose']
                                print(f"\n  {i+1}. {pose_info['class']} (置信度: {pose_info['confidence']:.2f})")
                                print(f"     位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] 米")
                                print(f"     姿态: [{pose[3]:.1f}°, {pose[4]:.1f}°, {pose[5]:.1f}°]")
                                
                                # 如果有额外信息（如勺头中心和勺柄姿态），也显示
                                if 'extra_info' in pose_info and pose_info['extra_info']:
                                    extra = pose_info['extra_info']
                                    if 'spoon_head_center' in extra:
                                        head_center = extra['spoon_head_center']
                                        head_radius = extra['spoon_head_radius']
                                        handle_pose = extra['handle_pose']
                                        print(f"     勺头中心位置: [{head_center[0]:.3f}, {head_center[1]:.3f}, {head_center[2]:.3f}] 米")
                                        print(f"     勺头半径: {head_radius*100:.1f}cm")
                                        print(f"     勺柄姿态: [roll={handle_pose[0]:.1f}°, pitch={handle_pose[1]:.1f}°, yaw={handle_pose[2]:.1f}°]")
                            
                            print("\n💡 提示: 按 'v' 键可进行3D可视化")
                        else:
                            print("\n⚠️ 没有成功估计任何物体的位姿")
                    else:
                        print("\n⚠️ 未获取到深度图，无法进行位姿估计")
                    
                else:
                    print(f"\n✗ 检测失败")
                
                # 删除临时文件
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                
                print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\n用户中断程序...")
    
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        cam.release()
        cv2.destroyAllWindows()
        
        # 清理临时目录
        if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) == 0:
            os.rmdir(temp_dir)
        
        print("\n资源已释放，程序结束")


if __name__ == "__main__":
    main()
