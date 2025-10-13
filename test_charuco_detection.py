"""
测试 ChArUco 标定板检测
用于验证打印的标定板是否能被正确检测

使用方法:
    python test_charuco_detection.py
"""

import cv2
import numpy as np
from lib.camera import Camera

def test_charuco_detection():
    """测试 ChArUco 标定板检测"""
    
    # 参数配置（与 eye_in_hand2.0_charuco.py 一致）
    squaresX = 14
    squaresY = 9
    squareLength = 0.02  # 20mm
    markerLength = 0.015  # 15mm
    aruco_type = cv2.aruco.DICT_5X5_1000
    
    print("="*60)
    print("ChArUco 标定板检测测试")
    print("="*60)
    print(f"标定板配置:")
    print(f"  - 尺寸: {squaresX} x {squaresY} 方块")
    print(f"  - 方块边长: {squareLength*1000} mm")
    print(f"  - ArUco标记边长: {markerLength*1000} mm")
    print(f"  - 字典类型: DICT_5X5_1000")
    print("="*60)
    print("\n操作说明:")
    print("  - 将标定板放在相机前")
    print("  - 绿色圆点：检测到的 ChArUco 角点")
    print("  - 彩色方框：检测到的 ArUco 标记")
    print("  - 坐标轴：标定板位姿")
    print("  - 按 's' 保存当前图像")
    print("  - 按 'q' 退出")
    print("="*60)
    
    # 初始化相机
    try:
        cam = Camera(camera_model='d405')
        camera_matrix = cam.get_camera_matrix('color')
        distortion_coeffs = cam.get_distortion_coeffs('color')
        print("\n✓ 相机初始化成功")
    except Exception as e:
        print(f"\n✗ 相机初始化失败: {e}")
        return
    
    # 创建 ChArUco 检测器
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
    board = cv2.aruco.CharucoBoard(
        (squaresX, squaresY),
        squareLength,
        markerLength,
        dictionary
    )
    
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    
    print("✓ 检测器初始化成功")
    print("\n开始检测...\n")
    
    saved_count = 0
    total_frames = 0
    detected_frames = 0
    
    try:
        while True:
            # 获取图像
            try:
                color_image, _ = cam.get_frames()
            except Exception as e:
                print(f"获取图像失败: {e}")
                continue
            
            total_frames += 1
            
            # 去畸变
            undistorted_image = cv2.undistort(color_image, camera_matrix, distortion_coeffs)
            
            # 检测 ArUco 标记
            marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(undistorted_image)
            
            # 创建显示图像
            display_image = color_image.copy()
            
            # 检测到标记
            num_markers = 0 if marker_ids is None else len(marker_ids)
            num_charuco_corners = 0
            pose_estimated = False
            
            if marker_ids is not None and len(marker_ids) > 0:
                # 绘制 ArUco 标记
                cv2.aruco.drawDetectedMarkers(display_image, marker_corners, marker_ids)
                
                # 检测 ChArUco 角点
                charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(undistorted_image)
                
                if charuco_corners is not None and len(charuco_corners) > 0:
                    num_charuco_corners = len(charuco_corners)
                    detected_frames += 1
                    
                    # 绘制 ChArUco 角点
                    cv2.aruco.drawDetectedCornersCharuco(
                        display_image,
                        charuco_corners,
                        charuco_ids,
                        (0, 255, 0)  # 绿色
                    )
                    
                    # 估计位姿
                    if num_charuco_corners >= 4:
                        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            charuco_corners,
                            charuco_ids,
                            board,
                            camera_matrix,
                            distortion_coeffs,
                            None,
                            None
                        )
                        
                        if retval:
                            pose_estimated = True
                            # 绘制坐标轴
                            cv2.drawFrameAxes(
                                display_image,
                                camera_matrix,
                                distortion_coeffs,
                                rvec,
                                tvec,
                                length=0.05  # 50mm轴长
                            )
            
            # 显示信息
            info_y = 30
            line_height = 35
            
            # 状态信息
            if pose_estimated:
                status_text = "检测状态: ✓ 位姿估计成功"
                status_color = (0, 255, 0)  # 绿色
            elif num_charuco_corners > 0:
                status_text = f"检测状态: 角点不足 ({num_charuco_corners}/4)"
                status_color = (0, 255, 255)  # 黄色
            elif num_markers > 0:
                status_text = f"检测状态: 仅检测到标记 ({num_markers})"
                status_color = (0, 165, 255)  # 橙色
            else:
                status_text = "检测状态: ✗ 未检测到标定板"
                status_color = (0, 0, 255)  # 红色
            
            cv2.putText(display_image, status_text, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # 详细信息
            info_y += line_height
            cv2.putText(display_image, f"ArUco标记数: {num_markers}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += line_height
            cv2.putText(display_image, f"ChArUco角点数: {num_charuco_corners}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 统计信息
            detection_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0
            info_y += line_height
            cv2.putText(display_image, f"检测率: {detection_rate:.1f}%",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 提示信息
            help_text = "按 's' 保存图像 | 按 'q' 退出"
            cv2.putText(display_image, help_text,
                       (10, display_image.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 显示图像
            cv2.imshow("ChArUco Detection Test", display_image)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n退出测试")
                break
            elif key == ord('s'):
                # 保存图像
                filename = f"charuco_test_{saved_count:03d}.png"
                cv2.imwrite(filename, display_image)
                print(f"✓ 已保存图像: {filename}")
                saved_count += 1
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        # 统计信息
        print("\n" + "="*60)
        print("检测统计:")
        print(f"  总帧数: {total_frames}")
        print(f"  成功检测帧数: {detected_frames}")
        print(f"  检测率: {detection_rate:.1f}%")
        print(f"  保存图像数: {saved_count}")
        print("="*60)
        
        # 评估
        print("\n质量评估:")
        if detection_rate > 90:
            print("  ✓ 优秀！标定板质量很好，检测稳定")
        elif detection_rate > 70:
            print("  ✓ 良好，可以用于标定")
        elif detection_rate > 50:
            print("  ⚠ 一般，建议检查:")
            print("    - 打印质量")
            print("    - 光照条件")
            print("    - 标定板平整度")
        else:
            print("  ✗ 较差，建议:")
            print("    - 重新打印标定板")
            print("    - 改善光照")
            print("    - 调整距离和角度")
        
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_charuco_detection()
