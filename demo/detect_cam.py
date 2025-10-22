"""
实时物体检测程序
按空格键检测所有类别的物体（置信度阈值0.1）
按 'q' 键退出
"""

import sys
import os
import cv2

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.camera import Camera
from lib.yolo_and_sam import YOLOSegmentator


def main():
    # 常见物体类别列表（YOLO World 支持的类别）
    categories = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    print("=" * 60)
    print("实时物体检测程序")
    print("=" * 60)
    print("操作说明：")
    print("  - 按 [空格键] 进行物体检测（置信度阈值: 0.1）")
    print("  - 按 [q] 键退出程序")
    print("=" * 60)
    
    # 初始化相机
    print("\n正在初始化相机...")
    try:
        cam = Camera(camera_model='AUTO')
    except Exception as e:
        print(f"❌ 相机初始化失败: {e}")
        print("提示: 请确保 RealSense 相机已连接")
        return
    
    # 初始化 YOLO 检测器
    print("\n正在加载 YOLO World 模型...")
    try:
        detector = YOLOSegmentator(
            yolo_weights=r"D:\Projects\capture\weights\yolov8s-world.pt",
            sam_weights=r"D:\Projects\capture\weights\sam2.1_b.pt"
        )
        print("✓ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        cam.release()
        return
    
    # 创建输出目录
    os.makedirs("result", exist_ok=True)
    
    print("\n✓ 初始化完成！相机已就绪")
    print("\n正在显示实时画面...\n")
    
    frame_count = 0
    
    try:
        while True:
            # 获取相机图像
            color_image, _ = cam.get_frames()
            
            if color_image is None:
                print("⚠ 无法获取图像")
                continue
            
            frame_count += 1
            
            # 在图像上显示提示信息
            display_image = color_image.copy()
            cv2.putText(display_image, "Press SPACE to detect objects", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, "Press 'q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, f"Frame: {frame_count}", 
                       (10, display_image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示图像
            cv2.imshow('Real-time Detection', display_image)
            
            # 等待键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            # 按 'q' 退出
            if key == ord('q'):
                print("\n退出程序...")
                break
            
            # 按空格键进行检测
            elif key == ord(' '):
                print("\n" + "=" * 60)
                print("🔍 开始检测物体...")
                print("=" * 60)
                
                try:
                    # 使用 YOLO World 检测
                    result = detector.detect(
                        image=color_image,
                        categories=categories,
                        output_dir="result",
                        conf=0.1,  # 置信度阈值 0.1
                        imgsz=640,
                        save_result=True
                    )
                    
                    if result['success']:
                        print(f"\n✓ 检测完成！共检测到 {len(result['det_bboxes'])} 个物体")
                        print(f"  结果已保存到: {result['detection_path']}")
                        
                        # 显示检测详情
                        print("\n检测详情：")
                        det_bboxes = result['det_bboxes']
                        for i in range(len(det_bboxes)):
                            class_id = int(det_bboxes.cls[i].cpu())
                            class_name = detector.det_model.names[class_id]
                            confidence = float(det_bboxes.conf[i].cpu())
                            bbox = det_bboxes.xyxy[i].cpu().numpy()
                            print(f"  [{i+1}] {class_name}: {confidence:.3f} - bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                        
                        # 显示检测结果图像
                        if result['detection_path'] and os.path.exists(result['detection_path']):
                            det_img = cv2.imread(result['detection_path'])
                            cv2.imshow('Detection Result', det_img)
                            print("\n提示: 检测结果已在新窗口显示")
                    else:
                        print("⚠ 未检测到任何物体（置信度阈值: 0.1）")
                    
                except Exception as e:
                    print(f"❌ 检测过程出错: {e}")
                
                print("=" * 60)
                print("继续显示实时画面...")
                print()
    
    except KeyboardInterrupt:
        print("\n\n检测到中断信号 (Ctrl+C)，正在退出...")
    
    finally:
        # 释放资源
        print("\n正在释放资源...")
        cam.release()
        cv2.destroyAllWindows()
        print("✓ 资源已释放，程序结束")


if __name__ == "__main__":
    main()

