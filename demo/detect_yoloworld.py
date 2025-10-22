"""
单张图片物体检测程序
直接硬编码图片路径进行检测
"""

import sys
import os
import cv2

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.yolo_and_sam import YOLOSegmentator


def main():
    # 硬编码的配置参数
    image_path = "lid1/color.png"  # 修改这里的图片路径
    confidence_threshold = 0.1
    output_dir = "result"
    show_result = True
    
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
    print("单张图片物体检测程序")
    print("=" * 60)
    print(f"输入图片: {image_path}")
    print(f"置信度阈值: {confidence_threshold}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 检查输入图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误: 图片文件不存在: {image_path}")
        return
    
    # 读取图片
    print(f"\n正在读取图片: {image_path}")
    try:
        color_image = cv2.imread(image_path)
        if color_image is None:
            print(f"❌ 错误: 无法读取图片文件: {image_path}")
            return
        print(f"✓ 图片读取成功，尺寸: {color_image.shape[1]}x{color_image.shape[0]}")
    except Exception as e:
        print(f"❌ 读取图片失败: {e}")
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
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🔍 开始检测物体...")
    print("=" * 60)
    
    try:
        # 使用 YOLO World 检测
        result = detector.detect(
            image=color_image,
            categories=categories,
            output_dir=output_dir,
            conf=confidence_threshold,
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
            if show_result and result['detection_path'] and os.path.exists(result['detection_path']):
                det_img = cv2.imread(result['detection_path'])
                cv2.imshow('Detection Result', det_img)
                print("\n提示: 检测结果已在新窗口显示，按任意键关闭窗口")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"⚠ 未检测到任何物体（置信度阈值: {confidence_threshold}）")
        
    except Exception as e:
        print(f"❌ 检测过程出错: {e}")
    
    print("=" * 60)
    print("检测完成！")


if __name__ == "__main__":
    main()

