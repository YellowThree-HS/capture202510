from ultralytics import YOLO
from pathlib import Path
import sys
import argparse

def test_model(weights_path, test_image_path, save_dir='runs/test'):
    """
    使用指定权重对单个图像进行测试
    
    Args:
        weights_path: 模型权重文件路径 (例如: runs/segment/grasp_seg/weights/best.pt)
        test_image_path: 测试图像路径 (例如: image/20251016_152805_color.png)
        save_dir: 结果保存目录
    """
    
    # 检查文件是否存在
    weights_path = Path(weights_path)
    test_image_path = Path(test_image_path)
    
    if not weights_path.exists():
        print(f"❌ 错误: 权重文件不存在: {weights_path}")
        return
    
    if not test_image_path.exists():
        print(f"❌ 错误: 测试图像不存在: {test_image_path}")
        return
    
    print("=" * 60)
    print("YOLOv8-Seg 模型测试")
    print("=" * 60)
    print(f"权重文件: {weights_path}")
    print(f"测试图像: {test_image_path}")
    print(f"保存目录: {save_dir}")
    print("=" * 60)
    
    # 加载模型
    print("\n正在加载模型...")
    model = YOLO(str(weights_path))
    print("✓ 模型加载完成")
    
    # 进行预测
    print("\n正在进行推理...")
    results = model.predict(
        source=str(test_image_path),
        save=True,              # 保存可视化结果
        save_txt=True,          # 保存标签文件
        save_conf=True,         # 保存置信度
        conf=0.25,              # 置信度阈值
        iou=0.7,                # NMS的IoU阈值
        project=save_dir,       # 保存目录
        name='predict',         # 子目录名称
        exist_ok=True,          # 允许覆盖
        show_labels=True,       # 显示标签
        show_conf=True,         # 显示置信度
        line_width=2,           # 边界框线宽
    )
    
    # 显示结果统计
    print("\n" + "=" * 60)
    print("推理结果:")
    print("=" * 60)
    
    for i, result in enumerate(results):
        print(f"\n图像 {i+1}: {result.path}")
        print(f"  - 图像尺寸: {result.orig_shape}")
        
        if result.masks is not None:
            num_detections = len(result.masks)
            print(f"  - 检测到的对象数量: {num_detections}")
            
            # 显示每个检测的详细信息
            for j in range(num_detections):
                class_id = int(result.boxes.cls[j])
                class_name = result.names[class_id]
                confidence = float(result.boxes.conf[j])
                box = result.boxes.xyxy[j].tolist()
                
                print(f"\n  对象 {j+1}:")
                print(f"    类别: {class_name} (ID: {class_id})")
                print(f"    置信度: {confidence:.4f}")
                print(f"    边界框: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        else:
            print("  - 未检测到对象")
    
    print("\n" + "=" * 60)
    print(f"✓ 结果已保存到: {save_dir}/predict")
    print("=" * 60)
    print("\n包含以下文件:")
    print("  - 可视化图像 (带分割掩码和边界框)")
    print("  - 标签文件 (.txt)")
    print("  - 置信度信息")
    
    return results


if __name__ == '__main__':
    # 设置命令行参数
    # parser = argparse.ArgumentParser(description='YOLOv8-Seg 模型测试脚本')
    # parser.add_argument('--weights', '-w', type=str, required=True,
    #                     help='模型权重文件路径 (例如: runs/segment/grasp_seg/weights/best.pt)')
    # parser.add_argument('--image', '-i', type=str, required=True,
    #                     help='测试图像路径 (例如: image/20251016_152805_color.png)')
    # parser.add_argument('--save-dir', '-s', type=str, default='runs/test',
    #                     help='结果保存目录 (默认: runs/test)')
    
    # args = parser.parse_args()
    
    weights_path = r"D:\Projects\capture\weights\best1021.pt"
    image_path = r"D:\Projects\capture\lid\color_20251022_170325.png"
    save_dir = r"D:\Projects\capture\demo\result" 
    test_model(weights_path, image_path, save_dir)
    
    # 运行测试
    # test_model(args.weights, args.image, args.save_dir)


# 使用示例:
# python test.py --weights runs/segment/grasp_seg/weights/best.pt --image image/20251016_152805_color.png
# 或简写:
# python test.py -w runs/segment/grasp_seg/weights/best.pt -i image/20251016_152805_color.png
