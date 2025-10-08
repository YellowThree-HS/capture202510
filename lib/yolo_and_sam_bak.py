from ultralytics import YOLOWorld, SAM
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob

# 获取 images 文件夹下的所有图像文件
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join("images", ext)))
    image_files.extend(glob.glob(os.path.join("images", ext.upper())))

if not image_files:
    print("在 images 文件夹中没有找到图像文件！")
    exit()

print(f"找到 {len(image_files)} 个图像文件")

# 创建结果保存目录
os.makedirs("result", exist_ok=True)

# Initialize YOLO-World model
det_model = YOLOWorld("weights/yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes
# 设置YOLO World检测的类别为杯子
det_model.set_classes(["cup"])

# Initialize SAM (Segment Anything) model
sam_model = SAM("weights/sam2.1_b.pt")

# 处理每个图像
for i, image_path in enumerate(image_files, 1):
    print(f"\n正在处理第 {i}/{len(image_files)} 个图像: {os.path.basename(image_path)}")
    
    try:
        # Execute inference on the specified image
        det_results = det_model.predict(image_path, save=False, verbose=False)
        
        # 检查是否检测到对象
        if len(det_results[0].boxes) > 0:
            # Get detected boxes (results.xywh for boxes, results.names for class names)
            det_boxes = det_results[0].boxes.xywh.cpu().numpy()  # Format: [x_center, y_center, width, height]
            scores = det_results[0].boxes.conf.cpu().numpy()  # Confidence scores
            labels = det_results[0].boxes.cls.cpu().numpy()  # Class labels
            
            print(f"检测到 {len(det_boxes)} 个对象:")
            for j, (box, score, label) in enumerate(zip(det_boxes, scores, labels)):
                class_name = det_model.names[int(label)]
                print(f"  对象 {j+1}: {class_name} (置信度: {score:.3f})")
            
            # Find the index of the box with the highest confidence score
            max_index = np.argmax(scores)
            
            # Extract the corresponding box coordinates (max box)
            max_box = det_boxes[max_index]
            x_center, y_center, width, height = max_box
            
            # Define the box's coordinates (xmin, ymin, xmax, ymax)
            xmin = int(x_center - width / 2)
            ymin = int(y_center - height / 2)
            xmax = int(x_center + width / 2)
            ymax = int(y_center + height / 2)
            
            print(f"最高置信度对象: {det_model.names[int(labels[max_index])]} (置信度: {scores[max_index]:.3f})")
            print(f"边界框坐标: ({xmin}, {ymin}, {xmax}, {ymax})")
            
            # Run inference with bboxes prompt for segmentation
            sam_results = sam_model(image_path, bboxes=[[xmin, ymin, xmax, ymax]])
            
            # 保存检测结果图像
            det_output_path = os.path.join("result", f"detected_{os.path.splitext(os.path.basename(image_path))[0]}.jpg")
            det_results[0].save(det_output_path)
            print(f"检测结果已保存到: {det_output_path}")
            
            # 保存分割结果图像
            seg_output_path = os.path.join("result", f"segmented_{os.path.splitext(os.path.basename(image_path))[0]}.jpg")
            sam_results[0].save(seg_output_path)
            print(f"分割结果已保存到: {seg_output_path}")
            
            # 保存分割掩码（mask）
            if hasattr(sam_results[0], 'masks') and sam_results[0].masks is not None:
                # 获取掩码数据
                masks = sam_results[0].masks.data.cpu().numpy()  # 形状: [num_masks, height, width]
                
                # 保存每个掩码
                for mask_idx, mask in enumerate(masks):
                    # 将掩码转换为0-255的灰度图像
                    mask_image = (mask * 255).astype(np.uint8)
                    
                    # 保存掩码图像
                    mask_output_path = os.path.join("result", f"mask_{os.path.splitext(os.path.basename(image_path))[0]}_{mask_idx}.png")
                    Image.fromarray(mask_image).save(mask_output_path)
                    print(f"掩码 {mask_idx+1} 已保存到: {mask_output_path}")
                
                # 保存所有掩码的合成图像
                if len(masks) > 1:
                    combined_mask = np.max(masks, axis=0)  # 取所有掩码的最大值
                    combined_mask_image = (combined_mask * 255).astype(np.uint8)
                    combined_mask_path = os.path.join("result", f"combined_mask_{os.path.splitext(os.path.basename(image_path))[0]}.png")
                    Image.fromarray(combined_mask_image).save(combined_mask_path)
                    print(f"合成掩码已保存到: {combined_mask_path}")
            else:
                print("未找到分割掩码数据")
            
        else:
            print("未检测到任何对象！")
            # 即使没有检测到对象，也保存原图
            img = Image.open(image_path)
            output_path = os.path.join("result", f"no_detection_{os.path.splitext(os.path.basename(image_path))[0]}.jpg")
            img.save(output_path)
            print(f"原图已保存到: {output_path}")
            
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        continue

print(f"\n所有图像处理完成！结果保存在 result 文件夹中。")
