import torch
from ultralytics import YOLOWorld, FastSAM
import numpy as np
import time
import os
import cv2
from PIL import Image
from functools import wraps


def timer(func):
    """
    计时装饰器：统计函数执行时间
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 获取函数名
        func_name = func.__name__
        print(f"⏱️  [{func_name}] took {elapsed_time:.3f} seconds")
        
        return result
    return wrapper


class YOLOSegmentator:
    def __init__(self, yolo_weights="weights/yolov8s-world.pt", sam_weights="weights/FastSAM-s.pt"):
        """
        正确初始化 YOLO-World 和 FastSAM 模型。
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # --- 正确的 YOLO-World 初始化流程 ---

        # 1. 加载模型 (默认在 CPU 上)
        print("Loading YOLO-World model ...")
        self.det_model = YOLOWorld(yolo_weights)
        self.det_model.to(self.device)

        # 加载 FastSAM 模型
        print("Loading FastSAM model...")
        self.seg_model = FastSAM(sam_weights)
        self.seg_model.to(self.device)

    @timer
    def detect(self, image, categories, output_dir="result", conf=0.1, imgsz=640, save_result=True):
        """
        使用 YOLO-World 检测图像中的物体
        
        参数:
            image: 图像数据 (numpy 数组格式)/图像路径
            categories: 要检测的类别列表
            output_dir: 输出目录
            conf: 置信度阈值
            imgsz: 图像大小
            
        返回:
            dict: {
                'success': bool,
                'det_bboxes': YOLO result object (包含boxes, cls, conf等),
                'detection_path': str (保存的检测结果路径)
            }
        """

        print("\n--- Running YOLO-World Detection ---")

        self.det_model.set_classes(categories)

        if type(image) == str:
            pass
        elif isinstance(image, list):
            image = Image.fromarray(image)
        #如果predict多个img，对于有多个img的结果.list
        det_results = self.det_model.predict(
            source=image,
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )

        #det_results包含了大量的信息，包括.boxes(xyxy,cls,conf),orig_img,names,speed等等
        result = det_results[0]

        #错误提示
        if len(result.boxes) == 0:
            print("No objects detected by YOLO-World.")
            return {'success': False, 'det_bboxes': None, 'detection_path': None}

        #保存结果
        if save_result:
             # 保存检测结果（带标签的检测框）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            det_output_filename = os.path.join(output_dir, f"det_{os.path.basename(image) if isinstance(image, str) else f'{timestamp}.png'}")
            result.save(det_output_filename)
            print('YOLO-World results:')
            print(f"Detection result saved to {det_output_filename}")
            print(f"Found {len(result.boxes)} objects")

        result_dict = {
            'success': True,
            'det_bboxes': result.boxes,
            'detection_path': det_output_filename if save_result else None
        }

        return result_dict

    @timer
    def segment(self, image, bbox_gpu, output_dir, save_result):
        """
        使用 FastSAM 对指定边界框进行分割
        
        参数:
            image: 图像数据 (numpy 数组格式) | 图片地址
            bbox_gpu: 边界框张量 (在GPU上，格式为 xyxy)
            output_dir: 输出目录
            save_result: 是否保存分割结果
            
        返回:
            dict: {
                'success': bool,
                'mask': np.ndarray (分割掩码),
                'segmentation_path': str (如果保存了结果)
            }
        """

        sam_results = self.seg_model(image, bboxes=bbox_gpu.unsqueeze(0))


        # 提取掩码数据
        mask = None
        if sam_results[0].masks is not None and len(sam_results[0].masks) > 0:
            mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)


        # 保存分割结果
        if save_result:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            seg_output_filename = os.path.join(output_dir, f"seg_{os.path.basename(image) if isinstance(image, str) else f'{timestamp}.png'}")
            sam_results[0].save(seg_output_filename)
            print('FastSAM results:')
            print(f"Segmentation result saved to {seg_output_filename}")

        result_dict = {
            'success': True,
            'mask': mask,
            'segmentation_path': seg_output_filename if save_result else None
        }

        return result_dict

    def detect_and_segment(self, image, categories, output_dir="./result", conf=0.1, imgsz=640, save_result=True):
        """
        先用 YOLO-World 检测，然后用 FastSAM 分割置信度最高的目标。
        **只分割全局置信度最高的1个物体
        """
        if save_result:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        # 步骤1: 检测
        det_result = self.detect(image, categories, output_dir, conf, imgsz)
        
        if not det_result['success']:
            return det_result
        
        det_bboxes = det_result['det_bboxes']
        

        # 找到置信度最高的一个物体-------------------------------------------------------
        max_index_gpu = det_bboxes.conf.argmax()
        best_box_gpu = det_bboxes.xyxy[max_index_gpu]
        best_class_id_gpu = det_bboxes.cls[max_index_gpu]
        best_confidence_gpu = det_bboxes.conf[max_index_gpu]

        # 将结果移至 CPU
        best_box_xyxy = best_box_gpu.cpu().numpy().astype(int)
        best_class_name = self.det_model.names[int(best_class_id_gpu.cpu())]
        best_confidence = float(best_confidence_gpu.cpu())

        #-------------------------------------------------------------------------------------------

        print(f"\nBest object: {best_class_name} (confidence: {best_confidence:.2f})")

        # 步骤2: 分割
        seg_result = self.segment(image, best_box_gpu, output_dir, save_result=save_result)

        return {
            'success': True,
            'best_object': {
                'class': best_class_name,
                'confidence': best_confidence,
                'bbox_xyxy': best_box_xyxy.tolist()
            },
            'mask': seg_result['mask'],
            'detection_path': det_result['detection_path'] if save_result else None,
            'segmentation_path': seg_result.get('segmentation_path') if save_result else None
        }
    
    def detect_and_segment_all(self, image, categories, output_dir="./result", conf=0.1, imgsz=640, save_result=False, 
                               multi_instance_classes=None, multi_conf_threshold=0.3, multi_max_count=3):
        """
        检测并分割所有指定类别的物体
        
        参数:
            image: 图像路径或数组
            categories: 要检测的类别列表
            output_dir: 输出目录
            conf: 最低置信度阈值
            imgsz: 图像大小
            save_result: 是否保存结果
            multi_instance_classes: 允许保留多个实例的类别列表（如 ['bowl']）
            multi_conf_threshold: 多实例类别的置信度阈值
            multi_max_count: 多实例类别的最大保留数量
        
        返回:
            dict: {
                'success': bool,
                'objects': [
                    {
                        'class': str,
                        'confidence': float,
                        'bbox_xyxy': list,
                        'mask': np.ndarray
                    },
                    ...
                ]
            }
        """
        if save_result:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        if multi_instance_classes is None:
            multi_instance_classes = []
            
        # 步骤1: 检测
        det_result = self.detect(image, categories, output_dir, conf, imgsz)
        
        if not det_result['success']:
            return det_result
        
        det_bboxes = det_result['det_bboxes']

        # 步骤2: 筛选物体
        print(f"\nFound {len(det_bboxes)} objects, filtering...")

        # 创建 (idx, class_name, confidence) 的列表并按类别分组
        objects_by_class = {}
        for idx in range(len(det_bboxes)):
            info = {
                'idx': idx,
                'class_name': self.det_model.names[int(det_bboxes.cls[idx].cpu())],
                'confidence': float(det_bboxes.conf[idx].cpu())
            }
            class_name = info['class_name']
            
            # 判断是否是多实例类别
            if class_name in multi_instance_classes:
                # 多实例类别：保留所有满足阈值的（按置信度排序）
                if class_name not in objects_by_class:
                    objects_by_class[class_name] = []
                if info['confidence'] >= multi_conf_threshold:
                    objects_by_class[class_name].append(info)
            else:
                # 单实例类别：只保留置信度最高的
                if class_name not in objects_by_class or info['confidence'] > objects_by_class[class_name]['confidence']:
                    objects_by_class[class_name] = info
        
        # 处理多实例类别：排序并限制数量
        selected_objects = []
        for class_name, objs in objects_by_class.items():
            if class_name in multi_instance_classes:
                # 按置信度降序排序，取前 N 个
                objs_sorted = sorted(objs, key=lambda x: x['confidence'], reverse=True)
                selected_objects.extend(objs_sorted[:multi_max_count])
                print(f"  {class_name}: {len(objs_sorted[:multi_max_count])} instances (from {len(objs)} detected, conf>={multi_conf_threshold})")
            else:
                # 单实例类别
                selected_objects.append(objs)
                print(f"  {class_name}: 1 instance (best)")
        
        print(f"\nTotal objects to segment: {len(selected_objects)}")
    
        # 步骤3: 对筛选后的物体进行分割
        print(f"\nStarting segmentation...")
        
        detected_objects = []
        for info in selected_objects:
            obj = {
                'class': info['class_name'],
                'confidence': info['confidence'],
                'bbox_xyxy': det_bboxes.xyxy[info['idx']].cpu().numpy().astype(int).tolist(),
                'mask': self.segment(image, det_bboxes.xyxy[info['idx']], output_dir, save_result=save_result)['mask']
            }
            detected_objects.append(obj)
            print(f"  Segmented: {info['class_name']} (conf={info['confidence']:.2f})")
        
        print(f"\nTotal objects processed: {len(detected_objects)}")
        
        # 步骤4: 创建合并的可视化图像（检测框 + 分割掩码）
        combined_path = None
        if save_result and len(detected_objects) > 0:
            combined_path = self._create_combined_visualization(
                image, detected_objects, output_dir
            )
        
        return {
            'success': True,
            'objects': detected_objects,
            'detection_path': det_result['detection_path'],
            'combined_path': combined_path
        }
    
    def _create_combined_visualization(self, image, objects, output_dir):
        """
        创建合并的可视化图像（检测框 + 分割掩码）
        
        参数:
            image: 图像路径或数组
            objects: 检测到的物体列表
            output_dir: 输出目录
        
        返回:
            combined_path: 合并图像的保存路径
        """
        # 读取原始图像
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)
        
        if len(img.shape) == 2:  # 灰度图
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 创建掩码叠加层
        overlay = img.copy()
        
        # 为不同物体定义颜色（BGR格式）
        colors = [
            (255, 0, 0),      # 蓝色
            (0, 255, 0),      # 绿色
            (0, 0, 255),      # 红色
            (255, 255, 0),    # 青色
            (255, 0, 255),    # 紫色
            (0, 255, 255),    # 黄色
            (128, 0, 255),    # 粉色
            (0, 128, 255),    # 橙色
        ]
        
        for idx, obj in enumerate(objects):
            color = colors[idx % len(colors)]
            
            # 1. 绘制分割掩码（半透明填充）
            if obj['mask'] is not None:
                mask = obj['mask']
                if mask.shape[:2] != img.shape[:2]:
                    # 如果掩码尺寸不匹配，调整大小
                    mask = cv2.resize(mask.astype(np.uint8), 
                                     (img.shape[1], img.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
                
                # 创建彩色掩码
                colored_mask = np.zeros_like(img)
                colored_mask[mask > 0] = color
                
                # 叠加掩码（透明度0.4）
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)
            
            # 2. 绘制检测框
            bbox = obj['bbox_xyxy']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # 3. 绘制标签背景
            label = f"{obj['class']} {obj['confidence']:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # 标签背景框
            cv2.rectangle(
                overlay,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1  # 填充
            )
            
            # 4. 绘制标签文字（白色）
            cv2.putText(
                overlay,
                label,
                (x1 + 2, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),  # 白色
                thickness
            )
        
        # 保存合并图像
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        combined_filename = f"combined_{timestamp}.jpg"
        combined_path = os.path.join(output_dir, combined_filename)
        cv2.imwrite(combined_path, overlay)
        
        print(f"\n💾 合并可视化已保存: {combined_path}")
        
        return combined_path

if __name__ == "__main__":
        # 你的图片路径 (请确保图片存在)
    image_to_process = "./images/cup2.jpg"  # <--- 在这里修改你的图片路径！

    # 你想要检测的物体类别 (YOLO-World 对自然语言描述更友好)
    categories_to_find = ["cup"]

    segmentator = YOLOSegmentator()

    # --- 执行检测和分割 ---
    result = segmentator.detect_and_segment(
        image=image_to_process,
        categories=categories_to_find
    )
