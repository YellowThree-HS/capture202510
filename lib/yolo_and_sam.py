import torch
from ultralytics import YOLOWorld, FastSAM
import numpy as np
import time
import os
from PIL import Image


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

    def detect_and_segment(self, image_path, categories, output_dir="result", conf=0.1, imgsz=640):
        """
        先用 YOLO-World 检测，然后用 FastSAM 分割置信度最高的目标。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # --- 步骤 1: 检测 (YOLO-World) ---
        print("\n--- Running YOLO-World Detection ---")
        start_det = time.time()

        self.det_model.set_classes(categories)

        det_results = self.det_model.predict(
            image_path,
            conf=conf,
            imgsz=imgsz,
            device=self.device,  # 明确指定设备
            verbose=False  # 减少不必要的输出
        )

        result = det_results[0]
        end_det = time.time()
        print(f"YOLO-World detection took {end_det - start_det:.3f} seconds.")

        if len(result.boxes) == 0:
            print("No objects detected by YOLO-World.")
            return {'success': False, 'message': 'No objects detected.'}

        if len(result.boxes.conf) == 0:
            print("No objects detected after filtering.")
            return {'success': False, 'message': 'No objects detected after filtering.'}

        max_index_gpu = result.boxes.conf.argmax()
        best_box_gpu = result.boxes.xyxy[max_index_gpu]
        best_class_id_gpu = result.boxes.cls[max_index_gpu]
        best_confidence_gpu = result.boxes.conf[max_index_gpu]

        # 将最终结果移至 CPU
        best_box_xyxy = best_box_gpu.cpu().numpy().astype(int)
        best_class_name = self.det_model.names[int(best_class_id_gpu.cpu())]
        best_confidence = float(best_confidence_gpu.cpu())

        # 保存检测结果（带标签的检测框）
        det_output_filename = os.path.join(output_dir, f"det_{os.path.basename(image_path)}")
        result.save(det_output_filename)
        print(f"Detection result with labels saved to {det_output_filename}")

        start_sam = time.time()
        sam_results = self.seg_model(image_path, bboxes=best_box_gpu)

        # 保存分割结果
        seg_output_filename = os.path.join(output_dir, f"seg_{os.path.basename(image_path)}")
        sam_results[0].save(seg_output_filename)
        print(f"Segmentation result saved to {seg_output_filename}")

        end_sam = time.time()
        print(f"FastSAM segmentation took {end_sam - start_sam:.3f} seconds.")

        # 提取掩码数据
        mask = None
        if sam_results[0].masks is not None and len(sam_results[0].masks) > 0:
            # 获取第一个掩码（通常是最主要的物体）
            mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
            print(f"Mask extracted with shape: {mask.shape}")

        return {
            'success': True,
            'best_object': {
                'class': best_class_name,
                'confidence': best_confidence,
                'bbox_xyxy': best_box_xyxy.tolist()
            },
            'detection_path': det_output_filename,
            'segmentation_path': seg_output_filename,
            'mask': mask  # 添加掩码数据
        }
    
    def detect_and_segment_all(self, image_path, categories, output_dir="result", conf=0.1, imgsz=640):
        """
        检测并分割所有指定类别的物体
        
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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # --- 步骤 1: 检测 (YOLO-World) ---
        print("\n--- Running YOLO-World Detection (All Objects) ---")
        start_det = time.time()

        self.det_model.set_classes(categories)

        det_results = self.det_model.predict(
            image_path,
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )

        result = det_results[0]
        end_det = time.time()
        print(f"YOLO-World detection took {end_det - start_det:.3f} seconds.")

        if len(result.boxes) == 0:
            print("No objects detected by YOLO-World.")
            return {'success': False, 'message': 'No objects detected.', 'objects': []}

        # 保存检测结果
        det_output_filename = os.path.join(output_dir, f"det_{os.path.basename(image_path)}")
        result.save(det_output_filename)
        print(f"Detection result saved to {det_output_filename}")

        # --- 步骤 2: 对每个类别只保留置信度最高的物体 ---
        print(f"\nFound {len(result.boxes)} objects, filtering to best per category...")
        
        # 按类别分组，找出每个类别置信度最高的物体
        best_per_class = {}
        for idx in range(len(result.boxes)):
            class_id = int(result.boxes.cls[idx].cpu())
            class_name = self.det_model.names[class_id]
            confidence = float(result.boxes.conf[idx].cpu())
            
            # 如果这个类别还没有记录，或者当前物体置信度更高
            if class_name not in best_per_class or confidence > best_per_class[class_name]['confidence']:
                best_per_class[class_name] = {
                    'idx': idx,
                    'confidence': confidence,
                    'class_name': class_name
                }
        
        print(f"Filtered to {len(best_per_class)} objects (one per category):")
        for class_name, info in best_per_class.items():
            print(f"  - {class_name}: confidence {info['confidence']:.2f}")
        
        # --- 步骤 3: 对筛选后的物体进行分割 ---
        detected_objects = []
        
        for class_name, info in best_per_class.items():
            idx = info['idx']
            box_gpu = result.boxes.xyxy[idx]
            class_id_gpu = result.boxes.cls[idx]
            confidence_gpu = result.boxes.conf[idx]
            
            # 转换到CPU
            box_xyxy = box_gpu.cpu().numpy().astype(int)
            confidence = float(confidence_gpu.cpu())
            
            print(f"\nProcessing {class_name} (confidence: {confidence:.2f})...")
            
            # 使用FastSAM分割
            start_sam = time.time()
            sam_results = self.seg_model(image_path, bboxes=box_gpu.unsqueeze(0))
            end_sam = time.time()
            
            # 提取掩码
            mask = None
            if sam_results[0].masks is not None and len(sam_results[0].masks) > 0:
                mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
                print(f"  Mask shape: {mask.shape}, SAM time: {end_sam - start_sam:.3f}s")
            
            detected_objects.append({
                'class': class_name,
                'confidence': confidence,
                'bbox_xyxy': box_xyxy.tolist(),
                'mask': mask
            })
        
        # 保存综合分割结果
        print(f"\nTotal objects after filtering: {len(detected_objects)}")
        
        return {
            'success': True,
            'objects': detected_objects,
            'detection_path': det_output_filename
        }

if __name__ == "__main__":
        # 你的图片路径 (请确保图片存在)
    image_to_process = "./images/cup2.jpg"  # <--- 在这里修改你的图片路径！

    # 你想要检测的物体类别 (YOLO-World 对自然语言描述更友好)
    categories_to_find = ["cup"]

    segmentator = YOLOSegmentator()

    # --- 执行检测和分割 ---
    result = segmentator.detect_and_segment(
        image_path=image_to_process,
        categories=categories_to_find
    )
