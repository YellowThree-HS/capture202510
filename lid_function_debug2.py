# 包含一个类
# lid的目标检测分割
# 并进行点云提取
# 转换到基坐标系
# 选择前50%的点云，用矩形框框选出来
# 选择靠近机械臂基座的长边中心，在中心周围选择10个点，Z轴垂直桌面，X轴沿矩形方向，Y轴在平面上指向矩形外侧

import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
import pyrealsense2 as rs
from datetime import datetime
import open3d as o3d

class LidFunction:
    def __init__(self):
        self.model = YOLO("weights/best1021.pt")
        self.conf_threshold = 0.5
        self.target_class = "lid"
        self.color_image_path = "lid/color_20251022_170343.png"
        self.depth_image_path = "lid/depth_20251022_170343.png"
        self.color_image = cv2.imread(self.color_image_path)
        self.depth_image = cv2.imread(self.depth_image_path, cv2.IMREAD_UNCHANGED)
        
        self.height_percentage = 0.5
        self.num_points = 10

        self.intrinsics = np.array([
            [652.76428223,   0.,         650.07250977],
            [  0.,         651.92443848, 366.10205078],
            [  0.,           0.,           1.        ]
        ])
        self.robot_matrix = np.array([
            [-0.99624,   0.065169,  -0.057046, -0.4543],
            [ 0.062097,  0.9966,     0.054061,  0.11598],
            [ 0.060375,  0.050315,  -0.99691,   0.36765],
            [ 0.,        0.,         0.,        1.      ]
        ], dtype=np.float64)
        self.hand_eye_matrix = np.array([
            [ 0.01949938,  0.99822277,  0.05631227,  0.0758227 ],
            [-0.99977,     0.01997063, -0.00781785,  0.05666132],
            [-0.00892854, -0.05614688,  0.9983826,  -0.10319311],
            [ 0.,          0.,          0.,          1.        ]
        ])
        self.points = []
        self.colors = []
        self.transformed_points = []
        self.top_points = []
        self.top_colors = []
        self.bbox = []
    
    def detect_with_yolo(self):
        results = self.model.predict(
            source=self.color_image,
            save=False,
            conf=self.conf_threshold,
            iou=0.7,
            verbose=False
        )

        result = results[0]
        class_names = result.names
        target_class_id = None
        for class_id, class_name in class_names.items():
            if class_name == self.target_class:
                target_class_id = class_id
                break
        all_bboxes = result.boxes.xyxy.cpu().numpy()
        all_masks = result.masks.data.cpu().numpy()
        all_confs = result.boxes.conf.cpu().numpy()
        all_classes = result.boxes.cls.cpu().numpy()
        target_indices = np.where(all_classes == target_class_id)[0]
        filtered_bboxes = all_bboxes[target_indices]
        filtered_masks = all_masks[target_indices]
        filtered_confs = all_confs[target_indices]
        filtered_classes = all_classes[target_indices]
        print(f"检测到 {len(filtered_bboxes)} 个 {self.target_class}")
        return filtered_bboxes, filtered_masks

    def depth_to_pointcloud(self, mask):
        depth_scale = 0.0001
        mask = cv2.resize(mask.astype(np.float32), 
                    (self.depth_image.shape[1], self.depth_image.shape[0]), 
                    interpolation=cv2.INTER_NEAREST)

        binary_mask = (mask > 0.5).astype(np.uint8)
        y_coords, x_coords = np.where(binary_mask == 1)
        if len(x_coords) == 0:
            return np.array([]), np.array([])
        depth_values = self.depth_image[y_coords, x_coords]
        if depth_values.ndim > 1:
            depth_values = depth_values.flatten()
        valid_depth_mask = (depth_values > 0) & (depth_values < 10000)
        
        if np.sum(valid_depth_mask) == 0:
            return np.array([]), np.array([])
        
        valid_x = x_coords[valid_depth_mask]
        valid_y = y_coords[valid_depth_mask]
        valid_depth = depth_values[valid_depth_mask]
        
        depth_meters = valid_depth * depth_scale
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        ppx, ppy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        
        x_3d = (valid_x - ppx) / fx * depth_meters
        y_3d = (valid_y - ppy) / fy * depth_meters
        z_3d = depth_meters
        
        self.points = np.column_stack([x_3d, y_3d, z_3d])

        if self.color_image.shape[:2] != self.depth_image.shape[:2]:
            color_image_resized = cv2.resize(self.color_image, 
                                           (self.depth_image.shape[1], self.depth_image.shape[0]))
        else:
            color_image_resized = self.color_image
        
        color_values = color_image_resized[valid_y, valid_x]
        self.colors = color_values[:, [2, 1, 0]]  # BGR -> RGB
        return True

    def transform_points_to_robot_base(self):
        points_homogeneous = np.hstack([self.points, np.ones((len(self.points), 1))])
        transform_matrix = self.robot_matrix @ self.hand_eye_matrix
        transformed_points_homogeneous = (transform_matrix @ points_homogeneous.T).T
        self.transformed_points = transformed_points_homogeneous[:, :3]
        return True

    def extract_top_height_points(self):
        # 获取Z坐标（高度）
        heights = self.transformed_points[:, 2]
        
        # 计算要保留的点数
        num_keep = max(int(len(self.transformed_points) * self.height_percentage), 1)
        
        # 获取高度最高的点的索引
        top_height_indices = np.argsort(heights)[-num_keep:]
        
        # 提取高度前percentage%的点
        top_points = self.transformed_points[top_height_indices]
        top_colors = self.colors[top_height_indices]
        
        # 计算边界框
        min_coords = top_points.min(axis=0)
        max_coords = top_points.max(axis=0)
        bbox = np.concatenate([min_coords, max_coords])  # [min_x, min_y, min_z, max_x, max_y, max_z]
        
        self.top_points = top_points
        self.top_colors = top_colors
        self.bbox = bbox
        return True

    def extract_lid_center_from_bbox(self):
        # 使用PCA分析点云的主方向
        if len(self.top_points) < 3:
            # 如果点太少，使用bbox方法
            bbox_min = self.bbox[:3]
            bbox_max = self.bbox[3:]
            bbox_center = (bbox_min + bbox_max) / 2
            bbox_size = bbox_max - bbox_min
            x_size = bbox_size[0]
            y_size = bbox_size[1]
            
            if x_size >= y_size:
                long_axis = 0
                short_axis = 1
                long_size = x_size
                short_size = y_size
            else:
                long_axis = 1
                short_axis = 0
                long_size = y_size
                short_size = x_size
        else:
            # 使用PCA分析主方向
            points_centered = self.top_points - np.mean(self.top_points, axis=0)
            cov_matrix = np.cov(points_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # 按特征值大小排序
            sorted_indices = np.argsort(eigenvalues)[::-1]
            principal_axes = eigenvectors[:, sorted_indices]
            
            # 第一主成分是长边方向
            long_axis_direction = principal_axes[:, 0]
            
            # 计算bbox
            bbox_min = self.top_points.min(axis=0)
            bbox_max = self.top_points.max(axis=0)
            bbox_center = (bbox_min + bbox_max) / 2
            bbox_size = bbox_max - bbox_min
            
            # 确定哪个世界坐标轴与长边方向最接近
            world_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            dot_products = np.abs(np.dot(long_axis_direction, world_axes.T))
            long_axis = np.argmax(dot_products)
            
            if long_axis == 0:
                short_axis = 1
                long_size = bbox_size[0]
                short_size = bbox_size[1]
            elif long_axis == 1:
                short_axis = 0
                long_size = bbox_size[1]
                short_size = bbox_size[0]
            else:  # long_axis == 2 (Z方向)
                # 如果长边是Z方向，选择X或Y中较大的作为长边
                if bbox_size[0] >= bbox_size[1]:
                    long_axis = 0
                    short_axis = 1
                    long_size = bbox_size[0]
                    short_size = bbox_size[1]
                else:
                    long_axis = 1
                    short_axis = 0
                    long_size = bbox_size[1]
                    short_size = bbox_size[0]
        if long_axis == 0:  # X方向是长边
            # 计算长边的中心位置
            long_edge_center_x = (bbox_min[0] + bbox_max[0]) / 2
            
            # 在长边中心附近选择点（Y和Z方向稍微内缩，X方向在中心附近）
            long_edge_mask = (
                (self.top_points[:, 0] >= long_edge_center_x - long_size * 0.2) &  # X方向：中心±20%范围
                (self.top_points[:, 0] <= long_edge_center_x + long_size * 0.2) &
                (self.top_points[:, 1] >= bbox_min[1] + short_size * 0.1) &  # Y方向稍微内缩
                (self.top_points[:, 1] <= bbox_max[1] - short_size * 0.1) &
                (self.top_points[:, 2] >= bbox_min[2] + bbox_size[2] * 0.1) &  # Z方向稍微内缩
                (self.top_points[:, 2] <= bbox_max[2] - bbox_size[2] * 0.1)
            )
            long_edge_points = self.top_points[long_edge_mask]
            
            if len(long_edge_points) > 0:
                # 按距离长边中心的距离排序，选择最近的num_points个点
                distances_to_center = np.abs(long_edge_points[:, 0] - long_edge_center_x)
                sorted_indices = np.argsort(distances_to_center)
                selected_points = long_edge_points[sorted_indices[:min(self.num_points, len(long_edge_points))]]
            else:
                # 如果没有找到合适的点，使用边界框中心
                selected_points = np.array([bbox_center])
        else:  # Y方向是长边
            # 计算长边的中心位置
            long_edge_center_y = (bbox_min[1] + bbox_max[1]) / 2
            
            # 在长边中心附近选择点（X和Z方向稍微内缩，Y方向在中心附近）
            long_edge_mask = (
                (self.top_points[:, 0] >= bbox_min[0] + short_size * 0.1) &  # X方向稍微内缩
                (self.top_points[:, 0] <= bbox_max[0] - short_size * 0.1) &
                (self.top_points[:, 1] >= long_edge_center_y - long_size * 0.2) &  # Y方向：中心±20%范围
                (self.top_points[:, 1] <= long_edge_center_y + long_size * 0.2) &
                (self.top_points[:, 2] >= bbox_min[2] + bbox_size[2] * 0.1) &  # Z方向稍微内缩
                (self.top_points[:, 2] <= bbox_max[2] - bbox_size[2] * 0.1)
            )
            long_edge_points = self.top_points[long_edge_mask]
            
            if len(long_edge_points) > 0:
                # 按距离长边中心的距离排序，选择最近的num_points个点
                distances_to_center = np.abs(long_edge_points[:, 1] - long_edge_center_y)
                sorted_indices = np.argsort(distances_to_center)
                selected_points = long_edge_points[sorted_indices[:min(self.num_points, len(long_edge_points))]]
            else:
                # 如果没有找到合适的点，使用边界框中心
                selected_points = np.array([bbox_center])
        
        # 计算盖子中心点
        lid_center = np.mean(selected_points, axis=0)

        # 建立局部坐标系
        # Z轴：垂直桌面向上 (0, 0, 1)
        z_axis = np.array([0, 0, 1])
        
        if len(self.top_points) >= 3:
            # 使用PCA分析得到的真实主方向
            x_axis = long_axis_direction  # 使用PCA第一主成分作为X轴
            # 确保X轴在水平面内（Z分量为0）
            x_axis[2] = 0
            x_axis = x_axis / np.linalg.norm(x_axis)  # 归一化
        else:
            # 使用bbox方法
            if long_axis == 0:  # X方向是长边
                x_axis = np.array([1, 0, 0])  # 与X轴平行
            else:  # Y方向是长边
                x_axis = np.array([0, 1, 0])  # 与Y轴平行
        
        # Y轴：垂直于长边向外（右手坐标系）
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)  # 归一化
        
        # 重新计算X轴确保正交
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)  # 归一化
        
        # 构建变换矩阵T
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = lid_center
        
        # 转换为pose [x, y, z, roll, pitch, yaw] (弧度制)
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz', degrees=False)  # 弧度制
        pose = np.concatenate([lid_center, euler_angles])
        
        return pose, T
    
    def draw_pose_axes(self, image, intrinsics, pose_matrix):
        # 将机器人基坐标系位姿转换回相机坐标系
        # 逆变换：T_camera = T_hand_eye^(-1) * T_robot^(-1) * T_base
        robot_inv = np.linalg.inv(self.robot_matrix)
        hand_eye_inv = np.linalg.inv(self.hand_eye_matrix)
        camera_pose = hand_eye_inv @ robot_inv @ pose_matrix
        
        # 提取相机坐标系下的旋转和平移
        R_camera = camera_pose[:3, :3]
        t_camera = camera_pose[:3, 3]
        
        # 转为OpenCV格式
        rvec, _ = cv2.Rodrigues(R_camera)
        tvec = t_camera.reshape(3, 1)
        
        # 检查位姿是否在合理范围内
        print(f"相机坐标系位姿位置: {t_camera.flatten()}")
        print(f"相机坐标系旋转矩阵:\n{R_camera}")
        
        # 绘制坐标轴 - 使用合适的轴长度
        axis_length = 0.05  # 5cm轴长度
        
        cv2.drawFrameAxes(image, intrinsics, np.zeros(5), rvec, tvec, axis_length)

        cv2.imshow("Pose Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image

    def detect_lid_pose(self):
        bboxes, masks = self.detect_with_yolo()
        for bbox, mask in zip(bboxes, masks):
            # 得到点云和颜色
            self.depth_to_pointcloud(mask)
            
            # 转换到基坐标系
            self.transform_points_to_robot_base()

            # 提取高度前50%的点云并用矩形框框选出来
            self.extract_top_height_points()

            if len(self.top_points) > 0:
                # 提取盖子中心点和位姿信息
                pose, T = self.extract_lid_center_from_bbox()
                print(f"盖子位姿 (弧度制): {pose}")
                print(f"变换矩阵T:\n{T}")
                
                self.draw_pose_axes(self.color_image, self.intrinsics, T)

def main():
    lid_function = LidFunction()
    lid_function.detect_lid_pose()

if __name__ == "__main__":
    main()