import os
import json
import numpy as np
import open3d as o3d

def reconstruct_point_cloud_from_json_files(input_directory, output_ply_file, output_png_file):
    # 初始化一个空的点列表
    all_points = []

    # 遍历输入目录中的所有JSON文件
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(input_directory, filename)
            
            # 读取JSON文件
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # 提取点云数据
            point_step = data['point_step']
            data_bytes = data['data']
            
            # 将字节数据转换为浮点数并添加到all_points
            for i in range(0, len(data_bytes), point_step):
                x = np.frombuffer(bytearray(data_bytes[i:i+4]), dtype=np.float32)[0]
                y = np.frombuffer(bytearray(data_bytes[i+4:i+8]), dtype=np.float32)[0]
                z = np.frombuffer(bytearray(data_bytes[i+8:i+12]), dtype=np.float32)[0]
                all_points.append([x, y, z])
    
    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(all_points))
    
    # 保存点云到PLY文件
    o3d.io.write_point_cloud(output_ply_file, point_cloud)
    
    # 可视化点云并保存为PNG文件
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_png_file)
    vis.destroy_window()

# 使用函数处理所有JSON文件并保存为PLY和PNG文件
reconstruct_point_cloud_from_json_files('output_json_frames', 'output_point_cloud.ply', 'output_point_cloud.png')