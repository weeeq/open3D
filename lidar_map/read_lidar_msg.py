import msgpack
import numpy as np
from typing import List, Dict, Tuple
import open3d as o3d
import os
import json
import matplotlib.pyplot as plt

def analyze_point_cloud(points: np.ndarray) -> Dict:
    """
    分析点云数据的基本特征
    Args:
        points: 点云数据数组
    Returns:
        stats: 包含点云统计信息的字典
    """
    stats = {
        "点数量": len(points),
        "坐标范围": {
            "X": (np.min(points[:, 0]), np.max(points[:, 0])),
            "Y": (np.min(points[:, 1]), np.max(points[:, 1])),
            "Z": (np.min(points[:, 2]), np.max(points[:, 2])),
        },
        "密度": len(points) / (np.max(points[:, 0]) - np.min(points[:, 0])) / (np.max(points[:, 1]) - np.min(points[:, 1]))
    }
    return stats

def visualize_point_cloud(points: np.ndarray) -> None:
    """
    使用Open3D可视化点云
    Args:
        points: 点云数据数组
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] > 3:  # 如果有强度信息
        colors = np.zeros((len(points), 3))
        colors[:, 0] = points[:, 3] / np.max(points[:, 3])  # 使用强度值作为红色通道
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def reconstruct_surface(points: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    从点云重建表面
    Args:
        points: 点云数据数组
    Returns:
        mesh: 重建的三角网格
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # 估计法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 使用泊松重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    return mesh

def parse_point_cloud_data(data: List) -> np.ndarray:
    """
    解析原始点云数据列表为numpy数组
    Args:
        data: 原始数据列表
    Returns:
        points: 解析后的点云数组
    """
    try:
        # 打印数据长度和部分数据用于调试
        print(f"原始数据长度: {len(data)}")
        print(f"原始数据示例: {data[:10]}")  # 打印前10个数据用于检查

        # 验证数据长度是否为4的倍数
        if len(data) % 4 != 0:
            print("数据长度不是4的倍数，可能数据不完整")
            return np.array([])

        # 将一维列表重塑为N×4的数组（x,y,z,intensity）
        points = np.array(data).reshape(-1, 4)
        return points
    except Exception as e:
        print(f"解析点云数据时发生错误: {str(e)}")
        return np.array([])

def save_frame_as_json(frame_data: Dict, frame_number: int, output_dir: str) -> None:
    """
    将单个数据包保存为 JSON 文件
    Args:
        frame_data: 数据包字典
        frame_number: 数据包编号
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"frame_{frame_number:04d}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(frame_data, f, ensure_ascii=False, indent=4)
        print(f"数据包 {frame_number} 已保存到 {file_path}")
    except Exception as e:
        print(f"保存数据包 {frame_number} 时发生错误: {str(e)}")

def read_lidar_msg(file_path: str, output_dir: str) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    读取.msg格式的LiDAR点云数据，并将每个数据包保存为JSON文件
    Args:
        file_path: msg文件路径
        output_dir: 输出目录
    Returns:
        points_list: 包含所有点云帧的列表
        metadata_list: 包含所有帧元数据的列表
    """
    points_list = []
    metadata_list = []
    
    try:
        with open(file_path, "rb") as f:
            # 读取整个文件内容
            data = f.read()
            
            # 使用msgpack的Unpacker来处理多个数据包
            unpacker = msgpack.Unpacker(raw=False)
            unpacker.feed(data)
            
            # 记录数据包数量
            frame_count = 0
            
            # 遍历所有数据包
            for frame_data in unpacker:
                frame_count += 1
                try:
                    if isinstance(frame_data, dict):
                        # 打印每个数据包的内容用于调试
                        print(f"数据包 {frame_count} 内容: {frame_data}")

                        # 保存数据包为 JSON 文件
                        save_frame_as_json(frame_data, frame_count, output_dir)

                        # 检查数据包是否包含'points'键
                        if 'points' not in frame_data:
                            print(f"数据包 {frame_count} 缺少'points'键，跳过此帧")
                            continue

                        # 提取点云数据
                        points = parse_point_cloud_data(frame_data['points'])
                        if len(points) > 0:
                            print(f"数据包 {frame_count} 成功解析出 {len(points)} 个点")
                            points_list.append(points)
                        else:
                            print(f"数据包 {frame_count} 解析点云数据失败，跳过此帧")
                        
                        # 提取元数据
                        metadata = {k: v for k, v in frame_data.items() if k != 'points'}
                        metadata_list.append(metadata)
                        print(f"数据包 {frame_count} 元数据信息:", metadata)
                
                except Exception as e:
                    print(f"处理数据包 {frame_count} 时发生错误: {str(e)}")
                    continue
            
            print(f"总共解析出 {frame_count} 个数据包")
            print(f"成功读取到 {len(points_list)} 帧点云数据")
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
    
    return points_list, metadata_list

def save_point_cloud(points: np.ndarray, save_path: str, format: str = 'pcd') -> bool:
    """
    保存点云数据到文件
    Args:
        points: 点云数据数组 (N,4) - x,y,z,intensity
        save_path: 保存路径（不包含扩展名）
        format: 保存格式 ('pcd', 'ply', 'xyz', 'pts')
    Returns:
        bool: 保存是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 只使用xyz坐标
        
        # 将强度值归一化并转换为颜色
        if points.shape[1] > 3:  # 如果有强度信息
            intensities = points[:, 3]
            colors = np.zeros((len(points), 3))
            # 将强度值归一化到[0,1]范围
            normalized_intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            # 使用灰度值作为颜色
            colors[:, :] = normalized_intensities.reshape(-1, 1)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        full_path = f"{save_path}.{format}"
        if format == 'pcd':
            o3d.io.write_point_cloud(full_path, pcd, write_ascii=False)
        elif format == 'ply':
            o3d.io.write_point_cloud(full_path, pcd)
        elif format in ['xyz', 'pts']:
            # 保存为带强度值的文本格式
            np.savetxt(full_path, points, fmt='%.6f')
        
        print(f"点云数据已保存到: {full_path}")
        return True
    except Exception as e:
        print(f"保存点云数据时发生错误: {str(e)}")
        return False

def save_mesh(mesh: o3d.geometry.TriangleMesh, save_path: str, format: str = 'ply') -> bool:
    """
    保存重建的网格模型
    Args:
        mesh: 三角网格模型
        save_path: 保存路径（不包含扩展名）
        format: 保存格式 ('ply', 'obj', 'stl')
    Returns:
        bool: 保存是否成功
    """
    try:
        full_path = f"{save_path}.{format}"
        if format == 'ply':
            o3d.io.write_triangle_mesh(full_path, mesh)
        elif format == 'obj':
            o3d.io.write_triangle_mesh(full_path, mesh)
        elif format == 'stl':
            o3d.io.write_triangle_mesh(full_path, mesh)
        
        print(f"网格模型已保存到: {full_path}")
        return True
    except Exception as e:
        print(f"保存网格模型时发生错误: {str(e)}")
        return False

def save_all_frames(point_clouds: List[np.ndarray], output_dir: str) -> None:
    """
    保存所有帧的点云数据
    Args:
        point_clouds: 点云数据列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每一帧的点云数据
    for i, points in enumerate(point_clouds):
        frame_name = os.path.join(output_dir, f"frame_{i:04d}")
        # 保存为不同格式
        save_point_cloud(points, frame_name, 'pcd')  # PCD格式
        save_point_cloud(points, frame_name, 'ply')  # PLY格式

def process_json_files(input_dir: str, output_dir: str) -> None:
    """
    批量处理JSON文件并将点云数据保存为图片
    Args:
        input_dir: JSON文件所在目录
        output_dir: 图片保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                frame_data = json.load(f)
            
            if 'data' in frame_data:
                # 解析点云数据
                data = frame_data['data']
                points = np.array(data).reshape(-1, 4)  # 假设数据是以x,y,z,intensity存储
                
                # 可视化并保存为图片
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], cmap='viridis', marker='o')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
                # 保存图片
                image_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
                plt.savefig(image_path)
                plt.close(fig)
                print(f"图片已保存到: {image_path}")

if __name__ == "__main__":
    # 使用示例
    file_path = "go2demodata/go2demodata/utlidar_cloud_deskewed.msg"
    output_dir = "output_json_frames"
    
    # 读取数据并保存每个数据包为 JSON 文件
    read_lidar_msg(file_path, output_dir)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    point_clouds, metadata = read_lidar_msg(file_path, output_dir)
    
    print(f"\n总共读取到 {len(point_clouds)} 帧点云数据")
    
    if point_clouds:
        # 获取第一帧数据
        first_frame = point_clouds[0]
        print(f"第一帧点云包含 {len(first_frame)} 个点")
        
        # 保存为不同格式
        frame_path = os.path.join(output_dir, "lidar_frame")
        save_point_cloud(first_frame, frame_path, 'pcd')
        save_point_cloud(first_frame, frame_path, 'ply')
        save_point_cloud(first_frame, frame_path, 'xyz')
        
        # 显示点云数据的基本信息
        print("\n点云数据统计信息:")
        print(f"坐标范围:")
        print(f"X: [{np.min(first_frame[:, 0]):.2f}, {np.max(first_frame[:, 0]):.2f}]")
        print(f"Y: [{np.min(first_frame[:, 1]):.2f}, {np.max(first_frame[:, 1]):.2f}]")
        print(f"Z: [{np.min(first_frame[:, 2]):.2f}, {np.max(first_frame[:, 2]):.2f}]")
        if first_frame.shape[1] > 3:
            print(f"强度值范围: [{np.min(first_frame[:, 3]):.2f}, {np.max(first_frame[:, 3]):.2f}]")
        
        # 可视化点云
        print("\n正在打开可视化窗口...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(first_frame[:, :3])
        if first_frame.shape[1] > 3:
            colors = np.zeros((len(first_frame), 3))
            normalized_intensities = (first_frame[:, 3] - np.min(first_frame[:, 3])) / (np.max(first_frame[:, 3]) - np.min(first_frame[:, 3]))
            colors[:, :] = normalized_intensities.reshape(-1, 1)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
        
        # 分析和显示第一帧数据
        print("\n第一帧数据分析:")
        stats = analyze_point_cloud(first_frame)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 重建并保存网格模型
        print("\n正在进行表面重建...")
        mesh = reconstruct_surface(first_frame)
        mesh_path = os.path.join(output_dir, "reconstructed_mesh")
        save_mesh(mesh, mesh_path, 'ply')
        save_mesh(mesh, mesh_path, 'obj')
        save_mesh(mesh, mesh_path, 'stl')
        
        # 可视化
        print("\n正在打开可视化窗口...")
        visualize_point_cloud(first_frame)
        o3d.visualization.draw_geometries([mesh])
        
        # 批量处理JSON文件并保存为图片
        json_input_dir = "output_json_frames"
        image_output_dir = "output_images"
        process_json_files(json_input_dir, image_output_dir) 