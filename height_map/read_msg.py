import msgpack
from pprint import pprint
import numpy as np
import cv2
import h5py
import os

def read_msg_file(file_name):
    """
    读取 MessagePack 文件并返回解包后的数据
    
    Args:
        file_name (str): 文件路径，包括文件名
        
    Returns:
        list: 包含所有解包数据的列表
    """
    try:
        results = []
        # 以二进制方式读取文件
        with open(file_name, "rb") as f:
            # 使用流式解包
            unpacker = msgpack.Unpacker(f, raw=False)
            for unpacked_dict in unpacker:
                results.append(unpacked_dict)
        return results
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_name}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return None

def analyze_data(data):
    """
    分析并打印数据的详细信息
    
    Args:
        data: 解包后的数据
    """
    print("\n=== 数据分析 ===")
    
    if not data:
        print("没有数据可供分析")
        return
        
    print(f"\n1. 数据类型: {type(data)}")
    print(f"2. 数据长度: {len(data)} 条记录")
    
    if len(data) > 0:
        print("\n3. 第一条记录详细信息:")
        print(f"   类型: {type(data[0])}")
        if isinstance(data[0], dict):
            print("\n   键值对:")
            for key, value in data[0].items():
                print(f"   - {key}:")
                print(f"     类型: {type(value)}")
                if isinstance(value, (list, np.ndarray)):
                    print(f"     长度: {len(value)}")
                    print(f"     前几个元素: {value[:5] if len(value) > 5 else value}")
                else:
                    print(f"     值: {value}")
        
        print("\n4. 数据示例 (前两条记录):")
        pprint(data[:2], indent=3, depth=2)

def save_as_images(data, output_dir):
    """
    将高度图数据保存为图片格式
    
    Args:
        data: MessagePack解包后的数据
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "height_maps"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
    
    print(f"\n开始保存图片到: {output_dir}")
    
    for i, frame in enumerate(data):
        # 将一维数组重塑为128x128的矩阵
        height_map = np.array(frame['data']).reshape(frame['height'], frame['width'])
        
        # 归一化到0-255范围
        height_map_norm = np.where(height_map >= 1e9, 0, height_map)  # 处理无效值
        height_map_norm = cv2.normalize(height_map_norm, None, 0, 255, cv2.NORM_MINMAX)
        height_map_norm = height_map_norm.astype(np.uint8)
        
        # 应用颜色映射以便更好地可视化
        height_map_color = cv2.applyColorMap(height_map_norm, cv2.COLORMAP_JET)
        
        # 保存原始灰度图和彩色图
        gray_path = os.path.join(output_dir, "height_maps", f"height_map_{i:03d}_gray.png")
        color_path = os.path.join(output_dir, "height_maps", f"height_map_{i:03d}_color.png")
        
        cv2.imwrite(gray_path, height_map_norm)
        cv2.imwrite(color_path, height_map_color)
        
        # 保存元数据
        metadata = {
            'stamp': frame['stamp'],
            'resolution': frame['resolution'],
            'origin': frame['origin']
        }
        metadata_path = os.path.join(output_dir, "metadata", f"metadata_{i:03d}.npy")
        np.save(metadata_path, metadata)
        
        if i % 10 == 0:  # 每10帧打印一次进度
            print(f"已处理: {i+1}/{len(data)} 帧")
    
    print(f"保存完成！共保存了 {len(data)} 帧图像")

def save_as_numpy(data, output_file):
    """
    将数据保存为单个NumPy文件
    """
    processed_data = {
        'height_maps': [],
        'timestamps': [],
        'resolution': data[0]['resolution'],
        'origins': []
    }
    
    for frame in data:
        height_map = np.array(frame['data']).reshape(frame['height'], frame['width'])
        processed_data['height_maps'].append(height_map)
        processed_data['timestamps'].append(frame['stamp'])
        processed_data['origins'].append(frame['origin'])
    
    np.savez_compressed(output_file, **processed_data)

def save_as_hdf5(data, output_file):
    """
    将数据保存为HDF5格式
    """
    with h5py.File(output_file, 'w') as f:
        # 创建数据集
        height_maps = np.array([np.array(frame['data']).reshape(frame['height'], frame['width']) 
                              for frame in data])
        timestamps = np.array([frame['stamp'] for frame in data])
        origins = np.array([frame['origin'] for frame in data])
        
        # 保存数据集
        f.create_dataset('height_maps', data=height_maps, compression='gzip')
        f.create_dataset('timestamps', data=timestamps)
        f.create_dataset('origins', data=origins)
        f.attrs['resolution'] = data[0]['resolution']

# 使用示例
if __name__ == "__main__":
    file_path = "D:\code\open3D\go2demodata\go2demodata\height_map_array.msg"
    data = read_msg_file(file_path)
    
    if data:
        # 保存为图片
        save_as_images(data, "height_map_output")
        
        # 或保存为NumPy文件
        save_as_numpy(data, "height_maps.npz")
        
        # 或保存为HDF5文件
        save_as_hdf5(data, "height_maps.h5") 