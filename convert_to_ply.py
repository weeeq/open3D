import numpy as np
import re

def read_point_cloud(file_path):
    # 读取点云数据
    with open(file_path, 'r') as file:
        data = file.read()
    
    # 使用正则表达式提取所有数字
    numbers = re.findall(r'-?\d+\.?\d*', data)
    
    # 将数据转换为浮点数数组
    points = np.array(numbers, dtype=np.float32)
    
    # 检查数据是否可以被3整除
    if len(points) % 3 != 0:
        print(f"警告：数据点数量 {len(points)} 不能被3整除，忽略最后 {len(points) % 3} 个数据点。")
        points = points[:-(len(points) % 3)]
    
    # 将数据重塑为Nx3的矩阵（假设每三个数代表一个点的坐标）
    num_points = len(points) // 3
    points = points.reshape(num_points, 3)
    
    return points

def write_ply(file_path, points):
    # 写入PLY文件
    with open(file_path, 'w') as file:
        # 写入PLY文件头
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        
        # 写入点云数据
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

def main():
    # 输入和输出文件路径
    input_file = 'point_cloud(1).txt'
    output_file = 'point_cloud.ply'
    
    # 读取点云数据
    points = read_point_cloud(input_file)
    
    # 写入PLY文件
    write_ply(output_file, points)
    
    print(f"PLY文件已成功保存为 {output_file}")

if __name__ == "__main__":
    main() 