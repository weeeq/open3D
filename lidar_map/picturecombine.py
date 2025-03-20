import os
import open3d as o3d
from PIL import Image

def convert_ply_to_image(ply_file, image_file):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 创建一个可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    
    # 渲染并捕获图像
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(image_file)
    vis.destroy_window()

def overlay_images_from_directory(directory_path, output_path):
    # 获取目录下的所有PLY文件
    ply_files = [f for f in os.listdir(directory_path) if f.endswith('.ply')]
    ply_files.sort()  # 确保文件按顺序排列

    image_files = []
    for ply_file in ply_files:
        image_file = os.path.join(directory_path, ply_file.replace('.ply', '.png'))
        convert_ply_to_image(os.path.join(directory_path, ply_file), image_file)
        image_files.append(image_file)

    # 打开所有图像
    base_image = Image.open(image_files[0]).convert("RGBA")
    
    for image_file in image_files[1:]:
        overlay_image = Image.open(image_file).convert("RGBA")
        base_image = Image.alpha_composite(base_image, overlay_image)
    
    # 保存叠加后的图像
    base_image.save(output_path)

# 示例用法
directory_path = 'output_point_clouds'
output_file = 'output_point_clouds/overlayed_image.png'
overlay_images_from_directory(directory_path, output_file)