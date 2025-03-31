import open3d as o3d
import numpy as np
import json
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# 配置文件路径
CONFIG_FILE = "view_config.json"

# 加载配置文件
def load_config(filename=None):
    if filename is None:
        filename = CONFIG_FILE
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
                print(f"已从 {filename} 加载配置: {config}")
                return config
        except Exception as e:
            print(f"配置文件 {filename} 读取失败: {e}，使用默认配置")
    else:
        print(f"配置文件 {filename} 不存在，使用默认配置")
    
    # 默认配置
    return {
        "zoom": 0.8,
        "front": [0, 0, -1],
        "up": [0, 1, 0],
        "lookat": [0, 0, 0],
        "fov": 60.0,
        "camera_position": [0, 0, 3]
    }

# 保存配置文件
def save_config(config, filename=None):
    if filename is None:
        filename = CONFIG_FILE
    
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"视图配置已保存到 {filename}: {config}")
        return True
    except Exception as e:
        print(f"保存配置到 {filename} 失败: {e}")
        return False

# 选择配置文件
def select_config_file():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    filename = filedialog.askopenfilename(
        title="选择配置文件",
        filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
    )
    
    root.destroy()
    return filename if filename else None

# 手动设置视图参数
def manual_set_params():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 默认配置
    default_config = load_config()
    
    # 获取缩放参数
    zoom_str = simpledialog.askstring("设置缩放", "请输入缩放值:", 
                                    initialvalue=str(default_config.get("zoom", 0.8)))
    if zoom_str is None:
        return None
    try:
        zoom = float(zoom_str)
    except:
        print("缩放值格式错误，将使用默认值")
        zoom = default_config.get("zoom", 0.8)
    
    # 获取前向方向
    front_str = simpledialog.askstring("设置前向方向", "请输入前向方向 [x,y,z]:", 
                                     initialvalue=str(default_config.get("front", [0, 0, -1])))
    if front_str is None:
        return None
    try:
        # 移除方括号并分割
        front_str = front_str.strip('[]').replace(' ', '')
        front = [float(x) for x in front_str.split(',')]
        if len(front) != 3:
            raise ValueError("需要3个值")
    except:
        print("前向方向格式错误，将使用默认值")
        front = default_config.get("front", [0, 0, -1])
    
    # 获取上方向
    up_str = simpledialog.askstring("设置上方向", "请输入上方向 [x,y,z]:", 
                                  initialvalue=str(default_config.get("up", [0, 1, 0])))
    if up_str is None:
        return None
    try:
        up_str = up_str.strip('[]').replace(' ', '')
        up = [float(x) for x in up_str.split(',')]
        if len(up) != 3:
            raise ValueError("需要3个值")
    except:
        print("上方向格式错误，将使用默认值")
        up = default_config.get("up", [0, 1, 0])
    
    # 获取观察点
    lookat_str = simpledialog.askstring("设置观察点", "请输入观察点 [x,y,z]:", 
                                       initialvalue=str(default_config.get("lookat", [0, 0, 0])))
    if lookat_str is None:
        return None
    try:
        lookat_str = lookat_str.strip('[]').replace(' ', '')
        lookat = [float(x) for x in lookat_str.split(',')]
        if len(lookat) != 3:
            raise ValueError("需要3个值")
    except:
        print("观察点格式错误，将使用默认值")
        lookat = default_config.get("lookat", [0, 0, 0])
    
    root.destroy()
    
    return {
        "zoom": zoom,
        "front": front,
        "up": up,
        "lookat": lookat
    }

# 应用视图配置 - 改进的版本
def apply_view_config(view_control, config):
    try:
        # 保存当前方法参考
        methods = dir(view_control)
        print(f"可用的视图控制方法: {[m for m in methods if not m.startswith('_')]}")
        
        # 设置相机视图 - 使用多种方法尝试
        
        # 方法1: 直接设置基本参数
        if hasattr(view_control, 'set_zoom'):
            view_control.set_zoom(config.get("zoom", 0.8))
            print(f"已设置缩放: {config.get('zoom', 0.8)}")
        
        if hasattr(view_control, 'set_front'):
            view_control.set_front(config.get("front", [0, 0, -1]))
            print(f"已设置前向方向: {config.get('front', [0, 0, -1])}")
        
        if hasattr(view_control, 'set_up'):
            view_control.set_up(config.get("up", [0, 1, 0]))
            print(f"已设置上方向: {config.get('up', [0, 1, 0])}")
        
        if hasattr(view_control, 'set_lookat'):
            view_control.set_lookat(config.get("lookat", [0, 0, 0]))
            print(f"已设置观察点: {config.get('lookat', [0, 0, 0])}")
            
        # 方法2: 使用相机JSON
        if "json_string" in config:
            try:
                # 一些版本支持直接加载相机参数的JSON字符串
                if hasattr(view_control, 'convert_from_string'):
                    view_control.convert_from_string(config["json_string"])
                    print("已通过JSON字符串设置相机参数")
            except Exception as e:
                print(f"通过JSON字符串设置相机参数失败: {e}")
        
        # 方法3: 尝试使用相机外参矩阵方法
        if "extrinsic" in config and hasattr(view_control, 'convert_from_pinhole_camera_parameters'):
            try:
                params = o3d.camera.PinholeCameraParameters()
                params.extrinsic = np.array(config["extrinsic"])
                if "intrinsic" in config:
                    intr = o3d.camera.PinholeCameraIntrinsic()
                    intr.set_intrinsics(
                        width=config["intrinsic"]["width"],
                        height=config["intrinsic"]["height"],
                        fx=config["intrinsic"]["fx"],
                        fy=config["intrinsic"]["fy"],
                        cx=config["intrinsic"]["cx"],
                        cy=config["intrinsic"]["cy"]
                    )
                    params.intrinsic = intr
                view_control.convert_from_pinhole_camera_parameters(params)
                print("已通过相机参数设置视图")
            except Exception as e:
                print(f"通过相机参数设置视图失败: {e}")
        
        print("成功应用视图配置")
        return True
    except Exception as e:
        print(f"应用视图配置时出错: {e}")
        return False

def main():
    # 读取点云文件
    try:
        point_cloud = o3d.io.read_point_cloud("point_cloud.ply")
        print("成功读取点云文件 point_cloud.ply")
    except Exception as e:
        print(f"读取点云文件失败: {e}")
        print("创建一个空点云以便继续")
        point_cloud = o3d.geometry.PointCloud()
    
    # 创建可视化器
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="点云查看器 - S:保存 L:加载 M:手动设置 R:重置 V:直接保存JSON")
    
    # 添加点云到可视化器
    vis.add_geometry(point_cloud)
    
    # 使用更简单的处理方式
    def save_callback(vis):
        view_control = vis.get_view_control()
        config = {}
        
        # 尝试获取当前视图的参数
        try:
            if hasattr(view_control, 'get_zoom'):
                config["zoom"] = view_control.get_zoom()
                print(f"获取到缩放值: {config['zoom']}")
        except Exception as e:
            print(f"获取缩放失败: {e}")
            config["zoom"] = 0.8
            
        try:
            if hasattr(view_control, 'get_front'):
                config["front"] = view_control.get_front()
                print(f"获取到前向方向: {config['front']}")
        except Exception as e:
            print(f"获取前向方向失败: {e}")
            config["front"] = [0, 0, -1]
            
        try:
            if hasattr(view_control, 'get_up'):
                config["up"] = view_control.get_up()
                print(f"获取到上方向: {config['up']}")
        except Exception as e:
            print(f"获取上方向失败: {e}")
            config["up"] = [0, 1, 0]
            
        try:
            if hasattr(view_control, 'get_lookat'):
                config["lookat"] = view_control.get_lookat()
                print(f"获取到观察点: {config['lookat']}")
        except Exception as e:
            print(f"获取观察点失败: {e}")
            config["lookat"] = [0, 0, 0]
        
        # 尝试获取相机参数 - 这在某些版本的Open3D中更可靠
        try:
            if hasattr(view_control, 'convert_to_pinhole_camera_parameters'):
                params = view_control.convert_to_pinhole_camera_parameters()
                if hasattr(params, 'extrinsic'):
                    config["extrinsic"] = params.extrinsic.tolist()
                    print(f"获取到相机外参: {config['extrinsic']}")
                
                if hasattr(params, 'intrinsic'):
                    intr = params.intrinsic
                    config["intrinsic"] = {
                        "width": intr.width,
                        "height": intr.height,
                        "fx": intr.get_focal_length()[0],
                        "fy": intr.get_focal_length()[1],
                        "cx": intr.get_principal_point()[0],
                        "cy": intr.get_principal_point()[1]
                    }
                    print(f"获取到相机内参: {config['intrinsic']}")
        except Exception as e:
            print(f"获取相机参数失败: {e}")
        
        # 尝试获取JSON表示 - 仅在某些Open3D版本中有效
        try:
            if hasattr(view_control, 'convert_to_string'):
                config["json_string"] = view_control.convert_to_string()
                print("已获取相机JSON表示")
        except Exception as e:
            print(f"获取相机JSON表示失败: {e}")
        
        print("\n当前视图参数:")
        for key, value in config.items():
            if key != "json_string" and key != "extrinsic" and key != "intrinsic":
                print(f"{key}: {value}")
        
        save_config(config)
        return False
    
    def reset_callback(vis):
        default_config = {
            "zoom": 0.8,
            "front": [0, 0, -1],
            "up": [0, 1, 0],
            "lookat": [0, 0, 0]
        }
        view_control = vis.get_view_control()
        apply_view_config(view_control, default_config)
        print("\n已重置到默认视图")
        return False
    
    def load_callback(vis):
        # 打开文件对话框选择配置文件
        filename = select_config_file()
        if filename:
            config = load_config(filename)
            view_control = vis.get_view_control()
            apply_view_config(view_control, config)
            print(f"\n已加载配置: {filename}")
        return False
    
    def manual_callback(vis):
        # 手动设置视图参数
        config = manual_set_params()
        if config:
            view_control = vis.get_view_control()
            apply_view_config(view_control, config)
            print("\n已应用手动设置的视图参数")
            
            # 询问是否保存到配置文件
            root = tk.Tk()
            root.withdraw()
            if messagebox.askyesno("保存配置", "是否将当前视图参数保存到配置文件?"):
                save_config(config)
            root.destroy()
        return False
    
    def save_viewpoint_json(vis):
        """直接保存Open3D自己的相机JSON文件，可以完美恢复视角"""
        try:
            # 创建相机参数对象
            params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            
            # 打开文件对话框
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.asksaveasfilename(
                title="保存相机JSON文件",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            root.destroy()
            
            if not filename:
                return False
                
            # 直接使用Open3D内置方法保存相机参数
            o3d.io.write_pinhole_camera_parameters(filename, params)
            print(f"已将相机参数直接保存到: {filename}")
            
            return True
        except Exception as e:
            print(f"保存相机参数失败: {e}")
            return False
    
    def load_viewpoint_json(vis):
        """从Open3D自己的相机JSON文件加载视角"""
        try:
            # 打开文件对话框
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(
                title="加载相机JSON文件",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            root.destroy()
            
            if not filename:
                return False
                
            # 直接使用Open3D内置方法加载相机参数
            params = o3d.io.read_pinhole_camera_parameters(filename)
            vis.get_view_control().convert_from_pinhole_camera_parameters(params)
            print(f"已从文件加载相机参数: {filename}")
            
            return True
        except Exception as e:
            print(f"加载相机参数失败: {e}")
            return False
    
    # 添加视点JSON保存和加载功能
    vis.register_key_callback(ord('V'), lambda vis: save_viewpoint_json(vis))
    vis.register_key_callback(ord('C'), lambda vis: load_viewpoint_json(vis))
    
    # 打印Open3D版本和其他调试信息
    print(f"Open3D版本: {o3d.__version__}")
    print(f"可视化器类型: {type(vis)}")
    print(f"ViewControl类型: {type(vis.get_view_control())}")
    
    # 设置键盘回调
    vis.register_key_callback(ord('S'), save_callback)  # S键保存配置
    vis.register_key_callback(ord('R'), reset_callback)  # R键重置视图
    vis.register_key_callback(ord('L'), load_callback)  # L键加载配置
    vis.register_key_callback(ord('M'), manual_callback)  # M键手动设置参数
    
    # 加载视图配置 - 启动时默认读取view_config.json
    print("\n尝试读取默认配置文件...")
    config = load_config(CONFIG_FILE)  # 明确指定配置文件名
    view_control = vis.get_view_control()
    apply_view_config(view_control, config)
    
    # 添加坐标轴以便于参考
    try:
        vis.get_render_option().show_coordinate_frame = True
        print("已启用坐标轴显示")
    except:
        print("启用坐标轴显示失败")
    
    # 输出操作说明
    print("\n==============点云查看器操作说明==============")
    print("可以使用鼠标拖动来旋转视图")
    print("按S键保存当前视图配置到 view_config.json")
    print("按L键加载不同的视图配置")
    print("按M键手动设置视图参数")
    print("按R键重置到默认视图")
    print("按V键直接保存相机JSON文件 (更可靠)")
    print("按C键从相机JSON文件加载视图 (更可靠)")
    print("=============================================")
    
    # 渲染点云
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main() 