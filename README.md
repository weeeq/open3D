# 激光雷达点云处理项目

## 项目概述

本项目用于处理激光雷达(LiDAR)点云数据，实现点云重建、可视化和图像生成功能。通过一系列Python脚本，可以将原始激光雷达消息数据转换为JSON格式，重建三维点云，并生成可视化图像。

## 目录结构

- `go2demodata`：存放原始激光雷达数据文件(.msg格式)
- `height_map`：用于生成高度图的模块
- `lidar_map`：包含处理激光雷达数据的核心脚本
  - `read_lidar_msg.py`：读取并解析原始激光雷达数据
  - `jsonprocess.py`：从JSON文件重建点云
  - `picturecombine.py`：合并点云图像
- `output_json_frames`：存储从原始数据中提取的JSON格式帧数据
- `output_point_clouds`：存储生成的点云文件和可视化图像

## 使用方法

### 1. 处理原始激光雷达数据

首先将原始激光雷达数据(.msg文件)放入`go2demodata`目录，然后运行： 