#!/usr/bin/env python3
"""
简单的 MapAnything 纯图片推理脚本
用于 road100 文件夹
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import numpy as np
from PIL import Image
import cv2
import trimesh
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.colmap_export import export_predictions_to_colmap

def main():
    # 设置路径
    image_folder = "/root/autodl-tmp/data/images/road100"
    output_dir = "/root/autodl-tmp/data/images/road100/outputs"
    output_glb_path = os.path.join(output_dir, "reconstruction.glb")
    depth_dir = os.path.join(output_dir, "depth_maps")
    pointcloud_path = os.path.join(output_dir, "merged_pointcloud.ply")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # 获取设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 初始化模型
    # 如果需要使用 Apache 2.0 模型，改为 "facebook/map-anything-apache"
    model_name = "facebook/map-anything"
    print(f"正在加载模型: {model_name}")
    print("注意: 如果模型未下载，需要网络连接 HuggingFace")
    
    try:
        model = MapAnything.from_pretrained(model_name).to(device)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 如果模型已下载到本地缓存，确保 HuggingFace 缓存路径正确")
        print("3. 或者使用本地模型权重（需要先下载模型）")
        return
    
    # 加载图片
    print(f"\n正在从文件夹加载图片: {image_folder}")
    views = load_images(image_folder)
    print(f"成功加载 {len(views)} 张图片")
    
    if len(views) == 0:
        print("错误: 未找到任何图片!")
        return
    
    # 运行推理
    print("\n开始推理...")
    predictions = model.infer(
        views,                            # 输入视图
        memory_efficient_inference=True,  # 内存高效模式，支持更多视图（最多2000张在140GB GPU上）
        minibatch_size=None,              # 动态计算批次大小（根据可用GPU内存自动调整）。使用1可获得最小GPU内存消耗
        use_amp=True,                     # 使用混合精度推理（推荐）
        amp_dtype="bf16",                 # bf16 推理（推荐；如果不支持会自动降级到 fp16）
        apply_mask=True,                  # 对密集几何输出应用掩码
        mask_edges=True,                  # 使用法线和深度移除边缘伪影
        apply_confidence_mask=False,      # 过滤低置信度区域
        confidence_percentile=10,         # 移除底部10百分位的置信度像素
        use_multiview_confidence=False,   # 启用基于多视图深度一致性的置信度（替代基于学习的置信度）
    )
    print("推理完成!")
    
    # 显示结果信息 - 展示所有可用的输出字段（与 README 示例一致）
    print("\n推理结果:")
    for i, pred in enumerate(predictions):
        print(f"\n视图 {i}:")
        # 几何输出
        pts3d = pred["pts3d"]                     # 世界坐标系中的 3D 点 (B, H, W, 3)
        pts3d_cam = pred["pts3d_cam"]             # 相机坐标系中的 3D 点 (B, H, W, 3)
        depth_z = pred["depth_z"]                 # 相机坐标系中的 Z 深度 (B, H, W, 1)
        depth_along_ray = pred["depth_along_ray"] # 相机坐标系中沿射线的深度 (B, H, W, 1)
        
        # 相机输出
        ray_directions = pred["ray_directions"]   # 相机坐标系中的射线方向 (B, H, W, 3)
        intrinsics = pred["intrinsics"]           # 恢复的针孔相机内参 (B, 3, 3)
        camera_poses = pred["camera_poses"]       # OpenCV (+X - Right, +Y - Down, +Z - Forward) 世界坐标系中的 cam2world 位姿 (B, 4, 4)
        cam_trans = pred["cam_trans"]             # OpenCV cam2world 平移 (B, 3)
        cam_quats = pred["cam_quats"]             # OpenCV cam2world 四元数 (B, 4)
        
        # 质量和掩码
        confidence = pred["conf"]                 # 每像素置信度分数 (B, H, W)
        mask = pred["mask"]                       # 组合的有效性掩码 (B, H, W, 1)
        non_ambiguous_mask = pred["non_ambiguous_mask"]                # 非模糊区域 (B, H, W)
        non_ambiguous_mask_logits = pred["non_ambiguous_mask_logits"]  # 掩码 logits (B, H, W)
        
        # 缩放
        metric_scaling_factor = pred["metric_scaling_factor"]  # 应用的度量缩放 (B,)
        
        # 原始输入
        img_no_norm = pred["img_no_norm"]         # 用于可视化的去归一化输入图像 (B, H, W, 3)
        
        # 打印形状信息
        print(f"  - 3D点云(世界): {pts3d.shape}")
        print(f"  - 3D点云(相机): {pts3d_cam.shape}")
        print(f"  - Z深度: {depth_z.shape}")
        print(f"  - 沿射线深度: {depth_along_ray.shape}")
        print(f"  - 射线方向: {ray_directions.shape}")
        print(f"  - 相机内参: {intrinsics.shape}")
        print(f"  - 相机位姿: {camera_poses.shape}")
        print(f"  - 相机平移: {cam_trans.shape}")
        print(f"  - 相机四元数: {cam_quats.shape}")
        print(f"  - 置信度: {confidence.shape}")
        print(f"  - 掩码: {mask.shape}")
        print(f"  - 非模糊掩码: {non_ambiguous_mask.shape}")
        print(f"  - 度量缩放因子: {metric_scaling_factor.shape}")
    
    # 导入几何处理函数
    from mapanything.utils.geometry import depthmap_to_world_frame
    
    # 保存选项
    save_glb = True           # 保存 GLB 文件
    save_depth_maps = True    # 保存每个视角的深度图
    save_pointcloud = True    # 保存合并后的点云
    save_colmap = True        # 保存 COLMAP 格式（稀疏重建格式）
    
    # 获取原始图片文件名（用于命名输出文件）
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) + 
                        glob.glob(os.path.join(image_folder, "*.png")))
    image_names = [os.path.basename(f) for f in image_files]
    
    # 保存每个视角的深度图
    if save_depth_maps:
        print(f"\n正在保存深度图到: {depth_dir}")
        try:
            for i, pred in enumerate(predictions):
                depth_z = pred["depth_z"][0].squeeze(-1).cpu().numpy()  # (H, W)
                mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)
                
                # 获取基础文件名（不含扩展名）
                base_name = os.path.splitext(image_names[i] if i < len(image_names) else f"view_{i:04d}")[0]
                
                # 1. 保存为 EXR 格式（原始深度值，高精度）
                depth_exr_path = os.path.join(depth_dir, f"{base_name}_depth.exr")
                # 使用 OpenCV 保存 EXR（16位半精度）
                cv2.imwrite(depth_exr_path, depth_z, [
                    cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
                    cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP
                ])
                
                # 2. 保存为 PNG 格式（可视化，归一化到 0-255）
                depth_vis = depth_z.copy()
                # 应用掩码，无效区域设为 0
                depth_vis[~mask] = 0
                
                # 归一化到 0-255 用于可视化
                if depth_vis[mask].size > 0:
                    depth_min = depth_vis[mask].min()
                    depth_max = depth_vis[mask].max()
                    if depth_max > depth_min:
                        depth_vis_norm = ((depth_vis - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                    else:
                        depth_vis_norm = np.zeros_like(depth_vis, dtype=np.uint8)
                else:
                    depth_vis_norm = np.zeros_like(depth_vis, dtype=np.uint8)
                
                # 保存为彩色深度图（使用 colormap）
                depth_colored = cv2.applyColorMap(depth_vis_norm, cv2.COLORMAP_VIRIDIS)
                depth_png_path = os.path.join(depth_dir, f"{base_name}_depth.png")
                cv2.imwrite(depth_png_path, depth_colored)
                
                # 3. 保存为 numpy 格式（方便后续处理）
                depth_npy_path = os.path.join(depth_dir, f"{base_name}_depth.npy")
                np.save(depth_npy_path, depth_z)
                
                if (i + 1) % 10 == 0:
                    print(f"  已保存 {i + 1}/{len(predictions)} 个深度图...")
            
            print(f"成功保存 {len(predictions)} 个深度图到: {depth_dir}")
            print(f"  - EXR 格式: 原始深度值（高精度）")
            print(f"  - PNG 格式: 可视化深度图（彩色）")
            print(f"  - NPY 格式: NumPy 数组（方便处理）")
        except Exception as e:
            print(f"保存深度图失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 收集所有视角的点云数据（用于合并）
    all_world_points = []
    all_colors = []
    
    # 保存 GLB 文件和准备点云数据
    if save_glb or save_pointcloud:
        print(f"\n正在处理点云数据...")
        try:
            from mapanything.utils.viz import predictions_to_glb
            
            world_points_list = []
            images_list = []
            masks_list = []
            
            for pred in predictions:
                depthmap_torch = pred["depth_z"][0].squeeze(-1)
                intrinsics_torch = pred["intrinsics"][0]
                camera_pose_torch = pred["camera_poses"][0]
                
                pts3d_computed, valid_mask = depthmap_to_world_frame(
                    depthmap_torch, intrinsics_torch, camera_pose_torch
                )
                
                mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
                mask = mask & valid_mask.cpu().numpy()
                pts3d_np = pts3d_computed.cpu().numpy()
                image_np = pred["img_no_norm"][0].cpu().numpy()
                
                world_points_list.append(pts3d_np)
                images_list.append(image_np)
                masks_list.append(mask)
                
                # 收集点云数据用于合并
                if save_pointcloud:
                    # 提取有效点
                    valid_pts = pts3d_np[mask]  # (N, 3)
                    valid_colors = (image_np[mask] * 255).astype(np.uint8)  # (N, 3) [0-255]
                    
                    all_world_points.append(valid_pts)
                    all_colors.append(valid_colors)
            
            # 保存 GLB 文件
            if save_glb:
                print(f"\n正在保存 GLB 文件到: {output_glb_path}")
                # 堆叠所有视图
                world_points = np.stack(world_points_list, axis=0)
                images = np.stack(images_list, axis=0)
                final_masks = np.stack(masks_list, axis=0)
                
                # 创建预测字典
                predictions_dict = {
                    "world_points": world_points,
                    "images": images,
                    "final_masks": final_masks,
                }
                
                # 转换为 GLB 场景
                scene_3d = predictions_to_glb(predictions_dict, as_mesh=True)
                
                # 保存 GLB 文件
                scene_3d.export(output_glb_path)
                print(f"成功保存 GLB 文件: {output_glb_path}")
            
            # 保存合并后的点云
            if save_pointcloud:
                print(f"\n正在保存合并后的点云到: {pointcloud_path}")
                # 合并所有视角的点云
                merged_points = np.concatenate(all_world_points, axis=0)  # (N_total, 3)
                merged_colors = np.concatenate(all_colors, axis=0)  # (N_total, 3)
                
                print(f"  合并后的点云包含 {len(merged_points):,} 个点")
                
                # 创建 trimesh 点云对象并保存为 PLY
                pointcloud = trimesh.PointCloud(vertices=merged_points, colors=merged_colors)
                pointcloud.export(pointcloud_path)
                print(f"成功保存点云到: {pointcloud_path}")
                print(f"  点云格式: PLY (二进制)")
                print(f"  点数量: {len(merged_points):,}")
                print(f"  颜色: RGB [0-255]")
                
        except Exception as e:
            print(f"保存点云/GLB 文件失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 导出 COLMAP 格式（稀疏重建格式）
    if save_colmap:
        print(f"\n正在导出 COLMAP 格式到: {output_dir}")
        try:
            colmap_output_dir = os.path.join(output_dir, "colmap_output")
            os.makedirs(colmap_output_dir, exist_ok=True)
            
            # 获取原始图片文件名
            image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) + 
                                glob.glob(os.path.join(image_folder, "*.png")))
            image_names = [os.path.basename(f) for f in image_files]
            
            # 导出为 COLMAP 格式
            # 注意：导出的是稀疏重建格式（经过体素下采样的密集重建结果）
            export_predictions_to_colmap(
                outputs=predictions,
                processed_views=views,
                image_names=image_names,
                output_dir=colmap_output_dir,
                voxel_fraction=0.01,  # 体素大小为场景范围的1%（默认值）
                voxel_size=None,      # 可以指定明确的体素大小（米），会覆盖 voxel_fraction
                data_norm_type=model.encoder.data_norm_type,
                save_ply=True,        # 保存 PLY 点云文件
                save_images=True,     # 保存处理后的图像
                skip_point2d=False,   # 是否跳过 Point2D 反投影（False=完整导出，True=更快但某些工具可能需要Point2D）
            )
            
            print(f"成功导出 COLMAP 格式到: {colmap_output_dir}")
            print(f"  - sparse/ 目录: 包含 cameras.bin, images.bin, points3D.bin, points.ply")
            print(f"  - images/ 目录: 包含处理后的图像（匹配模型推理分辨率）")
            print(f"  注意: 导出的是稀疏重建格式，但数据来源是密集重建结果经过体素下采样得到的")
        except Exception as e:
            print(f"导出 COLMAP 格式失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n完成!")

if __name__ == "__main__":
    main()
