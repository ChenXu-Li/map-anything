#!/usr/bin/env python3
"""
使用 MapAnything 优化 COLMAP 数据集

该脚本读取一个已存在的 COLMAP 数据集（包含图片和相机位姿），
使用 MapAnything 进行多模态推理（图片+位姿+内参），
输出新的 COLMAP 格式数据集，具有更好的稀疏点云。

输入要求：
    colmap_input_path/
      images/
        img1.jpg
        img2.jpg
        ...
      sparse/
        cameras.bin/txt
        images.bin/txt
        points3D.bin/txt

输出结构：
    colmap_output_path/
      images/           # 处理后的图像（匹配模型推理分辨率）
        img1.jpg
        img2.jpg
        ...
      sparse/           # 新的 COLMAP 稀疏重建（更好的点云）
        cameras.bin
        images.bin
        points3D.bin
        points.ply

使用方法:
    python refine_colmap_with_mapanything.py \
        --input_colmap_path /path/to/input/colmap \
        --output_colmap_path /path/to/output/colmap \
        [--stride 1] \
        [--voxel_fraction 0.01] \
        [--apache]
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import torch
from mapanything.models import MapAnything
from mapanything.utils.colmap import read_model, get_camera_matrix, qvec2rotmat
from mapanything.utils.colmap_export import export_predictions_to_colmap
from mapanything.utils.geometry import closed_form_pose_inverse
from mapanything.utils.image import preprocess_inputs
from PIL import Image
import numpy as np


def load_colmap_data_for_refinement(colmap_path, stride=1, verbose=False, ext=".bin"):
    """
    加载 COLMAP 格式数据用于 MapAnything 多模态推理。

    期望的文件夹结构:
    colmap_path/
      images/
        img1.jpg
        img2.jpg
        ...
      sparse/
        cameras.bin/txt
        images.bin/txt
        points3D.bin/txt

    Args:
        colmap_path (str): 包含 images/ 和 sparse/ 子文件夹的主文件夹路径
        stride (int): 每隔第几张图片加载一次（默认: 1，即加载所有图片）
        verbose (bool): 是否打印详细信息
        ext (str): COLMAP 文件扩展名 (".bin" 或 ".txt")

    Returns:
        tuple: (views_list, image_names_list)
            - views_list: MapAnything 推理所需的视图字典列表
            - image_names_list: 按顺序排列的图像文件名列表
    """
    # 定义路径
    images_folder = os.path.join(colmap_path, "images")
    sparse_folder = os.path.join(colmap_path, "sparse")

    # 检查必需的文件夹是否存在
    if not os.path.exists(images_folder):
        raise ValueError(f"必需的 'images' 文件夹未找到: {images_folder}")
    if not os.path.exists(sparse_folder):
        raise ValueError(f"必需的 'sparse' 文件夹未找到: {sparse_folder}")

    if verbose:
        print(f"从以下路径加载 COLMAP 数据: {colmap_path}")
        print(f"图像文件夹: {images_folder}")
        print(f"稀疏重建文件夹: {sparse_folder}")
        print(f"使用 COLMAP 文件扩展名: {ext}")

    # 读取 COLMAP 模型
    try:
        cameras, images_colmap, points3D = read_model(sparse_folder, ext=ext)
    except Exception as e:
        raise ValueError(f"从 {sparse_folder} 读取 COLMAP 模型失败: {e}")

    if verbose:
        print(
            f"加载的 COLMAP 模型包含 {len(cameras)} 个相机, {len(images_colmap)} 张图像, {len(points3D)} 个 3D 点"
        )

    # 获取可用图像文件列表
    available_images = set()
    for f in os.listdir(images_folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            available_images.add(f)

    if not available_images:
        raise ValueError(f"在 {images_folder} 中未找到图像文件")

    views_list = []
    image_names_list = []
    processed_count = 0

    # 获取所有 COLMAP 图像名称
    colmap_image_names = set(img_info.name for img_info in images_colmap.values())
    # 查找没有位姿的图像（在 images/ 中但不在 colmap 中）
    unposed_images = available_images - colmap_image_names

    if verbose and unposed_images:
        print(f"发现 {len(unposed_images)} 张没有 COLMAP 位姿的图像（将被跳过）")

    # 按 COLMAP 顺序处理图像
    for img_id, img_info in images_colmap.items():
        # 应用 stride
        if processed_count % stride != 0:
            processed_count += 1
            continue

        img_name = img_info.name

        # 检查图像文件是否存在
        image_path = os.path.join(images_folder, img_name)
        if not os.path.exists(image_path):
            if verbose:
                print(f"警告: 未找到图像文件 {img_name}，跳过")
            processed_count += 1
            continue

        try:
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image).astype(np.uint8)  # (H, W, 3) - [0, 255]

            # 获取相机信息
            cam_info = cameras[img_info.camera_id]
            cam_params = cam_info.params

            # 获取内参矩阵
            K, _ = get_camera_matrix(
                camera_params=cam_params, camera_model=cam_info.model
            )

            # 获取位姿（COLMAP 提供 world2cam，我们需要 cam2world）
            # COLMAP: world2cam 旋转和平移
            C_R_G, C_t_G = qvec2rotmat(img_info.qvec), img_info.tvec

            # 创建 4x4 world2cam 位姿矩阵
            world2cam_matrix = np.eye(4)
            world2cam_matrix[:3, :3] = C_R_G
            world2cam_matrix[:3, 3] = C_t_G

            # 使用闭式位姿逆变换转换为 cam2world
            pose_matrix = closed_form_pose_inverse(world2cam_matrix[None, :, :])[0]

            # 转换为张量
            image_tensor = torch.from_numpy(image_array)  # (H, W, 3)
            intrinsics_tensor = torch.from_numpy(K.astype(np.float32))  # (3, 3)
            pose_tensor = torch.from_numpy(pose_matrix.astype(np.float32))  # (4, 4)

            # 创建 MapAnything 推理所需的视图字典
            # 多模态输入：图片 + 内参 + 位姿
            view = {
                "img": image_tensor,  # (H, W, 3) - [0, 255]
                "intrinsics": intrinsics_tensor,  # (3, 3)
                "camera_poses": pose_tensor,  # (4, 4) OpenCV cam2world 约定
                "is_metric_scale": torch.tensor([False]),  # COLMAP 数据是非度量尺度的
            }

            views_list.append(view)
            image_names_list.append(img_name)
            processed_count += 1

            if verbose:
                print(
                    f"加载视图 {len(views_list) - 1}: {img_name} (形状: {image_array.shape})"
                )
                print(f"  - 相机 ID: {img_info.camera_id}")
                print(f"  - 相机模型: {cam_info.model}")
                print(f"  - 图像 ID: {img_id}")

        except Exception as e:
            if verbose:
                print(f"警告: 加载 {img_name} 的数据失败: {e}")
            processed_count += 1
            continue

    if not views_list:
        raise ValueError("未找到有效图像")

    if verbose:
        print(f"成功加载 {len(views_list)} 个视图 (stride={stride})")

    return views_list, image_names_list


def main():
    parser = argparse.ArgumentParser(
        description="使用 MapAnything 优化 COLMAP 数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_colmap_path",
        type=str,
        required=True,
        help="输入 COLMAP 数据集路径（包含 images/ 和 sparse/ 子文件夹）",
    )
    parser.add_argument(
        "--output_colmap_path",
        type=str,
        required=True,
        help="输出 COLMAP 数据集路径（将创建 images/ 和 sparse/ 子文件夹）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="每隔第几张图片加载一次（默认: 1，即加载所有图片）",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".bin",
        choices=[".bin", ".txt"],
        help="COLMAP 文件扩展名（默认: .bin）",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="使用 Apache 2.0 许可的模型 (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--voxel_fraction",
        type=float,
        default=0.01,
        help="基于 IQR 的场景范围分数用于体素下采样（默认: 0.01 = 1%%）",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="明确的体素大小（米）（如果提供，将覆盖 --voxel_fraction）",
    )
    parser.add_argument(
        "--skip_point2d",
        action="store_true",
        default=False,
        help="跳过 Point2D 反投影以加快 COLMAP 导出（某些工具可能需要 Point2D）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="打印详细加载信息",
    )
    parser.add_argument(
        "--ignore_calibration_inputs",
        action="store_true",
        default=False,
        help="忽略 COLMAP 标定输入（仅使用图像和位姿）",
    )
    parser.add_argument(
        "--ignore_pose_inputs",
        action="store_true",
        default=False,
        help="忽略 COLMAP 位姿输入（仅使用图像和标定）",
    )

    args = parser.parse_args()

    # 打印配置
    print("=" * 60)
    print("MapAnything COLMAP 数据集优化")
    print("=" * 60)
    print(f"输入 COLMAP 路径: {args.input_colmap_path}")
    print(f"输出 COLMAP 路径: {args.output_colmap_path}")
    print(f"Stride: {args.stride}")
    if args.voxel_size is not None:
        print(f"体素大小: {args.voxel_size}m (明确指定)")
    else:
        print(f"体素分数: {args.voxel_fraction} (自适应)")

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 初始化模型
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("加载 Apache 2.0 许可的 MapAnything 模型...")
    else:
        model_name = "facebook/map-anything"
        print("加载 CC-BY-NC 4.0 许可的 MapAnything 模型...")

    try:
        model = MapAnything.from_pretrained(model_name).to(device)
        model.eval()
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接（首次使用需要从 HuggingFace 下载模型）")
        print("2. 如果模型已下载到本地缓存，确保 HuggingFace 缓存路径正确")
        return

    # 加载 COLMAP 数据
    print(f"\n正在从以下路径加载 COLMAP 数据: {args.input_colmap_path}")
    try:
        views_list, image_names = load_colmap_data_for_refinement(
            args.input_colmap_path,
            stride=args.stride,
            verbose=args.verbose,
            ext=args.ext,
        )
        print(f"成功加载 {len(views_list)} 个视图")
    except Exception as e:
        print(f"加载 COLMAP 数据失败: {e}")
        return

    # 预处理输入到期望格式
    print("\n正在预处理 COLMAP 输入...")
    processed_views = preprocess_inputs(views_list, verbose=args.verbose)
    print("预处理完成")

    # 运行模型推理（多模态：图片 + 位姿 + 内参）
    print("\n正在运行 MapAnything 多模态推理...")
    print("  输入模式: 图片 + 相机内参 + 相机位姿")
    with torch.no_grad():
        outputs = model.infer(
            processed_views,
            memory_efficient_inference=True,
            minibatch_size=1,
            # 控制使用哪些 COLMAP 输入
            ignore_calibration_inputs=args.ignore_calibration_inputs,  # 是否使用 COLMAP 标定
            ignore_depth_inputs=True,  # COLMAP 不提供深度
            ignore_pose_inputs=args.ignore_pose_inputs,  # 是否使用 COLMAP 位姿
            ignore_depth_scale_inputs=True,  # 无深度数据
            ignore_pose_scale_inputs=True,  # COLMAP 位姿是非度量尺度的
            # 使用混合精度以提高性能
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )
    print("推理完成!")

    # 创建输出目录
    os.makedirs(args.output_colmap_path, exist_ok=True)

    # 导出为 COLMAP 格式
    print(f"\n正在导出到 COLMAP 格式: {args.output_colmap_path}")
    try:
        _ = export_predictions_to_colmap(
            outputs=outputs,
            processed_views=processed_views,
            image_names=image_names,
            output_dir=args.output_colmap_path,
            voxel_fraction=args.voxel_fraction,
            voxel_size=args.voxel_size,
            data_norm_type=model.encoder.data_norm_type,
            save_ply=True,
            save_images=True,
            skip_point2d=args.skip_point2d,
        )
        print(f"\n成功导出 COLMAP 重建到: {args.output_colmap_path}")
        print(f"  - sparse/ 目录: 包含 cameras.bin, images.bin, points3D.bin, points.ply")
        print(f"  - images/ 目录: 包含处理后的图像（匹配模型推理分辨率）")
        print(f"\n注意: 新的 COLMAP 数据集具有更好的稀疏点云质量")
        print(f"      （由 MapAnything 密集重建结果经过体素下采样得到）")
    except Exception as e:
        print(f"导出 COLMAP 格式失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
