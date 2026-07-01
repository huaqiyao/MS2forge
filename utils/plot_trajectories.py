"""
从保存的轨迹数据重新生成HTML可视化
无需重新运行模型推理
"""

import os
import sys
import argparse
sys.path.append('.')

import pickle
import numpy as np
from tqdm import tqdm

# 从visualization.py导入create_trajectory_html函数
from utils.visualization import create_trajectory_html


def main():
    parser = argparse.ArgumentParser(description='从保存的轨迹数据重新生成HTML可视化')
    parser.add_argument('--trajectory_file', type=str, default="flash_result/molacc1_trajectories.pkl",
                       help='轨迹数据文件路径')
    parser.add_argument('--output_dir', type=str, default="flash_result/molacc1_trajectories/",
                       help='输出目录（默认与轨迹文件同目录）')
    parser.add_argument('--max_htmls', type=int, default=None,
                       help='最多生成多少个HTML（默认全部生成）')
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.trajectory_file):
        print(f"[错误] 轨迹数据文件不存在: {args.trajectory_file}")
        return

    # 加载轨迹数据
    print(f"正在加载轨迹数据: {args.trajectory_file}")
    with open(args.trajectory_file, 'rb') as f:
        trajectory_data = pickle.load(f)

    molecules = trajectory_data['molecules']
    atomic_numbers = trajectory_data['atomic_numbers']
    mode = trajectory_data['mode']

    # 自动检测是molacc1还是molaccbelow1
    if 'total_molacc1_count' in trajectory_data:
        total_count = trajectory_data['total_molacc1_count']
        mol_type = "MolAcc=1"
    elif 'total_molaccbelow1_count' in trajectory_data:
        total_count = trajectory_data['total_molaccbelow1_count']
        mol_type = "MolAcc<1"
    else:
        total_count = len(molecules)
        mol_type = "未知类型"

    print(f"模型类型: {mode}")
    print(f"{mol_type}分子数量: {total_count}")
    print(f"采样步数: {trajectory_data.get('sample_steps', 'N/A')}")

    # 确定输出目录
    if args.output_dir is None:
        # 默认与轨迹文件同目录
        args.output_dir = os.path.join(os.path.dirname(args.trajectory_file), 'trajectories')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"输出目录: {args.output_dir}")
    print('='*60)

    # 确定要生成的HTML数量
    num_to_generate = len(molecules) if args.max_htmls is None else min(args.max_htmls, len(molecules))

    # 生成HTML
    print(f"开始生成 {num_to_generate} 个HTML...")
    for idx, mol_info in enumerate(tqdm(molecules[:num_to_generate], desc='生成HTML')):
        html_path = os.path.join(args.output_dir, f'mol_{mol_info["mol_id"]}_trajectory.html')
        create_trajectory_html(
            smiles=mol_info.get('smiles'),
            trajectory=mol_info['trajectory'],
            true_edge_types=mol_info['true_edge_types'],
            halfedge_index=mol_info['halfedge_index'],
            node_types=mol_info['node_types'],
            atomic_numbers=mol_info.get('atomic_numbers', atomic_numbers),  # 优先使用分子自己的原子序数
            output_path=html_path,
            mol_idx=mol_info['mol_id']
        )

    print('='*60)
    print(f"所有HTML已生成完成，保存在: {args.output_dir}")


if __name__ == '__main__':
    main()

