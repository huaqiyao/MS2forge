




import os
import sys
import argparse
sys.path.append('.')

import pickle
import numpy as np
from tqdm import tqdm


from utils.visualization import create_trajectory_html


def main():
    parser = argparse.ArgumentParser(description='Regenerate HTMLvisualization')
    parser.add_argument('--trajectory_file', type=str, default="flash_result/molacc1_trajectories.pkl",
                       help='Trajectory-data file path')
    parser.add_argument('--output_dir', type=str, default="flash_result/molacc1_trajectories/",
                       help='Output directory(defaults to the trajectory file directory)')
    parser.add_argument('--max_htmls', type=int, default=None,
                       help='maximum number of generated HTML(generate all by default)')
    args = parser.parse_args()


    if not os.path.exists(args.trajectory_file):
        print(f"[ERROR] Trajectory-data file does not exist: {args.trajectory_file}")
        return


    print(f"Loading trajectory data: {args.trajectory_file}")
    with open(args.trajectory_file, 'rb') as f:
        trajectory_data = pickle.load(f)

    molecules = trajectory_data['molecules']
    atomic_numbers = trajectory_data['atomic_numbers']
    mode = trajectory_data['mode']


    if 'total_molacc1_count' in trajectory_data:
        total_count = trajectory_data['total_molacc1_count']
        mol_type = "MolAcc=1"
    elif 'total_molaccbelow1_count' in trajectory_data:
        total_count = trajectory_data['total_molaccbelow1_count']
        mol_type = "MolAcc<1"
    else:
        total_count = len(molecules)
        mol_type = "unknown type"

    print(f"Model type: {mode}")
    print(f"{mol_type}Molecule count: {total_count}")
    print(f"Sampling steps: {trajectory_data.get('sample_steps', 'N/A')}")


    if args.output_dir is None:

        args.output_dir = os.path.join(os.path.dirname(args.trajectory_file), 'trajectories')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print('='*60)


    num_to_generate = len(molecules) if args.max_htmls is None else min(args.max_htmls, len(molecules))


    print(f"Generating  {num_to_generate}  HTML...")
    for idx, mol_info in enumerate(tqdm(molecules[:num_to_generate], desc='generateHTML')):
        html_path = os.path.join(args.output_dir, f'mol_{mol_info["mol_id"]}_trajectory.html')
        create_trajectory_html(
            smiles=mol_info.get('smiles'),
            trajectory=mol_info['trajectory'],
            true_edge_types=mol_info['true_edge_types'],
            halfedge_index=mol_info['halfedge_index'],
            node_types=mol_info['node_types'],
            atomic_numbers=mol_info.get('atomic_numbers', atomic_numbers),
            output_path=html_path,
            mol_idx=mol_info['mol_id']
        )

    print('='*60)
    print(f"all with HTMLgenerated successfully, saved under: : {args.output_dir}")


if __name__ == '__main__':
    main()

