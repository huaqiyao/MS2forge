import pickle
import os
import glob
import re
import json

import torch
import numpy as np
import pandas as pd
import lmdb
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # 静音 "Omitted undefined stereo"、"Charges were rearranged" 等
from tqdm import tqdm

from torch.utils.data import Subset, Dataset
from .parser import parse_conf_list
from .data import Drug3DData, torchify_dict
from .spectrum_utils import (
    build_mol_spectrum_mapping,
    create_spectrum_cache,
    load_spectrum_cache,
    get_spectrum_for_molecule,
    load_pretrained_embeddings,
    build_mol_pretrained_mapping,
    get_pretrained_embedding_for_molecule,
    create_pretrained_spectrum_cache,
    create_multi_instrument_pretrained_cache,
    get_pretrained_embedding_with_conditions,
    INSTRUMENT_TYPES,
    IONIZATION_TYPES,
    DEFAULT_SPECTRUM_CONFIG
)


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.root

    # 获取数据子集比例参数
    data_subset_ratio = getattr(config, 'data_subset_ratio', 1.0)

    # 获取数据划分模式参数
    data_split_mode = getattr(config, 'data_split_mode', 'natms')  # 默认使用natms模式

    # 获取质谱相关参数
    use_spectrum = getattr(config, 'use_spectrum', False)
    instrument_type = getattr(config, 'instrument_type', 'all')
    spectrum_config = getattr(config, 'spectrum_config', None)

    # 训练时是否实际使用质谱数据（用于对比实验）
    use_spectrum_in_training = getattr(config, 'use_spectrum_in_training', use_spectrum)
    # 是否使用预训练质谱特征
    use_pretrained_embeddings = getattr(config, 'use_pretrained_embeddings', False)
    pretrained_embeddings_path = getattr(config, 'pretrained_embeddings_path', None)

    # 多仪器模式配置
    pretrained_embeddings_paths = getattr(config, 'pretrained_embeddings_paths', None)
    
    if name == 'drug3d':
        dataset = Drug3DDataset(root, config.path_dict, *args, **kwargs)
    elif name == 'natgen':
        dataset = MolecularDataset(
            root,
            config.path_dict,
            dataset_name='natgen',
            data_subset_ratio=data_subset_ratio,
            use_spectrum=use_spectrum,
            instrument_type=instrument_type,
            spectrum_config=spectrum_config,
            use_spectrum_in_training=use_spectrum_in_training,
            use_pretrained_embeddings=use_pretrained_embeddings,
            pretrained_embeddings_path=pretrained_embeddings_path,
            pretrained_embeddings_paths=pretrained_embeddings_paths,
            *args,
            **kwargs
        )
    elif name == 'msg':
        dataset = MolecularDataset(
            root,
            config.path_dict,
            dataset_name='msg',
            data_subset_ratio=data_subset_ratio,
            use_spectrum=use_spectrum,
            instrument_type=instrument_type,
            spectrum_config=spectrum_config,
            use_spectrum_in_training=use_spectrum_in_training,
            use_pretrained_embeddings=use_pretrained_embeddings,
            pretrained_embeddings_path=pretrained_embeddings_path,
            pretrained_embeddings_paths=pretrained_embeddings_paths,
            *args,
            **kwargs
        )
    elif name == 'msfile':
        # 新的数据集类型：直接从MS文件读取，不需要SDF文件
        dataset = MSFileDataset(
            root,
            path_dict=getattr(config, 'path_dict', None),
            data_subset_ratio=data_subset_ratio,
            instrument_type=instrument_type,
            data_split_mode=data_split_mode,  # 传递数据分割模式
            *args,
            **kwargs
        )
    elif name == 'msg_diffms':
        # DiffMS 预处理过的 MSG（含 sub-formula）→ DeniMS 格式
        # 直接返回 (dataset, subsets)（dataset 内部已预切分）
        dataset = DiffMSMSGDataset(
            root,
            data_subset_ratio=data_subset_ratio,
            instrument_type=instrument_type,
            data_split_mode=data_split_mode,
            max_peaks=getattr(config, 'max_peaks', 128),
        )
        return dataset, dataset.subsets
    elif name == 'smiles':
        # 纯 SMILES 数据集：用于 formula 模式预训练（无谱图）
        atomic_numbers = list(getattr(config, 'atomic_numbers', []))
        if not atomic_numbers:
            raise ValueError("dataset.name='smiles' 需要在 dataset 配置中指定 atomic_numbers 列表")
        dataset = SmilesDataset(
            root=root,
            smiles_file=config.smiles_file,
            atomic_numbers=atomic_numbers,
            data_subset_ratio=data_subset_ratio,
            max_atoms=getattr(config, 'max_atoms', None),
            split_seed=getattr(config, 'split_seed', 2026),
            split_ratio=getattr(config, 'split_ratio', (0.95, 0.025, 0.025)),
        )
        # SmilesDataset 在内部已经预切分，直接暴露 subsets
        return dataset, dataset.subsets
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    
    if 'split' in config:
        # 使用预定义的split文件
        split_by_molid = torch.load(os.path.join(root, config.split))
        split = {
            k: [dataset.molid2idx[mol_id] for mol_id in mol_id_list if mol_id in dataset.molid2idx]
            for k, mol_id_list in split_by_molid.items()
        }
        subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
        print('Num of samples:', *{(k, len(v)) for k,v in split.items()})
        return dataset, subsets
    else:
        # 对于没有预定义split的数据集（如natgen、msg、msfile），创建自动划分
        if name in ['natgen', 'msg', 'msfile']:
            # ========== 根据data_split_mode选择划分方式 ==========
            if name == 'msfile' and hasattr(dataset, 'smiles2indices'):
                if data_split_mode == 'diffms':
                    # ========== DiffMS模式：按split_diffms.tsv预设划分 ==========
                    print('=== 使用DiffMS数据划分模式（按split_diffms.tsv预设划分）===')

                    # 缓存文件路径
                    instrument_type = getattr(config, 'instrument_type', 'all')
                    cache_file = os.path.join(root, f'split_indices_{instrument_type}_diffms.pt')

                    # 尝试加载缓存
                    if os.path.exists(cache_file):
                        print(f'  从缓存加载数据划分: {cache_file}')
                        split_indices = torch.load(cache_file)
                        train_indices = split_indices['train']
                        val_indices = split_indices['val']
                        test_indices = split_indices['test']
                    else:
                        # 读取split_diffms.tsv
                        split_file = os.path.join(root, 'split_diffms.tsv')
                        if not os.path.exists(split_file):
                            raise FileNotFoundError(f"DiffMS模式需要split_diffms.tsv文件，但未找到: {split_file}")

                        split_df = pd.read_csv(split_file, sep='\t')
                        gymid_to_split = dict(zip(split_df['name'], split_df['split']))
                        print(f'  从split_diffms.tsv加载了 {len(gymid_to_split)} 个GymID的划分信息')

                        # 遍历数据集，根据GymID划分
                        print(f'  正在划分数据集（首次运行较慢，结果会被缓存）...')
                        train_indices = []
                        val_indices = []
                        test_indices = []
                        missing_gymids = []

                        for idx in tqdm(range(len(dataset)), desc='  划分数据'):
                            # 获取该样本的MS文件路径
                            sample = dataset[idx]
                            ms_file_path = sample.ms_file_path

                            # 读取第一行提取GymID
                            try:
                                with open(ms_file_path, 'r') as f:
                                    first_line = f.readline().strip()
                                    # 格式: >compound MassSpecGymID0052662
                                    if first_line.startswith('>compound '):
                                        gym_id = first_line.split()[1]

                                        # 查找split标签
                                        split_label = gymid_to_split.get(gym_id, None)

                                        if split_label == 'train':
                                            train_indices.append(idx)
                                        elif split_label == 'val':
                                            val_indices.append(idx)
                                        elif split_label == 'test':
                                            test_indices.append(idx)
                                        else:
                                            missing_gymids.append((idx, gym_id))
                                    else:
                                        missing_gymids.append((idx, 'parse_error'))
                            except Exception as e:
                                missing_gymids.append((idx, f'error: {str(e)}'))

                        if missing_gymids:
                            print(f'  警告: {len(missing_gymids)} 个样本无法找到对应的split标签')

                        # 保存缓存
                        split_indices = {
                            'train': train_indices,
                            'val': val_indices,
                            'test': test_indices
                        }
                        torch.save(split_indices, cache_file)
                        print(f'  ✅ 数据划分已缓存到: {cache_file}')

                    subsets = {
                        'train': Subset(dataset, train_indices),
                        'val': Subset(dataset, val_indices),
                        'test': Subset(dataset, test_indices)
                    }

                    print(f'  Train: {len(train_indices)} 样本 ({len(train_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Val: {len(val_indices)} 样本 ({len(val_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Test: {len(test_indices)} 样本 ({len(test_indices)/len(dataset)*100:.2f}%)')

                    return dataset, subsets

                elif data_split_mode == 'split':
                    # ========== Split模式：按split.tsv预设划分 ==========
                    print('=== 使用Split数据划分模式（按split.tsv预设划分）===')

                    # 缓存文件路径
                    instrument_type = getattr(config, 'instrument_type', 'all')
                    cache_file = os.path.join(root, f'split_indices_{instrument_type}_split.pt')

                    # 尝试加载缓存
                    if os.path.exists(cache_file):
                        print(f'  从缓存加载数据划分: {cache_file}')
                        split_indices = torch.load(cache_file)
                        train_indices = split_indices['train']
                        val_indices = split_indices['val']
                        test_indices = split_indices['test']
                    else:
                        # 读取split.tsv
                        split_file = os.path.join(root, 'split.tsv')
                        if not os.path.exists(split_file):
                            raise FileNotFoundError(f"Split模式需要split.tsv文件，但未找到: {split_file}")

                        split_df = pd.read_csv(split_file, sep='\t')
                        gymid_to_split = dict(zip(split_df['name'], split_df['split']))
                        print(f'  从split.tsv加载了 {len(gymid_to_split)} 个GymID的划分信息')

                        # 遍历数据集，根据GymID划分
                        print(f'  正在划分数据集（首次运行较慢，结果会被缓存）...')
                        train_indices = []
                        val_indices = []
                        test_indices = []
                        missing_gymids = []

                        for idx in tqdm(range(len(dataset)), desc='  划分数据'):
                            sample = dataset[idx]
                            ms_file_path = sample.ms_file_path

                            try:
                                with open(ms_file_path, 'r') as f:
                                    first_line = f.readline().strip()
                                    if first_line.startswith('>compound '):
                                        gym_id = first_line.split()[1]
                                        split_label = gymid_to_split.get(gym_id, None)

                                        if split_label == 'train':
                                            train_indices.append(idx)
                                        elif split_label == 'val':
                                            val_indices.append(idx)
                                        elif split_label == 'test':
                                            test_indices.append(idx)
                                        else:
                                            missing_gymids.append((idx, gym_id))
                                    else:
                                        missing_gymids.append((idx, 'parse_error'))
                            except Exception as e:
                                missing_gymids.append((idx, f'error: {str(e)}'))

                        if missing_gymids:
                            print(f'  警告: {len(missing_gymids)} 个样本无法找到对应的split标签')

                        # 保存缓存
                        split_indices = {
                            'train': train_indices,
                            'val': val_indices,
                            'test': test_indices
                        }
                        torch.save(split_indices, cache_file)
                        print(f'  数据划分已缓存到: {cache_file}')

                    subsets = {
                        'train': Subset(dataset, train_indices),
                        'val': Subset(dataset, val_indices),
                        'test': Subset(dataset, test_indices)
                    }

                    print(f'  Train: {len(train_indices)} 样本 ({len(train_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Val: {len(val_indices)} 样本 ({len(val_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Test: {len(test_indices)} 样本 ({len(test_indices)/len(dataset)*100:.2f}%)')

                    return dataset, subsets

                else:
                    # ========== NatMS模式：按split_natms.tsv预设划分 ==========
                    print('=== 使用NatMS数据划分模式（按split_natms.tsv预设划分）===')

                    # 缓存文件路径
                    instrument_type = getattr(config, 'instrument_type', 'all')
                    cache_file = os.path.join(root, f'split_indices_natms_{instrument_type.lower()}.pt')

                    # 尝试加载缓存
                    if os.path.exists(cache_file):
                        print(f'  从缓存加载数据划分: {cache_file}')
                        split_indices = torch.load(cache_file)
                        train_indices = split_indices['train']
                        val_indices = split_indices['val']
                        test_indices = split_indices['test']
                    else:
                        # 读取split_natms.tsv
                        split_file = os.path.join(root, 'split_natms.tsv')
                        if not os.path.exists(split_file):
                            raise FileNotFoundError(f"NatMS模式需要split_natms.tsv文件，但未找到: {split_file}")

                        split_df = pd.read_csv(split_file, sep='\t')
                        gymid_to_split = dict(zip(split_df['name'], split_df['split']))
                        print(f'  从split_natms.tsv加载了 {len(gymid_to_split)} 个GymID的划分信息')

                        # 遍历数据集，根据GymID划分
                        print(f'  正在划分数据集（首次运行较慢，结果会被缓存）...')
                        train_indices = []
                        val_indices = []
                        test_indices = []
                        missing_gymids = []

                        for idx in range(len(dataset)):
                            try:
                                # 获取样本的ms文件路径
                                sample_info = dataset.get_raw(idx)
                                ms_file_path = sample_info.get('ms_file_path', None)

                                if ms_file_path and os.path.exists(ms_file_path):
                                    # 读取ms文件的第一行获取GymID
                                    with open(ms_file_path, 'r') as f:
                                        first_line = f.readline().strip()
                                        # 格式: >compound MassSpecGymID0052662
                                        if first_line.startswith('>compound '):
                                            gym_id = first_line.split()[1]

                                            # 查找split标签
                                            split_label = gymid_to_split.get(gym_id, None)

                                            if split_label == 'train':
                                                train_indices.append(idx)
                                            elif split_label == 'val':
                                                val_indices.append(idx)
                                            elif split_label == 'test':
                                                test_indices.append(idx)
                                            else:
                                                missing_gymids.append((idx, gym_id))
                                        else:
                                            missing_gymids.append((idx, 'parse_error'))
                            except Exception as e:
                                missing_gymids.append((idx, f'error: {str(e)}'))

                        if missing_gymids:
                            print(f'  警告: {len(missing_gymids)} 个样本无法找到对应的split标签')

                        # 保存缓存
                        split_indices = {
                            'train': train_indices,
                            'val': val_indices,
                            'test': test_indices
                        }
                        torch.save(split_indices, cache_file)
                        print(f'  ✅ 数据划分已缓存到: {cache_file}')

                    subsets = {
                        'train': Subset(dataset, train_indices),
                        'val': Subset(dataset, val_indices),
                        'test': Subset(dataset, test_indices)
                    }

                    print(f'  Train: {len(train_indices)} 样本 ({len(train_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Val: {len(val_indices)} 样本 ({len(val_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Test: {len(test_indices)} 样本 ({len(test_indices)/len(dataset)*100:.2f}%)')

                    return dataset, subsets
            else:
                # 其他数据集：按样本划分（原有逻辑）
                total_size = len(dataset)
                indices = list(range(total_size))
                np.random.seed(2023)  # 确保可重复性
                np.random.shuffle(indices)

                train_size = int(0.8 * total_size)
                val_size = int(0.1 * total_size)

                train_indices = indices[:train_size]
                val_indices = indices[train_size:train_size + val_size]
                test_indices = indices[train_size + val_size:]

                subsets = {
                    'train': Subset(dataset, train_indices),
                    'val': Subset(dataset, val_indices),
                    'test': Subset(dataset, test_indices)
                }

                print('Auto-generated dataset split:')
                print(f'  Train: {len(train_indices)} samples')
                print(f'  Val: {len(val_indices)} samples')
                print(f'  Test: {len(test_indices)} samples')

                return dataset, subsets
        else:
            return dataset


class MolecularDataset(Dataset):
    """
    通用分子3D数据集
    支持多种数据集类型：natgen（天然产物）、msg（MSG数据集）
    处理包含多个sdf文件的文件夹，每个sdf文件包含一个分子3D结构
    """

    def __init__(self, root, path_dict, dataset_name='natgen', transform=None, data_subset_ratio=1.0,
                 use_spectrum=False, instrument_type='all', spectrum_config=None,
                 use_spectrum_in_training=None,
                 use_pretrained_embeddings=False, pretrained_embeddings_path=None,
                 pretrained_embeddings_paths=None):
        super().__init__()
        self.root = root
        self.dataset_name = dataset_name  # 数据集名称：natgen 或 msg
        self.sdf_path = os.path.join(root, path_dict['sdf'])  # sdf文件夹路径
        self.data_subset_ratio = data_subset_ratio  # 数据处理比例，1.0表示全部，0.1表示10%

        # 质谱相关参数
        self.use_spectrum = use_spectrum
        self.instrument_type = instrument_type  # 'all' 表示使用所有仪器类型
        self.spectrum_config = spectrum_config if spectrum_config else DEFAULT_SPECTRUM_CONFIG
        self.use_spectrum_in_training = use_spectrum_in_training if use_spectrum_in_training is not None else use_spectrum

        # 预训练特征相关参数
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.pretrained_embeddings_path = pretrained_embeddings_path  # 单仪器模式
        self.pretrained_embeddings_paths = pretrained_embeddings_paths  # 多仪器模式 {'Orbitrap': path1, 'QTOF': path2}
        self.use_multi_instrument = pretrained_embeddings_paths is not None

        # [DEBUG] 打印数据集初始化参数
        print(f"[DEBUG Dataset Init] dataset_name: {self.dataset_name}")
        print(f"[DEBUG Dataset Init] use_spectrum: {self.use_spectrum}")
        print(f"[DEBUG Dataset Init] use_spectrum_in_training: {self.use_spectrum_in_training}")
        print(f"[DEBUG Dataset Init] 默认只处理有预训练特征的分子")
        print(f"[DEBUG Dataset Init] instrument_type: {self.instrument_type}")
        print(f"[DEBUG Dataset Init] use_pretrained_embeddings: {self.use_pretrained_embeddings}")
        print(f"[DEBUG Dataset Init] use_multi_instrument: {self.use_multi_instrument}")

        # 根据数据集名称和数据比例、预训练特征使用情况调整处理后文件的名称，避免冲突
        suffix_parts = [self.dataset_name]  # 加入数据集名称
        if data_subset_ratio < 1.0:
            suffix_parts.append(f"{int(data_subset_ratio * 100)}pct")
        # 添加质谱标识
        if self.use_multi_instrument:
            suffix_parts.append("spectrum_multi_instrument")
        else:
            suffix_parts.append(f"spectrum_{instrument_type.lower()}")
        if use_pretrained_embeddings:
            suffix_parts.append("pretrained")

        suffix = "_" + "_".join(suffix_parts)

        # 根据数据集类型决定处理文件的保存位置
        if self.dataset_name == 'msg':
            # MSG数据集的处理文件保存在msg_data文件夹下
            msg_data_dir = os.path.join(root, 'msg_data')
            os.makedirs(msg_data_dir, exist_ok=True)
            self.processed_path = os.path.join(msg_data_dir, path_dict['processed'].replace('.lmdb', f'{suffix}.lmdb'))
        else:
            # 其他数据集保存在根目录下
            self.processed_path = os.path.join(root, path_dict['processed'].replace('.lmdb', f'{suffix}.lmdb'))

        self.molid2idx_path = self.processed_path[:self.processed_path.find('.lmdb')]+'_molid2idx.pt'

        self.transform = transform
        self.db = None
        self.keys = None

        # 质谱数据相关
        self.spectrum_cache = None
        self.pretrained_cache = None
        self.mol_spectrum_mapping = {}  # 分子-质谱映射（统一缓存格式）
        self.cache_stats = {}  # 缓存统计信息
        self.cache_metadata = {}  # 缓存元数据

        # 修改：默认只处理有预训练特征的分子，总是加载预训练特征缓存用于筛选
        print("加载预训练特征缓存用于筛选有预训练特征的分子...")
        self._setup_pretrained_embeddings()

        if (not os.path.exists(self.processed_path)) or (not os.path.exists(self.molid2idx_path)):
            self._process()
            self._precompute_molid2idx()
        self.molid2idx = torch.load(self.molid2idx_path)
    
    def _setup_spectrum_data(self):
        """设置质谱数据"""
        # 根据数据集类型生成不同的缓存文件名和路径
        if self.dataset_name == 'natgen':
            spectrum_cache_path = os.path.join(self.root, f'spectrum_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')
        elif self.dataset_name == 'msg':
            # MSG数据集的缓存文件保存在msg_data文件夹下
            msg_data_dir = os.path.join(self.root, 'msg_data')
            os.makedirs(msg_data_dir, exist_ok=True)
            spectrum_cache_path = os.path.join(msg_data_dir, f'spectrum_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')
        else:
            spectrum_cache_path = os.path.join(self.root, f'spectrum_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')
        
        if os.path.exists(spectrum_cache_path):
            print(f"加载已存在的质谱缓存: {spectrum_cache_path}")
            self.spectrum_cache = load_spectrum_cache(spectrum_cache_path)
        else:
            print(f"创建新的质谱缓存...")
            # 根据数据集类型构建不同的质谱文件路径
            if self.dataset_name == 'natgen':
                spec_dir = os.path.join(self.root, 'natgen_matched_spec_files')
            elif self.dataset_name == 'msg':
                spec_dir = os.path.join(self.root, 'msg_matched_spec_files')
            else:
                raise ValueError(f"不支持的数据集类型: {self.dataset_name}")
                
            if os.path.exists(spec_dir):
                mol_spec_mapping = build_mol_spectrum_mapping(
                    self.sdf_path, 
                    spec_dir, 
                    self.instrument_type
                )
                # 创建质谱缓存
                self.spectrum_cache = create_spectrum_cache(
                    mol_spec_mapping, 
                    self.spectrum_config, 
                    spectrum_cache_path
                )
            else:
                print(f"警告: 质谱文件目录不存在: {spec_dir}")
                print("将在不使用质谱条件的情况下继续训练")
                self.use_spectrum = False
                self.spectrum_cache = None
        
        # 如果只使用有质谱的分子，记录有质谱的分子ID列表
        self.spectrum_mol_ids = None
        if self.spectrum_cache:
            self.spectrum_mol_ids = set(self.spectrum_cache.keys())
            print(f"找到 {len(self.spectrum_mol_ids)} 个有质谱数据的分子")
            print("将只处理有质谱数据的分子")

    def _setup_pretrained_embeddings(self):
        """设置预训练质谱特征（支持多仪器类型）"""
        # 多仪器模式
        if self.use_multi_instrument and self.pretrained_embeddings_paths:
            print("[INFO] 使用多仪器模式加载预训练特征...")

            # 根据数据集类型生成缓存文件名
            if self.dataset_name == 'msg':
                msg_data_dir = os.path.join(self.root, 'msg_data')
                os.makedirs(msg_data_dir, exist_ok=True)
                pretrained_cache_path = os.path.join(msg_data_dir, f'pretrained_cache_{self.dataset_name}_multi_instrument.pkl')
            else:
                pretrained_cache_path = os.path.join(self.root, f'pretrained_cache_{self.dataset_name}_multi_instrument.pkl')

            if os.path.exists(pretrained_cache_path):
                print(f"加载已存在的多仪器预训练特征缓存: {pretrained_cache_path}")
                with open(pretrained_cache_path, 'rb') as f:
                    loaded_cache = pickle.load(f)

                # 支持新的统一缓存格式（version 2.0）
                if isinstance(loaded_cache, dict) and loaded_cache.get('version') == '2.0':
                    print(f"[INFO] 检测到统一缓存格式 v2.0")
                    self.pretrained_cache = loaded_cache['pretrained_features']
                    self.mol_spectrum_mapping = loaded_cache.get('mol_spectrum_mapping', {})
                    self.cache_stats = loaded_cache.get('stats', {})
                    self.cache_metadata = loaded_cache.get('metadata', {})
                    print(f"[INFO] 缓存统计: {self.cache_stats.get('total_molecules', 0)} 个分子, {self.cache_stats.get('total_features', 0)} 个特征")
                else:
                    # 旧格式：直接使用
                    self.pretrained_cache = loaded_cache
                    self.mol_spectrum_mapping = {}
                    self.cache_stats = {}
                    self.cache_metadata = {}
            else:
                print(f"创建新的多仪器预训练特征缓存...")
                # 根据数据集类型构建质谱文件路径
                if self.dataset_name == 'natgen':
                    spec_dir = os.path.join(self.root, 'natgen_matched_spec_files')
                elif self.dataset_name == 'msg':
                    spec_dir = os.path.join(self.root, 'msg_matched_spec_files')
                else:
                    raise ValueError(f"不支持的数据集类型: {self.dataset_name}")

                if os.path.exists(spec_dir):
                    self.pretrained_cache = create_multi_instrument_pretrained_cache(
                        self.sdf_path,
                        spec_dir,
                        self.pretrained_embeddings_paths,
                        pretrained_cache_path
                    )
                else:
                    print(f"警告: 质谱文件目录不存在: {spec_dir}")
                    self.use_spectrum = False
                    self.use_pretrained_embeddings = False
                    self.pretrained_cache = None

        # 单仪器模式（向后兼容）
        elif self.pretrained_embeddings_path and os.path.exists(self.pretrained_embeddings_path):
            print("[INFO] 使用单仪器模式加载预训练特征...")

            # 根据数据集类型生成预训练特征缓存文件名
            if self.dataset_name == 'natgen':
                pretrained_cache_path = os.path.join(self.root, f'pretrained_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')
            elif self.dataset_name == 'msg':
                msg_data_dir = os.path.join(self.root, 'msg_data')
                os.makedirs(msg_data_dir, exist_ok=True)
                pretrained_cache_path = os.path.join(msg_data_dir, f'pretrained_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')
            else:
                pretrained_cache_path = os.path.join(self.root, f'pretrained_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')

            if os.path.exists(pretrained_cache_path):
                print(f"加载已存在的预训练特征缓存: {pretrained_cache_path}")
                with open(pretrained_cache_path, 'rb') as f:
                    self.pretrained_cache = pickle.load(f)
            else:
                print(f"创建新的预训练特征缓存...")
                if self.dataset_name == 'natgen':
                    spec_dir = os.path.join(self.root, 'natgen_matched_spec_files')
                elif self.dataset_name == 'msg':
                    spec_dir = os.path.join(self.root, 'msg_matched_spec_files')
                else:
                    raise ValueError(f"不支持的数据集类型: {self.dataset_name}")

                if os.path.exists(spec_dir):
                    self.pretrained_cache = create_pretrained_spectrum_cache(
                        self.sdf_path,
                        spec_dir,
                        self.pretrained_embeddings_path,
                        self.instrument_type,
                        pretrained_cache_path
                    )
                else:
                    print(f"警告: 质谱文件目录不存在: {spec_dir}")
                    self.use_spectrum = False
                    self.use_pretrained_embeddings = False
                    self.pretrained_cache = None
        else:
            print(f"警告: 预训练特征文件不存在")
            print("将在不使用质谱条件的情况下继续训练")
            self.use_spectrum = False
            self.use_pretrained_embeddings = False
            self.pretrained_cache = None

        # 设置有预训练特征的分子ID列表，默认只处理这些分子
        self.spectrum_mol_ids = None
        if self.pretrained_cache:
            self.spectrum_mol_ids = set(self.pretrained_cache.keys())
            print(f"找到 {len(self.spectrum_mol_ids)} 个有预训练特征的分子")
            print("将只处理有预训练特征的分子")
        else:
            print("警告: 没有预训练特征缓存，将处理所有分子")

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        print(f"Processing {self.dataset_name.upper()} dataset from {self.sdf_path}")
        print(f"Data subset ratio: {self.data_subset_ratio:.1%}")

        # 获取所有sdf文件
        sdf_files = glob.glob(os.path.join(self.sdf_path, "*.sdf"))
        if len(sdf_files) == 0:
            raise ValueError(f"No SDF files found in {self.sdf_path}")

        # 根据subset_ratio选择要处理的文件
        if self.data_subset_ratio < 1.0:
            np.random.seed(2023)  # 确保可重复性
            np.random.shuffle(sdf_files)
            num_files_to_process = int(len(sdf_files) * self.data_subset_ratio)
            sdf_files = sdf_files[:num_files_to_process]
            print(f"Selected {num_files_to_process} out of {len(glob.glob(os.path.join(self.sdf_path, '*.sdf')))} SDF files for processing")
        else:
            print(f"Processing all {len(sdf_files)} SDF files")

        # 第一遍扫描：自动检测所有原子类型
        print("\n[INFO] 第一遍扫描：自动检测数据集中的原子类型...")
        all_elements = set()
        valid_mol_ids = set()  # 记录有效的分子ID

        for sdf_file in tqdm(sdf_files, desc='扫描原子类型'):
            mol_name = os.path.splitext(os.path.basename(sdf_file))[0]
            mol_id = mol_name

            # 只扫描有预训练特征的分子
            if self.spectrum_mol_ids and mol_id not in self.spectrum_mol_ids:
                continue

            try:
                suppl = Chem.SDMolSupplier(sdf_file)
                if len(suppl) == 0:
                    continue
                mol = suppl[0]
                if mol is None:
                    continue
                mol = Chem.RemoveAllHs(mol)
                if mol.GetNumAtoms() == 0:
                    continue
                if mol.GetNumAtoms() > 100 or mol.GetNumAtoms() < 5:
                    continue

                mol_elements = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
                all_elements.update(mol_elements)
                valid_mol_ids.add(mol_id)
            except:
                continue

        # 原子序数到元素符号的映射
        atomic_num_to_symbol = {
            1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 33: 'As', 34: 'Se', 35: 'Br', 53: 'I'
        }

        # 按原子序数排序
        supported_elements = sorted(list(all_elements))
        element_symbols = [atomic_num_to_symbol.get(z, f'Z{z}') for z in supported_elements]

        print(f"\n[INFO] 检测到 {len(supported_elements)} 种原子类型:")
        print(f"       原子序数: {supported_elements}")
        print(f"       元素符号: {element_symbols}")
        print(f"       有效分子数: {len(valid_mol_ids)}")

        # 保存检测到的原子类型
        self.detected_atomic_numbers = supported_elements
        supported_elements_set = set(supported_elements)

        # 创建lmdb数据库
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )

        num_skipped = 0
        num_processed = 0

        print("\n[INFO] 第二遍扫描：处理分子数据...")
        with db.begin(write=True, buffers=True) as txn:
            for sdf_file in tqdm(sdf_files, desc='Processing SDF files'):
                try:
                    # 从文件名生成mol_id
                    mol_name = os.path.splitext(os.path.basename(sdf_file))[0]
                    mol_id = mol_name  # 使用文件名作为mol_id

                    # 默认只处理有预训练特征的分子
                    if self.spectrum_mol_ids and mol_id not in self.spectrum_mol_ids:
                        num_skipped += 1
                        continue  # 跳过没有预训练特征的分子

                    # 读取sdf文件
                    suppl = Chem.SDMolSupplier(sdf_file)

                    # 检查文件是否有效
                    if len(suppl) == 0:
                        num_skipped += 1
                        continue

                    # 读取第一个分子（假设每个sdf文件只有一个分子）
                    mol = suppl[0]
                    if mol is None:
                        num_skipped += 1
                        continue

                    # 移除氢原子
                    mol = Chem.RemoveAllHs(mol)

                    # 生成SMILES
                    smiles = Chem.MolToSmiles(mol)

                    # 检查分子有效性
                    if mol.GetNumAtoms() == 0:
                        num_skipped += 1
                        continue

                    # 使用自动检测的原子类型进行验证
                    mol_elements = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
                    if not mol_elements.issubset(supported_elements_set):
                        unsupported = mol_elements - supported_elements_set
                        num_skipped += 1
                        continue

                    # 检查分子大小（可选的过滤条件）
                    if mol.GetNumAtoms() > 100:  # 限制最大原子数
                        num_skipped += 1
                        continue

                    if mol.GetNumAtoms() < 5:  # 限制最小原子数
                        num_skipped += 1
                        continue
                    
                    # 由于每个sdf文件只有一个构象，我们将其包装成list
                    confs_list = [mol]
                    
                    # 解析分子数据
                    ligand_dict = parse_conf_list(confs_list, smiles=smiles)
                    if ligand_dict['num_confs'] == 0:
                        print(f"Warning: No valid conformers found in {sdf_file}")
                        num_skipped += 1
                        continue
                    
                    # 转换为torch格式
                    ligand_dict = torchify_dict(ligand_dict)
                    data = Drug3DData.from_drug3d_dicts(ligand_dict)

                    # 添加额外信息
                    data.smiles = smiles
                    data.mol_id = mol_id
                    data.source_file = sdf_file
                    
                    # [NEW] 质谱数据扩增：支持预训练特征和原始质谱两种模式
                    if self.use_spectrum_in_training:
                        if self.use_pretrained_embeddings and self.pretrained_cache is not None and mol_id in self.pretrained_cache:
                            # 预训练特征模式：为每个预训练特征创建独立的训练样本
                            pretrained_list = self.pretrained_cache[mol_id]

                            # 为每个预训练特征创建一个训练样本
                            for feat_idx, feature_entry in enumerate(pretrained_list):
                                # 创建数据副本
                                data_with_pretrained = Drug3DData.from_drug3d_dicts(ligand_dict)
                                data_with_pretrained.smiles = smiles
                                data_with_pretrained.mol_id = f"{mol_id}_feat{feat_idx}"  # 唯一的样本ID
                                data_with_pretrained.original_mol_id = mol_id  # 保留原始分子ID
                                data_with_pretrained.source_file = sdf_file

                                # 添加预训练特征数据
                                data_with_pretrained.has_spectrum = True
                                data_with_pretrained.spec_data = None  # 不使用原始质谱数据
                                data_with_pretrained.spec_env = None
                                data_with_pretrained.feature_index = feat_idx

                                # 多仪器模式：feature_entry是字典，包含embedding和条件信息
                                if self.use_multi_instrument and isinstance(feature_entry, dict):
                                    data_with_pretrained.pretrained_embedding = torch.from_numpy(feature_entry['embedding']).float()
                                    data_with_pretrained.instrument_type = feature_entry['instrument_type']
                                    data_with_pretrained.ionization = feature_entry['ionization']
                                    data_with_pretrained.instrument_type_idx = feature_entry['instrument_type_idx']
                                    data_with_pretrained.ionization_type_idx = feature_entry['ionization_type_idx']
                                else:
                                    # 单仪器模式：feature_entry是numpy数组
                                    data_with_pretrained.pretrained_embedding = torch.from_numpy(feature_entry).float()
                                    data_with_pretrained.instrument_type = self.instrument_type
                                    data_with_pretrained.ionization = '[M+H]+'  # 默认值
                                    data_with_pretrained.instrument_type_idx = INSTRUMENT_TYPES.index(self.instrument_type) if self.instrument_type in INSTRUMENT_TYPES else INSTRUMENT_TYPES.index('NONE')
                                    data_with_pretrained.ionization_type_idx = 0

                                # 存储到lmdb（使用包含特征索引的唯一key）
                                unique_key = f"{mol_id}_feat{feat_idx}"
                                txn.put(
                                    key=unique_key.encode('utf-8'),
                                    value=pickle.dumps(data_with_pretrained)
                                )
                                num_processed += 1

                        elif not self.use_pretrained_embeddings and self.spectrum_cache is not None and mol_id in self.spectrum_cache:
                            # 原始质谱模式：为每个质谱创建独立的训练样本
                            spec_list = self.spectrum_cache[mol_id]
                            print(f"[DEBUG Preprocessing] Found {len(spec_list)} spectra for mol_id: {mol_id}")
                            
                            # 为每个质谱创建一个训练样本
                            for spec_idx, spectrum_data in enumerate(spec_list):
                                # 创建数据副本
                                data_with_spec = Drug3DData.from_drug3d_dicts(ligand_dict)
                                data_with_spec.smiles = smiles
                                data_with_spec.mol_id = f"{mol_id}_spec{spec_idx}"  # 唯一的样本ID
                                data_with_spec.original_mol_id = mol_id  # 保留原始分子ID
                                data_with_spec.source_file = sdf_file
                                
                                # 添加质谱数据
                                data_with_spec.has_spectrum = True
                                data_with_spec.spec_data = torch.from_numpy(spectrum_data['spec'][:, 0]).float()
                                data_with_spec.spec_env = torch.from_numpy(spectrum_data['env']).float()
                                data_with_spec.instrument_type = self.instrument_type
                                data_with_spec.spectrum_index = spec_idx
                                
                                # 存储到lmdb（使用包含质谱索引的唯一key）
                                unique_key = f"{mol_id}_spec{spec_idx}"
                                txn.put(
                                    key=unique_key.encode('utf-8'),
                                    value=pickle.dumps(data_with_spec)
                                )
                                num_processed += 1
                                print(f"[DEBUG Preprocessing] Stored sample with spectrum: {unique_key}")
                            
                            print(f"[DEBUG Preprocessing] Created {len(spec_list)} training samples for molecule {mol_id}")
                        else:
                            # 没有质谱数据的情况，创建一个不带质谱的样本
                            data.has_spectrum = False
                            data.spec_data = None
                            data.spec_env = None
                            data.pretrained_embedding = None
                            data.instrument_type = None
                            
                            # 存储到lmdb
                            txn.put(
                                key=str(mol_id).encode('utf-8'),
                                value=pickle.dumps(data)
                            )
                            num_processed += 1
                            print(f"[DEBUG Preprocessing] Stored sample without spectrum: {mol_id}")
                    else:
                        # 不使用质谱训练时，创建标准的不带质谱的样本
                        data.has_spectrum = False
                        data.spec_data = None
                        data.spec_env = None
                        data.pretrained_embedding = None
                        data.instrument_type = None
                        
                        # 存储到lmdb
                        txn.put(
                            key=str(mol_id).encode('utf-8'),
                            value=pickle.dumps(data)
                        )
                        num_processed += 1
                        print(f"[DEBUG Preprocessing] Stored standard sample (no spectrum): {mol_id}")
                    
                except Exception as e:
                    print(f"Error processing {sdf_file}: {str(e)}")
                    num_skipped += 1
                    continue
        
        db.close()
        
        # 统计质谱数据扩增效果
        if self.use_spectrum_in_training:
            if self.use_pretrained_embeddings and self.pretrained_cache:
                total_molecules = len([f for f in sdf_files if os.path.splitext(os.path.basename(f))[0] in self.pretrained_cache])
                total_features = sum(len(feat_list) for feat_list in self.pretrained_cache.values())
                print(f'=== 预训练特征数据扩增统计 ===')
                print(f'有预训练特征的分子数: {total_molecules}')
                print(f'预训练特征总数: {total_features}')
                print(f'平均每个分子的预训练特征数: {total_features/total_molecules if total_molecules > 0 else 0:.2f}')
                print(f'数据扩增倍数: {total_features/total_molecules if total_molecules > 0 else 1:.2f}x')
            elif not self.use_pretrained_embeddings and self.spectrum_cache:
                total_molecules = len([f for f in sdf_files if os.path.splitext(os.path.basename(f))[0] in self.spectrum_cache])
                total_spectra = sum(len(spec_list) for spec_list in self.spectrum_cache.values())
                print(f'=== 质谱数据扩增统计 ===')
                print(f'有质谱的分子数: {total_molecules}')
                print(f'质谱总数: {total_spectra}')
                print(f'平均每个分子的质谱数: {total_spectra/total_molecules if total_molecules > 0 else 0:.2f}')
                print(f'数据扩增倍数: {total_spectra/total_molecules if total_molecules > 0 else 1:.2f}x')
            
        print(f'Processed {num_processed} training samples, skipped {num_skipped} molecules')
        print(f'Using {self.data_subset_ratio:.1%} of available data')

    def _precompute_molid2idx(self):
        molid2idx = {}
        for i in tqdm(range(self.__len__()), 'Indexing dataset'):
            try:
                data = self.__getitem__(i)
            except Exception as e:
                print(f"Error at index {i}: {e}")
                continue
            mol_id = data.mol_id
            molid2idx[mol_id] = i
        torch.save(molid2idx, self.molid2idx_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        if self.transform is not None:
            data = self.transform(data)
        return data


class Drug3DDataset(Dataset):

    def __init__(self, root, path_dict, transform=None):
        super().__init__()
        self.root = root
        self.sdf_path = os.path.join(root, path_dict['sdf'])
        self.summary_path = os.path.join(root, path_dict['summary'])
        
        self.processed_path = os.path.join(root, path_dict['processed'])
        self.molid2idx_path = self.processed_path[:self.processed_path.find('.lmdb')]+'_molid2idx.pt'
        # self.filter = filter

        self.transform = transform
        self.db = None
        self.keys = None

        if (not os.path.exists(self.processed_path)) or (not os.path.exists(self.molid2idx_path)):
            self._process()
            self._precompute_molid2idx()
        self.molid2idx = torch.load(self.molid2idx_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )
        
        # read summary
        df_summary = pd.read_csv(self.summary_path, index_col=0)
        
        # filter 
        df_use = df_summary[df_summary['pass_size'] & df_summary['pass_element'] &
                            (~df_summary['broken']) & (~df_summary['error_mol'])]
        
        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for _, line in tqdm(df_use.iterrows(), total=len(df_use), desc='Preprocessing data'):
                # mol info
                mol_id = line['mol_id']
                smiles = line['smiles']
                
                try:
                    # load all confs of the mol
                    suppl = Chem.SDMolSupplier(os.path.join(self.sdf_path, 'mol_%d.sdf' % mol_id))
                    confs_list = []
                    for i_conf in range(len(suppl)):
                        mol = Chem.MolFromMolBlock(suppl.GetItemText(i_conf).replace(
                            "RDKit          3D", "RDKit          2D"
                        ))  # removeHs=True is default
                        mol = Chem.RemoveAllHs(mol)
                        confs_list.append(mol)
                    
                    # build data
                    ligand_dict = parse_conf_list(confs_list, smiles=smiles)
                    if ligand_dict['num_confs'] == 0:
                        raise ValueError('No conformers found')
                    ligand_dict = torchify_dict(ligand_dict)
                    data = Drug3DData.from_drug3d_dicts(ligand_dict)

                    data.smiles = smiles
                    data.mol_id = mol_id
                    
                    txn.put(
                        key = str(mol_id).encode(),
                        value = pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) Num: %s, %s' % (num_skipped, mol_id, smiles))
                    continue
        db.close()
        print('Processed %d molecules' % (len(df_use) - num_skipped), 'Skipped %d molecules' % num_skipped)


    def _precompute_molid2idx(self):
        molid2idx = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            mol_id = data.mol_id
            molid2idx[mol_id] = i
        torch.save(molid2idx, self.molid2idx_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        
        # [DEBUG] 添加调试信息
        global debug_getitem_counter
        if not 'debug_getitem_counter' in globals():
            debug_getitem_counter = 0
        debug_getitem_counter += 1
        
        # 只在前10次调用时打印调试信息
        should_debug = debug_getitem_counter <= 10
        
        if should_debug:
            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] Processing sample {idx}")
            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] use_pretrained_embeddings: {self.use_pretrained_embeddings}")
            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] Loaded data.has_spectrum: {getattr(data, 'has_spectrum', 'NOT SET')}")
        
        # 处理质谱数据 - 支持预训练特征和原始质谱两种模式
        if self.use_spectrum_in_training:
            if self.use_pretrained_embeddings:
                # 使用预训练特征模式
                mol_id = str(data.mol_id) if hasattr(data, 'mol_id') else str(data.original_mol_id) if hasattr(data, 'original_mol_id') else None
                
                if should_debug:
                    print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] 预训练特征模式 - mol_id: {mol_id}")
                    print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] pretrained_cache是否存在: {self.pretrained_cache is not None}")
                    if self.pretrained_cache:
                        print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] pretrained_cache大小: {len(self.pretrained_cache)}")
                        print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] mol_id在cache中: {mol_id in self.pretrained_cache if mol_id else False}")
                
                if mol_id and self.pretrained_cache and mol_id in self.pretrained_cache:
                    # 获取预训练特征
                    pretrained_embedding = get_pretrained_embedding_for_molecule(mol_id, self.pretrained_cache, mode='random')
                    if pretrained_embedding is not None:
                        data.has_spectrum = True
                        data.pretrained_embedding = torch.from_numpy(pretrained_embedding).float()  # [1024]
                        data.spec_data = None  # 不使用原始质谱数据
                        data.spec_env = None
                        
                        if should_debug:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] ✅ 成功添加预训练特征: shape={data.pretrained_embedding.shape}")
                    else:
                        data.has_spectrum = False
                        data.pretrained_embedding = None
                        data.spec_data = None
                        data.spec_env = None
                        
                        if should_debug:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] ❌ 预训练特征获取失败")
                else:
                    data.has_spectrum = False
                    data.pretrained_embedding = None
                    data.spec_data = None
                    data.spec_env = None
                    
                    if should_debug:
                        if not mol_id:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] ❌ mol_id为空")
                        elif not self.pretrained_cache:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] ❌ pretrained_cache为空")
                        elif mol_id not in self.pretrained_cache:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] ❌ mol_id {mol_id} 不在pretrained_cache中")
            else:
                # 使用原始质谱数据模式（保持现有逻辑）
                if hasattr(data, 'spec_data'):
                    if should_debug:
                        print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] spec_data is not None: {data.spec_data is not None}")
                        if data.spec_data is not None:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] spec_data shape: {data.spec_data.shape}")
                    data.pretrained_embedding = None  # 确保没有预训练特征
                else:
                    data.has_spectrum = False
                    data.spec_data = None
                    data.spec_env = None
                    data.pretrained_embedding = None
        else:
            # 不使用质谱条件
            data.has_spectrum = False
            data.spec_data = None
            data.spec_env = None
            data.pretrained_embedding = None
        
        # data.id = idx
        if self.transform is not None:
            if should_debug:
                print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] Applying transform...")
                print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] Before transform has_spectrum: {getattr(data, 'has_spectrum', 'NOT SET')}")
            data = self.transform(data)
            if should_debug:
                print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] After transform has_spectrum: {getattr(data, 'has_spectrum', 'NOT SET')}")
        return data


def collate_with_pretrained_features(batch):
    """
    自定义collate函数，用于处理带有预训练质谱特征的批次数据

    Args:
        batch: 数据样本列表
    Returns:
        batched_data: 合并后的批次数据
    """
    from torch_geometric.data import Batch

    # 检查批次中是否有预训练特征
    has_pretrained = any(
        hasattr(data, 'pretrained_embedding') and data.pretrained_embedding is not None
        for data in batch
    )

    # 保存质谱相关数据，然后从原始数据中移除，避免collate冲突
    spectrum_info = []
    cleaned_data_list = []

    for data in batch:
        # 保存质谱相关信息（包括条件信息）
        info = {
            'has_spectrum': getattr(data, 'has_spectrum', False),
            'pretrained_embedding': getattr(data, 'pretrained_embedding', None),
            'spec_data': getattr(data, 'spec_data', None),
            'spec_env': getattr(data, 'spec_env', None),
            'instrument_type_idx': getattr(data, 'instrument_type_idx', 2),
            'ionization_type_idx': getattr(data, 'ionization_type_idx', 0),
        }
        spectrum_info.append(info)

        # 创建数据副本并移除质谱属性，避免collate时的张量尺寸冲突
        data_copy = data.clone()

        # 移除所有可能导致collate冲突的质谱相关属性
        attrs_to_remove = ['pretrained_embedding', 'spec_data', 'spec_env', 'has_spectrum',
                          'instrument_type', 'ionization', 'instrument_type_idx', 'ionization_type_idx',
                          'spectrum_index', 'original_mol_id', 'feature_index']
        for attr in attrs_to_remove:
            if hasattr(data_copy, attr):
                delattr(data_copy, attr)

        cleaned_data_list.append(data_copy)

    # 使用清理后的数据进行标准PyG批处理
    follow_batch = ['node_type', 'halfedge_type']
    exclude_keys = ['orig_keys', 'pos_all_confs', 'smiles', 'num_confs', 'i_conf_list',
                   'bond_index', 'bond_type', 'num_bonds', 'num_atoms']

    batched_data = Batch.from_data_list(cleaned_data_list, follow_batch=follow_batch, exclude_keys=exclude_keys)

    # 处理预训练特征和条件信息
    if has_pretrained:
        pretrained_embeddings = []
        has_spectrum_mask = []
        instrument_type_indices = []
        ionization_type_indices = []

        for info in spectrum_info:
            if info['pretrained_embedding'] is not None:
                pretrained_embeddings.append(info['pretrained_embedding'])
                has_spectrum_mask.append(True)
            else:
                # 对于没有预训练特征的样本，添加零向量占位
                pretrained_embeddings.append(torch.zeros(1024))
                has_spectrum_mask.append(False)

            instrument_type_indices.append(info['instrument_type_idx'])
            ionization_type_indices.append(info['ionization_type_idx'])

        # 堆叠预训练特征 [batch_size, 1024]
        batched_data.pretrained_embedding_batch = torch.stack(pretrained_embeddings, dim=0)
        batched_data.has_spectrum_mask = torch.tensor(has_spectrum_mask, dtype=torch.bool)
        batched_data.batch_has_spectrum = any(has_spectrum_mask)

        # 添加条件索引 [batch_size]
        batched_data.instrument_type_idx_batch = torch.tensor(instrument_type_indices, dtype=torch.long)
        batched_data.ionization_type_idx_batch = torch.tensor(ionization_type_indices, dtype=torch.long)
    else:
        batched_data.pretrained_embedding_batch = None
        batched_data.has_spectrum_mask = torch.zeros(len(batch), dtype=torch.bool)
        batched_data.batch_has_spectrum = False

    return batched_data


def collate_mol2d(batch):
    """
    简化的collate函数，用于MSFileDataset的2D分子数据

    处理的数据格式：
    - node_type: 节点类型索引 [num_atoms]
    - halfedge_index: 半边索引 [2, num_halfedges]
    - halfedge_type: 半边类型（标签）[num_halfedges]
    - pretrained_embedding: 预训练特征 [1024]
    - instrument_type_idx: 仪器类型索引
    - ionization_type_idx: 离子化方式索引
    """
    from torch_geometric.data import Batch

    # 保存质谱相关数据
    spectrum_info = []
    cleaned_data_list = []

    for data in batch:
        info = {
            'has_spectrum': getattr(data, 'has_spectrum', False),
            'pretrained_embedding': getattr(data, 'pretrained_embedding', None),
            'instrument_type_idx': getattr(data, 'instrument_type_idx', 2),
            'ionization_type_idx': getattr(data, 'ionization_type_idx', 0),
        }
        spectrum_info.append(info)

        # 创建数据副本并移除质谱属性
        data_copy = data.clone()
        attrs_to_remove = ['pretrained_embedding', 'has_spectrum', 'instrument_type_idx',
                          'ionization_type_idx', 'smiles', 'mol_id']
        for attr in attrs_to_remove:
            if hasattr(data_copy, attr):
                delattr(data_copy, attr)
        cleaned_data_list.append(data_copy)

    # 标准PyG批处理
    follow_batch = ['node_type', 'halfedge_type']
    batched_data = Batch.from_data_list(cleaned_data_list, follow_batch=follow_batch)

    # 处理预训练特征
    has_pretrained = any(info['pretrained_embedding'] is not None for info in spectrum_info)

    if has_pretrained:
        pretrained_embeddings = []
        has_spectrum_mask = []
        instrument_type_indices = []
        ionization_type_indices = []

        for info in spectrum_info:
            if info['pretrained_embedding'] is not None:
                pretrained_embeddings.append(info['pretrained_embedding'])
                has_spectrum_mask.append(True)
            else:
                pretrained_embeddings.append(torch.zeros(1024))
                has_spectrum_mask.append(False)
            instrument_type_indices.append(info['instrument_type_idx'])
            ionization_type_indices.append(info['ionization_type_idx'])

        batched_data.pretrained_embedding_batch = torch.stack(pretrained_embeddings, dim=0)
        batched_data.has_spectrum_mask = torch.tensor(has_spectrum_mask, dtype=torch.bool)
        batched_data.batch_has_spectrum = any(has_spectrum_mask)
        batched_data.instrument_type_idx_batch = torch.tensor(instrument_type_indices, dtype=torch.long)
        batched_data.ionization_type_idx_batch = torch.tensor(ionization_type_indices, dtype=torch.long)
    else:
        batched_data.pretrained_embedding_batch = None
        batched_data.has_spectrum_mask = torch.zeros(len(batch), dtype=torch.bool)
        batched_data.batch_has_spectrum = False
        batched_data.instrument_type_idx_batch = None
        batched_data.ionization_type_idx_batch = None

    return batched_data


class MSFileDataset(Dataset):
    """
    直接从MS文件读取数据的数据集
    不需要SDF文件，直接从MS文件中提取SMILES并生成2D分子图

    数据来源：
    - data/msg_matched_spec_files/Orbitrap/*.ms
    - data/msg_matched_spec_files/QTOF/*.ms
    - data/msg_matched_spec_files/Orbitrap_embedding/batch_embeddings.pkl
    - data/msg_matched_spec_files/QTOF_embedding/batch_embeddings.pkl
    """

    def __init__(self, root, path_dict=None, transform=None, data_subset_ratio=1.0,
                 instrument_type='all', data_split_mode='natms', num_workers=8):
        """
        Args:
            root: 数据根目录
            path_dict: 路径配置（可选）
            transform: 数据转换
            data_subset_ratio: 数据子集比例
            instrument_type: 仪器类型 ('all', 'Orbitrap', 'QTOF', 'NONE')
            data_split_mode: 数据分割模式 ('split', 'natms', 'diffms')
            num_workers: 并行处理的进程数
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.data_subset_ratio = data_subset_ratio
        self.instrument_type = instrument_type
        self.data_split_mode = data_split_mode
        self.num_workers = num_workers

        # MS文件目录
        self.ms_base_dir = os.path.join(root, 'msg_processed')

        # 预训练特征路径
        self.embedding_paths = {
            'Orbitrap': os.path.join(self.ms_base_dir, 'Orbitrap_embedding', 'batch_embeddings.pkl'),
            'QTOF': os.path.join(self.ms_base_dir, 'QTOF_embedding', 'batch_embeddings.pkl'),
            'NONE': os.path.join(self.ms_base_dir, 'NONE_embedding', 'batch_embeddings.pkl')
        }

        # 处理后的LMDB路径（包含数据分割模式）
        suffix = f"msfile_{instrument_type.lower()}_{data_split_mode}"
        if data_subset_ratio < 1.0:
            suffix += f"_{int(data_subset_ratio * 100)}pct"
        self.processed_path = os.path.join(root, f'processed_{suffix}.lmdb')
        self.molid2idx_path = self.processed_path.replace('.lmdb', '_molid2idx.pt')

        self.db = None
        self.keys = None

        # 加载预训练特征
        print(f"[MSFileDataset] 加载预训练特征...")
        self.pretrained_embeddings = {}
        self._load_pretrained_embeddings()

        # 处理数据
        self.smiles2indices_path = self.molid2idx_path.replace('_molid2idx.pt', '_smiles2indices.pt')
        if not os.path.exists(self.processed_path) or not os.path.exists(self.molid2idx_path) or not os.path.exists(self.smiles2indices_path):
            self._process_fast()
            self._precompute_molid2idx()

        self.molid2idx = torch.load(self.molid2idx_path)
        self.smiles2indices = torch.load(self.smiles2indices_path)  # 加载smiles到indices的映射

    def _load_pretrained_embeddings(self):
        """加载预训练质谱特征"""
        if self.instrument_type == 'all':
            instruments_to_load = ['Orbitrap', 'QTOF', 'NONE']
        else:
            instruments_to_load = [self.instrument_type]

        for inst in instruments_to_load:
            path = self.embedding_paths.get(inst)
            if path and os.path.exists(path):
                print(f"  加载 {inst} 预训练特征: {path}")
                with open(path, 'rb') as f:
                    embeddings = pickle.load(f)
                self.pretrained_embeddings[inst] = embeddings
                print(f"    加载了 {len(embeddings)} 个特征")
            else:
                print(f"  警告: {inst} 预训练特征文件不存在: {path}")

    def _parse_ms_file_fast(self, ms_file_path):
        """快速解析MS文件，只提取SMILES和ionization"""
        smiles = None
        ionization = None
        try:
            with open(ms_file_path, 'r') as f:
                for line in f:
                    if line.startswith('#smiles '):
                        smiles = line[8:].strip()
                    elif line.startswith('>ionization '):
                        ionization = line[12:].strip()
                    if smiles and ionization:
                        break
        except:
            pass
        return smiles, ionization

    def _smiles_to_graph(self, smiles):
        """
        从SMILES直接提取2D图结构（无坐标）
        返回: (node_type, edge_index, edge_type, num_atoms) 或 None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.RemoveAllHs(mol)
            num_atoms = mol.GetNumAtoms()
            if num_atoms == 0:
                return None

            # 节点类型（原子序数）
            node_type = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

            # 边索引和边类型
            edge_index = [[], []]
            edge_type = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_index[0].extend([i, j])
                edge_index[1].extend([j, i])
                bt = bond.GetBondType()
                if bt == Chem.BondType.SINGLE:
                    edge_type.extend([1, 1])
                elif bt == Chem.BondType.DOUBLE:
                    edge_type.extend([2, 2])
                elif bt == Chem.BondType.TRIPLE:
                    edge_type.extend([3, 3])
                elif bt == Chem.BondType.AROMATIC:
                    edge_type.extend([4, 4])
                else:
                    edge_type.extend([1, 1])

            # 标准化SMILES
            canonical_smiles = Chem.MolToSmiles(mol)

            return {
                'node_type': np.array(node_type, dtype=np.int64),
                'edge_index': np.array(edge_index, dtype=np.int64),
                'edge_type': np.array(edge_type, dtype=np.int64),
                'num_atoms': num_atoms,
                'smiles': canonical_smiles
            }
        except:
            return None

    def _process_fast(self):
        """快速处理MS文件（单遍扫描，无坐标计算）"""
        from torch_geometric.data import Data

        print(f"[MSFileDataset] 快速处理MS文件...")
        print(f"  仪器类型: {self.instrument_type}")
        print(f"  数据比例: {self.data_subset_ratio:.1%}")

        # 收集所有有预训练特征的MS文件
        if self.instrument_type == 'all':
            instruments_to_process = ['Orbitrap', 'QTOF', 'NONE']
        else:
            instruments_to_process = [self.instrument_type]

        # 构建 emb_key -> (ms_file, inst) 的映射
        entries_to_process = []
        for inst in instruments_to_process:
            if inst not in self.pretrained_embeddings:
                continue
            inst_dir = os.path.join(self.ms_base_dir, inst)
            if not os.path.exists(inst_dir):
                continue

            # 只处理有预训练特征的文件
            for emb_key in self.pretrained_embeddings[inst].keys():
                ms_file = os.path.join(inst_dir, f"{emb_key}.ms")
                if os.path.exists(ms_file):
                    entries_to_process.append((ms_file, inst, emb_key))

        print(f"  找到 {len(entries_to_process)} 个有预训练特征的MS文件")

        # 根据subset_ratio选择文件
        if self.data_subset_ratio < 1.0:
            np.random.seed(2023)
            np.random.shuffle(entries_to_process)
            num_files = int(len(entries_to_process) * self.data_subset_ratio)
            entries_to_process = entries_to_process[:num_files]
            print(f"  选择 {num_files} 个文件进行处理")

        # 创建LMDB数据库
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),
            create=True,
            subdir=False,
            readonly=False,
        )

        num_processed = 0
        num_skipped = 0
        all_elements = set()

        print("\n[INFO] 处理分子数据（单遍扫描）...")
        with db.begin(write=True, buffers=True) as txn:
            for ms_file, inst, emb_key in tqdm(entries_to_process, desc='处理分子'):
                try:
                    # 解析MS文件获取SMILES和ionization
                    smiles, ionization = self._parse_ms_file_fast(ms_file)
                    if not smiles:
                        num_skipped += 1
                        continue

                    # 从SMILES提取图结构
                    graph_data = self._smiles_to_graph(smiles)
                    if graph_data is None:
                        num_skipped += 1
                        continue

                    # 收集原子类型
                    all_elements.update(graph_data['node_type'].tolist())

                    # 获取预训练特征
                    embedding = self.pretrained_embeddings[inst][emb_key]
                    if embedding.ndim == 2:
                        embedding = embedding.squeeze(0)

                    # 创建PyG Data对象
                    data = Data(
                        node_type=torch.from_numpy(graph_data['node_type']),
                        edge_index=torch.from_numpy(graph_data['edge_index']),
                        edge_type=torch.from_numpy(graph_data['edge_type']),
                        num_nodes=graph_data['num_atoms'],
                    )

                    # 添加元数据
                    data.smiles = graph_data['smiles']
                    data.mol_id = f"{inst}_{emb_key}"
                    data.ms_file_path = ms_file  # 添加MS文件路径，用于DiffMS模式的数据划分
                    data.has_spectrum = True
                    data.pretrained_embedding = torch.from_numpy(embedding).float()

                    # 设置条件索引
                    data.instrument_type_idx = INSTRUMENT_TYPES.index(inst) if inst in INSTRUMENT_TYPES else INSTRUMENT_TYPES.index('NONE')
                    ionization_clean = ionization if ionization else '[M+H]+'
                    data.ionization_type_idx = IONIZATION_TYPES.index(ionization_clean) if ionization_clean in IONIZATION_TYPES else 0

                    # 存储到LMDB
                    unique_key = f"{inst}_{emb_key}"
                    txn.put(
                        key=unique_key.encode('utf-8'),
                        value=pickle.dumps(data)
                    )
                    num_processed += 1

                except Exception as e:
                    num_skipped += 1
                    continue

        db.close()

        # 打印统计信息
        atomic_num_to_symbol = {
            1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 33: 'As', 34: 'Se', 35: 'Br', 53: 'I'
        }
        supported_elements = sorted(list(all_elements))
        element_symbols = [atomic_num_to_symbol.get(z, f'Z{z}') for z in supported_elements]

        print(f"\n[INFO] 处理完成:")
        print(f"       成功处理: {num_processed}")
        print(f"       跳过: {num_skipped}")
        print(f"       检测到 {len(supported_elements)} 种原子类型: {element_symbols}")

    def _connect_db(self):
        """建立数据库连接"""
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _precompute_molid2idx(self):
        """预计算mol_id到索引的映射，以及smiles到indices的映射（用于防止数据泄露）"""
        self._connect_db()
        molid2idx = {}
        smiles2indices = {}  # 新增：smiles到indices的映射

        for i, key in enumerate(self.keys):
            data = pickle.loads(self.db.begin().get(key))
            if data is None:
                continue
            mol_id = data.mol_id
            molid2idx[mol_id] = i

            # 记录smiles到indices的映射
            smiles = getattr(data, 'smiles', None)
            if smiles:
                if smiles not in smiles2indices:
                    smiles2indices[smiles] = []
                smiles2indices[smiles].append(i)

        torch.save(molid2idx, self.molid2idx_path)
        # 保存smiles到indices的映射
        smiles2indices_path = self.molid2idx_path.replace('_molid2idx.pt', '_smiles2indices.pt')
        torch.save(smiles2indices, smiles2indices_path)
        self._close_db()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def get_raw(self, idx):
        """获取原始数据（不加载质谱峰），用于数据划分"""
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))

        # 返回包含ms_file_path的字典
        return {
            'ms_file_path': getattr(data, 'ms_file_path', None),
            'mol_id': getattr(data, 'mol_id', None),
            'smiles': getattr(data, 'smiles', None)
        }

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))

        if self.transform is not None:
            data = self.transform(data)
        return data




class SmilesDataset(Dataset):
    """
    纯 SMILES 数据集：用于 formula 模式预训练（无谱图）

    输入文件支持 .csv / .tsv（取 SMILES 列）、.smi / .txt（一行一个 SMILES）。
    流程：标准化 → 去手性 → 去离子（含'.'丢弃）→ InChI 去重
         → 过滤原子不在 atomic_numbers 表内的分子
         → 过滤 num_atoms > max_atoms 的分子
         → 写入 LMDB 缓存
         → 按 split_ratio 切 train/val/test

    输出 Data 字段（与 MSFileDataset 兼容，方便共用 FeaturizeMol2D 与 collate）：
      - node_type, edge_index, edge_type, num_nodes
      - smiles, mol_id
      - has_spectrum = False
      - instrument_type_idx = INSTRUMENT_TYPES.index('NONE')   # 默认 NONE
      - ionization_type_idx = IONIZATION_TYPES.index('[M+H]+') # 默认 [M+H]+
      - 不设 pretrained_embedding（collate 自动占位）
    """

    SMILES_COLUMN_CANDIDATES = ('smiles', 'SMILES', 'canonical_smiles', 'inchi', 'InChI')

    def __init__(self, root, smiles_file, atomic_numbers,
                 data_subset_ratio=1.0, max_atoms=None,
                 split_seed=2026, split_ratio=(0.95, 0.025, 0.025),
                 transform=None):
        super().__init__()
        self.root = root
        self.smiles_file = smiles_file
        self.atomic_numbers = list(atomic_numbers)
        self.atomic_set = set(self.atomic_numbers)
        self.detected_atomic_numbers = list(self.atomic_numbers)  # 与 train.py 接口契约
        self.max_atoms = max_atoms
        self.data_subset_ratio = data_subset_ratio
        self.transform = transform

        os.makedirs(root, exist_ok=True)
        base = os.path.splitext(os.path.basename(smiles_file))[0]
        suffix = f"smiles_{base}_max{max_atoms if max_atoms is not None else 'inf'}_atoms{len(self.atomic_numbers)}"
        if data_subset_ratio < 1.0:
            suffix += f"_{int(data_subset_ratio * 100)}pct"
        self.processed_path = os.path.join(root, f'processed_{suffix}.lmdb')
        self.keys_path = self.processed_path.replace('.lmdb', '_keys.pt')

        self.db = None
        self.keys = None

        # 若 SMILES 文件不存在，自动构建（HMDB+DSSTox+COCONUT+MOSES，去 MSG 测试/验证集泄漏）
        if not os.path.isfile(smiles_file):
            print(f"[SmilesDataset] {smiles_file} 不存在，自动构建预训练 SMILES csv...")
            build_pretrain_smiles_csv(
                output_csv=smiles_file,
                cache_dir=os.path.join(os.path.dirname(smiles_file) or '.', 'raw'),
                msg_split_file=os.path.join(os.path.dirname(os.path.dirname(smiles_file) or '.'), 'split.tsv'),
            )

        if not os.path.exists(self.processed_path) or not os.path.exists(self.keys_path):
            self._process()

        keys_obj = torch.load(self.keys_path)
        if isinstance(keys_obj, dict):
            self.keys = keys_obj['keys']
            self._splits_in_order = keys_obj.get('splits', [None] * len(self.keys))
        else:
            self.keys = keys_obj
            self._splits_in_order = [None] * len(self.keys)

        # 切分
        self.subsets = self._build_subsets(split_seed=split_seed, split_ratio=split_ratio)

    # ------------------- 文件读取 -------------------
    def _read_smiles_file(self):
        """返回 (smiles_list, sources_list, splits_list, mol_ids_list)"""
        path = self.smiles_file
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SMILES 文件不存在: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.csv', '.tsv'):
            sep = '\t' if ext == '.tsv' else ','
            df = pd.read_csv(path, sep=sep)
            col = None
            for cand in self.SMILES_COLUMN_CANDIDATES:
                if cand in df.columns:
                    col = cand
                    break
            if col is None:
                raise ValueError(
                    f"在 {path} 中未找到 SMILES/inchi 列。期望列名之一: {self.SMILES_COLUMN_CANDIDATES}"
                )
            df = df[df[col].notna()].reset_index(drop=True)
            smis = df[col].astype(str).tolist()
            if 'source' in df.columns:
                sources = df['source'].fillna('unknown').astype(str).tolist()
            else:
                sources = ['unknown'] * len(smis)
            if 'split' in df.columns:
                splits = df['split'].astype(str).tolist()
            else:
                splits = [None] * len(smis)
            if 'mol_id' in df.columns:
                mol_ids = df['mol_id'].fillna('').astype(str).tolist()
            else:
                mol_ids = [''] * len(smis)
            return smis, sources, splits, mol_ids
        else:  # .smi / .txt
            with open(path, 'r') as f:
                smis = [line.strip().split()[0] for line in f if line.strip()]
            return smis, ['unknown'] * len(smis), [None] * len(smis), [''] * len(smis)

    @staticmethod
    def _smi_to_canonical(smi):
        """SMILES/InChI → canonical SMILES (去手性, 单组分)"""
        try:
            if smi.startswith('InChI='):
                mol = Chem.MolFromInchi(smi)
            else:
                mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            mol = Chem.RemoveAllHs(mol)
            cano = Chem.MolToSmiles(mol, isomericSmiles=False)
            if '.' in cano:
                return None
            mol = Chem.MolFromSmiles(cano)
            if mol is None or mol.GetNumAtoms() == 0:
                return None
            return cano
        except Exception:
            return None

    def _smiles_to_graph(self, smiles):
        """SMILES → 2D 图（与 MSFileDataset._smiles_to_graph 同构）"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.RemoveAllHs(mol)
            num_atoms = mol.GetNumAtoms()
            if num_atoms == 0 or (self.max_atoms is not None and num_atoms > self.max_atoms):
                return None

            atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
            if not all(z in self.atomic_set for z in atomic_nums):
                return None

            edge_index = [[], []]
            edge_type = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_index[0].extend([i, j])
                edge_index[1].extend([j, i])
                bt = bond.GetBondType()
                if bt == Chem.BondType.SINGLE:
                    edge_type.extend([1, 1])
                elif bt == Chem.BondType.DOUBLE:
                    edge_type.extend([2, 2])
                elif bt == Chem.BondType.TRIPLE:
                    edge_type.extend([3, 3])
                elif bt == Chem.BondType.AROMATIC:
                    edge_type.extend([4, 4])
                else:
                    edge_type.extend([1, 1])

            return {
                'node_type': np.array(atomic_nums, dtype=np.int64),
                'edge_index': np.array(edge_index, dtype=np.int64),
                'edge_type': np.array(edge_type, dtype=np.int64),
                'num_atoms': num_atoms,
                'smiles': smiles,
            }
        except Exception:
            return None

    # ------------------- 处理 -------------------
    def _process(self):
        from torch_geometric.data import Data

        print(f"[SmilesDataset] 处理 SMILES 文件: {self.smiles_file}")
        raw_smiles, raw_sources, raw_splits, raw_mol_ids = self._read_smiles_file()
        print(f"  原始条目数: {len(raw_smiles)}")

        # csv 已经在 build_pretrain_smiles_csv 阶段做了 InChI 跨源去重，这里不再重复去重
        canonical_list = list(zip(raw_smiles, raw_sources, raw_splits, raw_mol_ids))

        if self.data_subset_ratio < 1.0:
            np.random.seed(2026)
            rng_idx = np.arange(len(canonical_list))
            np.random.shuffle(rng_idx)
            n_keep = int(len(canonical_list) * self.data_subset_ratio)
            canonical_list = [canonical_list[k] for k in rng_idx[:n_keep]]
            print(f"  按比例 {self.data_subset_ratio:.1%} 取: {n_keep}")

        # 写 LMDB
        none_idx = INSTRUMENT_TYPES.index('NONE') if 'NONE' in INSTRUMENT_TYPES else 0
        mhplus_idx = IONIZATION_TYPES.index('[M+H]+') if '[M+H]+' in IONIZATION_TYPES else 0

        os.makedirs(os.path.dirname(self.processed_path) or '.', exist_ok=True)
        db = lmdb.open(
            self.processed_path,
            map_size=200 * (1024 ** 3),  # 200 GB（稀疏分配，仅占实际写入量）
            create=True,
            subdir=False,
            readonly=False,
        )
        keys = []
        splits_in_order = []
        num_kept = 0
        num_skipped_atom = 0
        num_skipped_size = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (smi, src, sp, mid) in enumerate(tqdm(canonical_list, desc='  写入LMDB')):
                graph = self._smiles_to_graph(smi)
                if graph is None:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None or mol.GetNumAtoms() == 0 or (self.max_atoms is not None and mol.GetNumAtoms() > self.max_atoms):
                        num_skipped_size += 1
                    else:
                        num_skipped_atom += 1
                    continue
                data = Data(
                    node_type=torch.from_numpy(graph['node_type']),
                    edge_index=torch.from_numpy(graph['edge_index']),
                    edge_type=torch.from_numpy(graph['edge_type']),
                    num_nodes=int(graph['num_atoms']),
                )
                data.smiles = graph['smiles']
                # MSG 用 GymID 作 mol_id；外部源用 'smiles_<i>'
                data.mol_id = str(mid) if mid else f"smiles_{i:08d}"
                data.source = str(src)
                data.split = (str(sp) if sp is not None and str(sp) != 'nan' else None)
                data.has_spectrum = False
                data.instrument_type_idx = none_idx
                data.ionization_type_idx = mhplus_idx
                key = f"{i:08d}".encode()
                txn.put(key, pickle.dumps(data))
                keys.append(key)
                splits_in_order.append(data.split)
                num_kept += 1
        db.sync()
        db.close()

        torch.save({'keys': keys, 'splits': splits_in_order}, self.keys_path)
        print(f"[SmilesDataset] 完成: 保留 {num_kept}, 跳过(原子表外) {num_skipped_atom}, 跳过(大小) {num_skipped_size}")

    # ------------------- LMDB 连接 -------------------
    def _connect_db(self):
        self.db = lmdb.open(
            self.processed_path,
            map_size=200 * (1024 ** 3),
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __len__(self):
        if self.keys is None:
            self.keys = torch.load(self.keys_path)
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        if self.transform is not None:
            data = self.transform(data)
        return data

    # ------------------- 切分 -------------------
    def _build_subsets(self, split_seed=2026, split_ratio=(0.95, 0.025, 0.025)):
        n = len(self.keys)
        # 优先按 csv 中的 split 列（MSG train/val/test 已落地，其它源默认 train）
        if any(s is not None for s in self._splits_in_order):
            train_idx, val_idx, test_idx = [], [], []
            for i, sp in enumerate(self._splits_in_order):
                if sp == 'val':
                    val_idx.append(i)
                elif sp == 'test':
                    test_idx.append(i)
                else:  # 'train' 或缺失
                    train_idx.append(i)
            print(f"[SmilesDataset] split (按 csv 列): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        else:
            rng = np.random.default_rng(split_seed)
            perm = rng.permutation(n)
            r_train, r_val, _ = split_ratio
            n_train = int(n * r_train)
            n_val = int(n * r_val)
            train_idx = perm[:n_train].tolist()
            val_idx = perm[n_train:n_train + n_val].tolist()
            test_idx = perm[n_train + n_val:].tolist()
            print(f"[SmilesDataset] split (随机比例): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        return {
            'train': Subset(self, train_idx),
            'val':   Subset(self, val_idx),
            'test':  Subset(self, test_idx),
        }


# =====================================================================
# 预训练 SMILES csv 构建工具
# 参考 DualLGD 的 build_fp2mol_datasets.py 思路：
#   下载 HMDB / DSSTox / COCONUT / MOSES 四个分子库 → 标准化（去手性）
#   → InChI 去重 → 排除 MSG 测试/验证集分子（防泄漏） → 写出
#   两列 csv: smiles, source（来源数据集名）
# 默认下载 URL 与 DualLGD 保持一致，用户可通过参数覆盖。
# =====================================================================

DEFAULT_PRETRAIN_SOURCES = {
    'hmdb':    'https://hmdb.ca/system/downloads/current/structures.zip',
    'dsstox':  'https://clowder.edap-cluster.com/api/files/6616d8d7e4b063812d70fc95/blob',
    'coconut': 'https://coconut.s3.uni-jena.de/prod/downloads/2025-03/coconut_csv-03-2025.zip',
    'moses':   'https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv',
}

DEFAULT_FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P', 'Br', 'I', 'B', 'Si', 'Se'}


def _download_file(url, dst):
    """下载文件到 dst，带进度条；已存在跳过"""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"  已存在跳过下载: {dst} ({size_mb:.1f} MB)")
        return dst
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    print(f"  下载 {url} -> {dst}")
    try:
        import urllib.request
        # HEAD 拿 content-length 给进度条
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get('Content-Length') or 0)
            chunk = 1 << 20  # 1 MB
            with open(dst, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True, unit_divisor=1024,
                desc=f"  下载 {os.path.basename(dst)}", leave=False,
            ) as pbar:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    pbar.update(len(buf))
        return dst
    except Exception as e:
        print(f"  [警告] 下载失败 {url}: {e}")
        return None


def _filter_mol_for_pretrain(mol, max_mw=1500.0, allowed_atoms=None):
    """与 DualLGD filter() 等价：单组分、不带电、MW<1500、原子白名单内"""
    if mol is None:
        return False
    try:
        from rdkit.Chem import Descriptors
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        if '.' in smi:
            return False
        m2 = Chem.MolFromSmiles(smi)
        if m2 is None:
            return False
        if Descriptors.MolWt(m2) >= max_mw:
            return False
        for atom in m2.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return False
            if allowed_atoms is not None and atom.GetSymbol() not in allowed_atoms:
                return False
        return True
    except Exception:
        return False


def _read_sdf_smiles(sdf_path):
    """从 SDF 中按 '> <SMILES>' 字段提取（兼容 HMDB 的导出格式）"""
    smis = []
    if not os.path.isfile(sdf_path):
        return smis
    with open(sdf_path, 'r', errors='ignore') as f:
        app = False
        for line in f:
            if app:
                smis.append(line.strip())
                app = False
            if line.startswith('> <SMILES>'):
                app = True
    return smis


def _extract_zip(zip_path, extract_to):
    import zipfile
    if not os.path.isfile(zip_path):
        return
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)


def _collect_msg_smiles_by_split(msg_split_file):
    """
    从 MSG split.tsv 收集每个 split（train/val/test）的 SMILES + InChI + GymID。
    每个 .ms 文件视为一条样本（即每张谱图一个样本，与 formula+dreams 阶段的
    MSFileDataset 行为一致），**不做 InChI 去重**。

    Returns:
        smiles_by_split: {'train': [(smiles, inchi, gymid), ...], 'val': [...], 'test': [...]}
        若 split.tsv 或 msg_processed/ 不存在则返回空 dict。
    """
    out = {'train': [], 'val': [], 'test': []}
    if not msg_split_file or not os.path.isfile(msg_split_file):
        print(f"  [警告] 未找到 MSG split 文件: {msg_split_file}，跳过 MSG 合并/排除")
        return out

    try:
        df = pd.read_csv(msg_split_file, sep='\t')
    except Exception as e:
        print(f"  [警告] 读取 MSG split 失败: {e}")
        return out

    gymid_to_split = dict(zip(df['name'].astype(str), df['split'].astype(str)))

    base_root = os.path.dirname(msg_split_file)
    msg_processed_dir = os.path.join(base_root, 'msg_processed')
    if not os.path.isdir(msg_processed_dir):
        print(f"  [警告] 未找到 {msg_processed_dir}/，跳过 MSG 合并")
        return out

    # 收集所有 .ms 文件
    ms_files = []
    for inst in os.listdir(msg_processed_dir):
        inst_dir = os.path.join(msg_processed_dir, inst)
        if not os.path.isdir(inst_dir) or inst.endswith('_embedding'):
            continue
        for fn in os.listdir(inst_dir):
            if fn.endswith('.ms'):
                ms_files.append(os.path.join(inst_dir, fn))

    for ms_path in tqdm(ms_files, desc='  扫描 MSG .ms', leave=False):
        gymid = os.path.basename(ms_path)[:-3]
        split_label = gymid_to_split.get(gymid)
        if split_label not in out:
            continue
        try:
            with open(ms_path, 'r') as f:
                for line in f:
                    if line.startswith('#smiles '):
                        smi = line[8:].strip()
                        mol = Chem.MolFromSmiles(smi)
                        if mol is None:
                            break
                        try:
                            cano = Chem.MolToSmiles(mol, isomericSmiles=False)
                            inchi = Chem.MolToInchi(Chem.MolFromSmiles(cano))
                        except Exception:
                            break
                        if not inchi:
                            break
                        out[split_label].append((cano, inchi, gymid))  # 不去重，每谱一条
                        break
        except Exception:
            continue
    print(f"  MSG split 收集（每谱一条）: train={len(out['train'])}, val={len(out['val'])}, test={len(out['test'])}")
    return out


def build_pretrain_smiles_csv(
    output_csv,
    cache_dir=None,
    msg_split_file=None,
    sources=None,
    allowed_atoms=None,
    max_mw=1500.0,
    max_atoms=None,
):
    """
    构建预训练 SMILES csv（三列：smiles, source, split）。

    流程：
      1) 读 MSG split.tsv：把 train/val/test 分子的 SMILES 全部纳入（split 直接落地为对应标签）
      2) 下载 HMDB / DSSTox / COCONUT / MOSES → 解析 → RDKit 标准化（去手性、单组分、不带电、MW<max_mw）→ 原子白名单
      3) 全程按 InChI 去重；任何与 MSG val/test 同一 InChI 的预训练分子被剔除（防泄漏）
      4) 输出 csv: smiles, source, split。MSG 之外的分子统一 split=train（用作纯预训练扩充）。

    参数:
        output_csv: 最终 csv 路径
        cache_dir: 原始文件下载/解压目录（默认 output_csv 同级 raw/）
        msg_split_file: MSG split.tsv 路径（用于合并/防泄漏；可为 None）
        sources: dict {source_name: download_url}，None 时用 DEFAULT_PRETRAIN_SOURCES
        allowed_atoms: 原子符号集合，None 时用 DEFAULT_FILTER_ATOMS
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(output_csv) or '.', 'raw')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

    sources = sources or DEFAULT_PRETRAIN_SOURCES
    allowed_atoms = allowed_atoms or DEFAULT_FILTER_ATOMS

    print("=" * 60)
    print("[build_pretrain_smiles_csv] 开始构建预训练 SMILES csv")
    print(f"  缓存目录: {cache_dir}")
    print(f"  输出: {output_csv}")
    print("=" * 60)

    # 1) 收集 MSG 三个 split 的 SMILES（每谱一条，不去重）
    msg_by_split = _collect_msg_smiles_by_split(msg_split_file) if msg_split_file else {
        'train': [], 'val': [], 'test': []
    }
    excluded_inchis = set(inchi for _, inchi, _ in msg_by_split.get('val', []))
    excluded_inchis.update(inchi for _, inchi, _ in msg_by_split.get('test', []))
    print(f"  MSG val+test 防泄漏 InChI 数（去重后）: {len(excluded_inchis)}")

    # 外部源跨源 InChI 去重；MSG 不参与跨源去重（每谱一条），但其 InChI 加入 seen
    # 以防外部源出现同一 InChI 时重复入库
    seen_inchi = set()
    rows = []  # (smiles, source, split, mol_id)

    for split_label in ('train', 'val', 'test'):
        for cano, inchi, gymid in msg_by_split.get(split_label, []):
            rows.append((cano, 'msg', split_label, gymid))
            if inchi:
                seen_inchi.add(inchi)
    print(f"  MSG 已并入（每谱一条）: {len(rows)} 条")

    def _ingest_smiles_iterable(iter_smis, source_name, total=None):
        kept = 0
        skipped = 0
        pbar = tqdm(iter_smis, desc=f'  {source_name} 清洗', total=total, leave=False)
        for smi in pbar:
            if not smi or not isinstance(smi, str):
                skipped += 1
                continue
            try:
                if smi.startswith('InChI='):
                    mol = Chem.MolFromInchi(smi)
                else:
                    mol = Chem.MolFromSmiles(smi)
            except Exception:
                skipped += 1
                continue
            if mol is None:
                skipped += 1
                continue
            try:
                mol = Chem.RemoveAllHs(mol)
            except Exception:
                skipped += 1
                continue
            if mol.GetNumAtoms() == 0 or (max_atoms is not None and mol.GetNumAtoms() > max_atoms):
                skipped += 1
                continue
            try:
                if not _filter_mol_for_pretrain(mol, max_mw=max_mw, allowed_atoms=allowed_atoms):
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue
            try:
                cano = Chem.MolToSmiles(mol, isomericSmiles=False)
                inchi = Chem.MolToInchi(Chem.MolFromSmiles(cano))
            except Exception:
                skipped += 1
                continue
            if not inchi or inchi in seen_inchi:
                skipped += 1
                continue
            if inchi in excluded_inchis:
                skipped += 1
                continue
            seen_inchi.add(inchi)
            rows.append((cano, source_name, 'train', ''))  # 外部源无 mol_id
            kept += 1
            if (kept + skipped) % 5000 == 0:
                pbar.set_postfix(kept=kept, skipped=skipped)
        return kept, skipped

    # 2) HMDB（用户手动放置 raw/hmdb.sdf 或 raw/structures.sdf）
    if 'hmdb' in sources:
        print("\n[HMDB]")
        sdf_candidates = (
            glob.glob(os.path.join(cache_dir, 'hmdb*.sdf'))
            + glob.glob(os.path.join(cache_dir, 'structures*.sdf'))
        )
        if not sdf_candidates:
            print("  [跳过] 未找到 HMDB sdf 文件。请手动下载并放到 raw/ 下：")
            print(f"    {sources['hmdb']}")
            print(f"    期望命名: {cache_dir}/hmdb.sdf 或 {cache_dir}/structures.sdf")
        else:
            smis = []
            for sdf in sdf_candidates:
                smis.extend(_read_sdf_smiles(sdf))
            print(f"  [HMDB] 原始分子数: {len(smis)}")
            before = len(rows)
            kept, skipped = _ingest_smiles_iterable(smis, 'hmdb', total=len(smis))
            print(f"  [HMDB] 保留 {kept} / 跳过 {skipped} (csv 累计 +{len(rows) - before})")

    # 3) DSSTox（用户手动放 raw/DSSTox/DSSToxDump*.xlsx）
    if 'dsstox' in sources:
        print("\n[DSSTox]")
        xlsx_files = (
            glob.glob(os.path.join(cache_dir, 'DSSToxDump*.xlsx'))
            + glob.glob(os.path.join(cache_dir, 'DSSTox', 'DSSToxDump*.xlsx'))
            + glob.glob(os.path.join(cache_dir, 'DSSTox', '*.xlsx'))
        )
        if not xlsx_files:
            print("  [跳过] 未找到 DSSTox xlsx 文件。请手动下载并放到 raw/ 下：")
            print(f"    {sources['dsstox']}")
            print(f"    期望命名: {cache_dir}/DSSTox/DSSToxDump*.xlsx")
        else:
            dss_smis = []
            for fp in tqdm(xlsx_files, desc='  DSSTox xlsx 加载'):
                try:
                    df = pd.read_excel(fp)
                    if 'SMILES' in df.columns:
                        dss_smis.extend(df['SMILES'].dropna().astype(str).tolist())
                except Exception as e:
                    print(f"  [警告] 加载 {fp} 失败: {e}")
            print(f"  [DSSTox] 原始分子数: {len(dss_smis)}")
            before = len(rows)
            kept, skipped = _ingest_smiles_iterable(dss_smis, 'dsstox', total=len(dss_smis))
            print(f"  [DSSTox] 保留 {kept} / 跳过 {skipped} (csv 累计 +{len(rows) - before})")

    # 4) COCONUT（用户手动放 raw/coconut_csv*.csv，例如 coconut_csv_lite-06-2026.csv）
    if 'coconut' in sources:
        print("\n[COCONUT]")
        csv_candidates = glob.glob(os.path.join(cache_dir, 'coconut_csv*.csv'))
        if not csv_candidates:
            print("  [跳过] 未找到 COCONUT csv。请手动下载并放到 raw/ 下：")
            print(f"    {sources['coconut']}")
            print(f"    期望命名: {cache_dir}/coconut_csv*.csv")
        else:
            coconut_smis = []
            for fp in csv_candidates:
                try:
                    df = pd.read_csv(fp)
                    col = None
                    for c in ('canonical_smiles', 'SMILES', 'smiles'):
                        if c in df.columns:
                            col = c
                            break
                    if col is None:
                        print(f"  [警告] {fp} 无 SMILES/smiles/canonical_smiles 列，跳过")
                        continue
                    coconut_smis.extend(df[col].dropna().astype(str).tolist())
                except Exception as e:
                    print(f"  [警告] 加载 {fp} 失败: {e}")
            print(f"  [COCONUT] 原始分子数: {len(coconut_smis)}")
            before = len(rows)
            kept, skipped = _ingest_smiles_iterable(coconut_smis, 'coconut', total=len(coconut_smis))
            print(f"  [COCONUT] 保留 {kept} / 跳过 {skipped} (csv 累计 +{len(rows) - before})")

    # 5) MOSES（用户手动放 raw/moses.csv 或 raw/dataset_v1.csv）
    if 'moses' in sources:
        print("\n[MOSES]")
        moses_paths = [
            os.path.join(cache_dir, 'moses.csv'),
            os.path.join(cache_dir, 'dataset_v1.csv'),
        ]
        moses_path = next((p for p in moses_paths if os.path.isfile(p)), None)
        if moses_path is None:
            print("  [跳过] 未找到 MOSES csv。请手动下载并放到 raw/ 下：")
            print(f"    {sources['moses']}")
            print(f"    期望命名: {cache_dir}/moses.csv 或 {cache_dir}/dataset_v1.csv")
        else:
            try:
                df = pd.read_csv(moses_path)
                col = 'SMILES' if 'SMILES' in df.columns else ('smiles' if 'smiles' in df.columns else None)
                moses_smis = df[col].dropna().astype(str).tolist() if col is not None else []
            except Exception as e:
                print(f"  [警告] 加载 MOSES 失败: {e}")
                moses_smis = []
            print(f"  [MOSES] 原始分子数 (来自 {os.path.basename(moses_path)}): {len(moses_smis)}")
            before = len(rows)
            kept, skipped = _ingest_smiles_iterable(moses_smis, 'moses', total=len(moses_smis))
            print(f"  [MOSES] 保留 {kept} / 跳过 {skipped} (csv 累计 +{len(rows) - before})")

    # 6) 写出
    print("\n[写出]")
    if not rows:
        raise RuntimeError("预训练数据集为空，请检查下载是否成功")
    df_out = pd.DataFrame(rows, columns=['smiles', 'source', 'split', 'mol_id'])
    df_out.to_csv(output_csv, index=False)
    print(f"  总计 {len(df_out)} 条 ⇒ {output_csv}")
    print(f"  各来源分布: {df_out['source'].value_counts().to_dict()}")
    print(f"  各 split 分布: {df_out['split'].value_counts().to_dict()}")
    return output_csv


# ============================================================================
# DiffMSMSGDataset：DeniMS 格式输入（fragment formula 序列 + dense X/E/y/node_mask）
# 数据源：DiffMS 预处理过的 MSG（spec_files/.ms + subformulae/default_subformulae/.json
#                                   + labels.tsv + split.tsv）
# 用于 align / ms2mol 阶段（取代旧 MSFileDataset 的 raw DreaMS 路径）
# ============================================================================

# DeniMS 9 元素表（与 ckpt 完全对齐）
_DENIMS_ELEMENTS = ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I"]
_DENIMS_PRECURSOR = {
    '[M+H]+':  0,   # DeniMS 原生支持
    '[M-H]-':  1,   # DeniMS 原生支持
    '[M+Na]+': 0,   # 临时方案：当作 [M+H]+ 处理（DeniMS ckpt 没见过，但 MSG 里 35867 条不舍）
}

# DiffMS atom_types（与 DeniMS graph_encoder X[B,N,11] 对齐：去掉 H 列后是 11 维）
_DIFFMS_ATOM_TYPES = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Si': 5, 'P': 6, 'S': 7,
                       'Cl': 8, 'Br': 9, 'I': 10, 'H': 11}


class _DenIMSPositionalEncoding:
    """DeniMS 16-d 元素计数 sinusoidal 编码（precompute 0~149）"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.div_term = torch.exp(
            torch.arange(0, 16, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / 8)
        )
        self.pos_dict = {}
        for i in range(150):
            enc = torch.zeros(16)
            enc[0::2] = torch.sin(i * self.div_term)
            enc[1::2] = torch.cos(i * self.div_term)
            self.pos_dict[i] = enc

    def encode(self, n_atoms):
        return self.pos_dict[min(n_atoms, 149)]


def _formula_str_to_array(formula_str):
    """'C16H17NO4' → np.array([H, C, N, O, F, S, Cl, Br, I]) 9 元素计数"""
    counts = np.zeros(len(_DENIMS_ELEMENTS), dtype=int)
    for sym, num in re.findall(r'([A-Z][a-z]?)(\d*)', formula_str):
        if sym in _DENIMS_ELEMENTS:
            counts[_DENIMS_ELEMENTS.index(sym)] += int(num) if num else 1
    return counts


def _encode_peaks_to_formula_array(per_peak_formulas, max_peaks=128):
    """每峰 → 9 元素 × 16-d sinusoidal → padded [max_peaks, 144], mask [max_peaks+1]"""
    pos_enc = _DenIMSPositionalEncoding()
    total_dim = len(_DENIMS_ELEMENTS) * 16   # 144
    tensors = []
    for f in per_peak_formulas[:max_peaks]:
        arr = _formula_str_to_array(f)
        t = torch.zeros(total_dim)
        for i, v in enumerate(arr):
            t[i*16:(i+1)*16] = pos_enc.encode(int(v))
        tensors.append(t)
    n = len(tensors)
    padded = torch.zeros((max_peaks, total_dim), dtype=torch.float32)
    if n > 0:
        padded[:n] = torch.stack(tensors)
    mask = torch.ones(max_peaks + 1, dtype=torch.bool)
    mask[:n + 1] = 0
    return padded, mask


def _smi_to_denims_graph(smiles):
    """SMILES → (X[N,11], edge_index[2,M], edge_attr[M,5])  与 DeniMS 完全一致

    参考 DeniMS/Preprocessing/generate_graph_dict.py:mol_to_graph
    """
    from rdkit.Chem.rdchem import BondType as BT
    from torch_geometric.utils import subgraph
    BOND_TYPES_DEN = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    N = mol.GetNumAtoms()
    type_idx = [_DIFFMS_ATOM_TYPES.get(a.GetSymbol(), -1) for a in mol.GetAtoms()]
    if -1 in type_idx:
        return None
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [s, e]
        col += [e, s]
        edge_type += 2 * [BOND_TYPES_DEN[bond.GetBondType()] + 1]
    if not row:
        return None
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type_t = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = torch.nn.functional.one_hot(edge_type_t, num_classes=len(BOND_TYPES_DEN) + 1).float()
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    x = torch.nn.functional.one_hot(torch.tensor(type_idx), num_classes=len(_DIFFMS_ATOM_TYPES)).float()
    type_idx_t = torch.tensor(type_idx).long()
    to_keep = type_idx_t <= 11
    edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr,
                                      relabel_nodes=True, num_nodes=len(to_keep))
    x = x[to_keep][:, :-1]   # 删 H 列 → x [N, 11]
    if x.size(0) == 0:
        return None
    return x, edge_index, edge_attr


class DiffMSMSGDataset(Dataset):
    """读 DiffMS 预处理过的 MSG 数据，输出 DeniMS 格式。

    每个样本 (Data 对象) 字段：
        - smiles                : str（原 SMILES，align 阶段 multi_positive 用）
        - mol_id                : str（spec_id，如 "MassSpecGymID0000123"）
        - has_spectrum          : bool（恒 True）
        - spec_sos              : [1, 13]   precursor 2 + collision_energy 11 (NCE=0 兜底)
        - spec_formula_array    : [128, 144]
        - spec_mask             : [129] bool
        - dense_X               : [N, 11]
        - dense_edge_index      : [2, M]
        - dense_edge_attr       : [M, 5]
        - num_nodes             : int N
        - instrument_type_idx   : int
        - ionization_type_idx   : int
        - 兼容字段（防止 collate 失败）：
            node_type, edge_index, edge_type
    """

    def __init__(self, root, path_dict=None, transform=None,
                 data_subset_ratio=1.0, instrument_type='all',
                 data_split_mode='split', num_workers=8,
                 max_peaks=128):
        super().__init__()
        self.root = root
        self.transform = transform
        self.data_subset_ratio = data_subset_ratio
        self.instrument_type = instrument_type
        self.data_split_mode = data_split_mode
        self.max_peaks = max_peaks

        # 路径
        self.spec_files_dir = os.path.join(root, 'spec_files')
        self.subformulae_dir = os.path.join(root, 'subformulae', 'default_subformulae')
        self.labels_path = os.path.join(root, 'labels.tsv')
        self.split_path = os.path.join(root, 'split.tsv')

        for p in (self.spec_files_dir, self.subformulae_dir, self.labels_path, self.split_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"DiffMSMSGDataset 路径缺失: {p}")

        print(f"[DiffMSMSGDataset] 加载 labels + split ...")
        labels_df = pd.read_csv(self.labels_path, sep='\t')
        split_df = pd.read_csv(self.split_path, sep='\t')
        spec2split = dict(zip(split_df['name'], split_df['split']))
        labels_df['split'] = labels_df['spec'].map(spec2split)

        # 仅保留 [M+H]+ / [M-H]- 与可识别的 instrument
        labels_df = labels_df[labels_df['ionization'].isin(_DENIMS_PRECURSOR.keys())].copy()
        # 用于跨阶段的 instrument 索引（与 INSTRUMENT_TYPES 对应；NONE 兜底）
        from models.model import INSTRUMENT_TYPES
        labels_df['instrument_type_idx'] = labels_df['instrument'].map(
            lambda x: INSTRUMENT_TYPES.index(x) if x in INSTRUMENT_TYPES else INSTRUMENT_TYPES.index('NONE')
        )
        # ionization 仍记录到 idx，但 ms_encoder 实际用 sos one-hot；这里仅作 BFN 主干 condition
        labels_df['ionization_type_idx'] = labels_df['ionization'].map(
            lambda x: 0 if x == '[M+H]+' else (1 if x == '[M+Na]+' else 0)
        )

        # 抽样
        if data_subset_ratio < 1.0:
            n_keep = int(len(labels_df) * data_subset_ratio)
            labels_df = labels_df.sample(n=n_keep, random_state=2026).reset_index(drop=True)
        else:
            labels_df = labels_df.reset_index(drop=True)

        self.labels_df = labels_df
        print(f"[DiffMSMSGDataset] 总样本: {len(labels_df)}, "
              f"unique smiles: {labels_df['smiles'].nunique()}")
        print(f"[DiffMSMSGDataset] split 分布: {labels_df['split'].value_counts().to_dict()}")

        # 缓存 SMILES → (X, edge_index, edge_attr)（每个 SMILES 只算一次）
        self._graph_cache = {}

        # 切分（按 'split' 列直接组织 Subset）
        self.subsets = {}
        for sp in ('train', 'val', 'test'):
            mask = (labels_df['split'] == sp).values
            indices = np.where(mask)[0].tolist()
            self.subsets[sp] = Subset(self, indices)

        self.detected_atomic_numbers = [5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        spec_id = row['spec']
        smiles = row['smiles']

        # ---- 1. 读 subformulae JSON → spec_sos / spec_formula_array / spec_mask ----
        subform_path = os.path.join(self.subformulae_dir, f'{spec_id}.json')
        per_peak_formulas = []
        if os.path.exists(subform_path):
            try:
                with open(subform_path) as f:
                    data = json.load(f)
            except Exception:
                data = None
            if data is not None:
                output_tbl = data.get('output_tbl')
                if output_tbl is not None:
                    per_peak_formulas = output_tbl.get('formula', []) or []

        formula_array, spec_mask = _encode_peaks_to_formula_array(
            per_peak_formulas, max_peaks=self.max_peaks
        )
        # sos = precursor 2 + energy 11
        precursor_oh = torch.zeros(2)
        precursor_oh[_DENIMS_PRECURSOR[row['ionization']]] = 1.0
        energy_oh = torch.zeros(11)
        energy_oh[0] = 1.0  # MSG 没 NCE 信号，用 0 兜底
        sos = torch.cat([precursor_oh, energy_oh], dim=0).view(1, -1)  # [1, 13]

        # ---- 2. SMILES → DeniMS graph ----
        if smiles in self._graph_cache:
            x, edge_index, edge_attr = self._graph_cache[smiles]
        else:
            res = _smi_to_denims_graph(smiles)
            if res is None:
                # 出问题：构造空图
                x = torch.zeros(1, 11)
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                edge_attr = torch.zeros(0, 5)
            else:
                x, edge_index, edge_attr = res
            self._graph_cache[smiles] = (x, edge_index, edge_attr)

        # ---- 3. 组成 PyG Data ----
        from torch_geometric.data import Data as PygData
        d = PygData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=int(x.size(0)),
        )
        d.smiles = smiles
        d.mol_id = spec_id
        d.has_spectrum = True
        d.spec_sos = sos
        d.spec_formula_array = formula_array
        d.spec_mask_per_sample = spec_mask
        d.instrument_type_idx = int(row['instrument_type_idx'])
        d.ionization_type_idx = int(row['ionization_type_idx'])

        if self.transform is not None:
            d = self.transform(d)
        return d


# ============================================================================
# Stage 1 输出缓存：Zmol / Zms 一次性构建到磁盘
# graph2mol/ms2mol 训练时直接读，不实时跑 encoder（与 DeniMS 默认 finetune_ms_encoder=False 等价）
# ============================================================================

# Cache version：变了就重建。改 align ckpt / 改 encoder 架构 / 改维度都要 bump
_CACHE_VERSION = 'v1'


def _cache_paths(cache_dir, version=_CACHE_VERSION):
    return {
        'zmol': os.path.join(cache_dir, f'zmol_{version}.pt'),
        'zms':  os.path.join(cache_dir, f'zms_{version}.pt'),
        'meta': os.path.join(cache_dir, f'meta_{version}.json'),
    }


def build_zmol_cache(align_ckpt_path, smiles_list, cache_path, device='cpu',
                      batch_size=64, dtype=torch.float16):
    """对一批 SMILES 用 align ckpt 跑 graph_encoder → 保存 dict[smi] = emb[512]

    Args:
        align_ckpt_path: DeniMS Encoder_Contrastive_FragHub.pth 路径
        smiles_list: 唯一 SMILES 列表
        cache_path: 输出路径，例如 data/cache/zmol_v1.pt
        device: 推理 device
        batch_size: 编码 batch
        dtype: 缓存精度 (fp16 节省一半空间)
    """
    from models.model import FLASH
    from utils.transforms import _BFN2DIFFMS
    from easydict import EasyDict

    print(f'[zmol cache] 构建中: {len(smiles_list)} unique SMILES → {cache_path}')

    # 构造 align 模型并加载 ckpt
    cfg = EasyDict({
        'stage': 'align',
        'contrastive_dim': 512,
        'noise_scale': 0.0,
        'condition_embedding': {'embed_dim': 256},
        'gat': {'hidden_dim': 256, 'num_layers': 4, 'dropout': 0.0},
        'flow': {'n_timesteps': 100},
        'flash': {'beta1': 3.0},
        'graph_encoder': {'n_layers': 4},
        'ms_encoder': {'dim_sos': 13, 'dim_formula': 144, 'hidden_dim': 512,
                       'num_transformer_layers': 3, 'nhead': 8,
                       'dropout': 0.0, 'input_dropout': 0.0, 'max_len': 129},
        'node_dim': 256,
        'contrastive_temperature_init': 30.0,
    })
    m = FLASH(cfg, num_node_types=14, num_edge_types=5).to(device)
    try:
        sd = torch.load(align_ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(align_ckpt_path, map_location=device)
    sd = sd['model'] if 'model' in sd else sd
    m.load_state_dict(sd, strict=False)
    m.eval()

    # SMILES → DeniMS PyG Data（RDKit 解析，每条几毫秒，3.3M 条约 2-3 小时）
    from torch_geometric.data import Data as PygData
    valid_smis = []
    pyg_data_list = []
    for smi in tqdm(smiles_list, desc='  RDKit 解析 SMILES → graph', unit='mol', mininterval=2.0):
        res = _smi_to_denims_graph(smi)
        if res is None:
            continue
        x, edge_index, edge_attr = res
        pyg_data_list.append(PygData(x=x, edge_index=edge_index, edge_attr=edge_attr))
        valid_smis.append(smi)
    print(f'  RDKit 成功: {len(valid_smis)} / {len(smiles_list)}')

    # batch 推理
    cache = {}
    n = len(valid_smis)
    n_batches = (n + batch_size - 1) // batch_size
    with torch.no_grad():
        for s in tqdm(range(0, n, batch_size), total=n_batches, desc='  zmol encoder'):
            chunk_smis = valid_smis[s:s+batch_size]
            chunk_data = pyg_data_list[s:s+batch_size]
            B = len(chunk_data)
            N_max = max(d.x.size(0) for d in chunk_data)
            X = torch.zeros(B, N_max, 11, device=device)
            E = torch.zeros(B, N_max, N_max, 5, device=device)
            y = torch.ones(B, 1, device=device)
            node_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)
            for i, d in enumerate(chunk_data):
                nn = d.x.size(0)
                X[i, :nn] = d.x.to(device)
                node_mask[i, :nn] = True
                if d.edge_index.size(1) > 0:
                    src, dst = d.edge_index
                    E[i, src.to(device), dst.to(device)] = d.edge_attr.to(device)
                eb = E[i, :nn, :nn]
                zm = (eb == 0).all(dim=-1)
                eb[zm] = torch.tensor([1, 0, 0, 0, 0], dtype=eb.dtype, device=device)

            graph_emb = m.graph_encoder(X, E, y, node_mask)   # [B, 512]
            for i, smi in enumerate(chunk_smis):
                cache[smi] = graph_emb[i].cpu().to(dtype)

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    torch.save(cache, cache_path)
    print(f'  ✓ saved {len(cache)} entries to {cache_path}')
    return cache


def build_zms_cache(align_ckpt_path, msg_root, cache_path, device='cpu',
                    batch_size=64, dtype=torch.float16):
    """对 MSG 全部样本（按 spec_id 索引）用 align ckpt 跑 ms_encoder

    Args:
        msg_root: data/msg_diffms 目录
        cache_path: data/cache/zms_v1.pt
    """
    from models.model import FLASH
    from easydict import EasyDict

    print(f'[zms cache] 构建中: 扫 {msg_root} → {cache_path}')

    # 构造 align 模型
    cfg = EasyDict({
        'stage': 'align',
        'contrastive_dim': 512,
        'noise_scale': 0.0,
        'condition_embedding': {'embed_dim': 256},
        'gat': {'hidden_dim': 256, 'num_layers': 4, 'dropout': 0.0},
        'flow': {'n_timesteps': 100},
        'flash': {'beta1': 3.0},
        'graph_encoder': {'n_layers': 4},
        'ms_encoder': {'dim_sos': 13, 'dim_formula': 144, 'hidden_dim': 512,
                       'num_transformer_layers': 3, 'nhead': 8,
                       'dropout': 0.0, 'input_dropout': 0.0, 'max_len': 129},
        'node_dim': 256,
        'contrastive_temperature_init': 30.0,
    })
    m = FLASH(cfg, num_node_types=14, num_edge_types=5).to(device)
    try:
        sd = torch.load(align_ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(align_ckpt_path, map_location=device)
    sd = sd['model'] if 'model' in sd else sd
    m.load_state_dict(sd, strict=False)
    m.eval()

    # 加载 MSG labels
    labels_df = pd.read_csv(os.path.join(msg_root, 'labels.tsv'), sep='\t')
    labels_df = labels_df[labels_df['ionization'].isin(_DENIMS_PRECURSOR.keys())].copy()
    print(f'  MSG 样本: {len(labels_df)} (仅 [M+H]+/[M-H]-)')

    subform_dir = os.path.join(msg_root, 'subformulae', 'default_subformulae')
    cache = {}

    # 分批处理
    rows = labels_df.to_dict('records')
    n = len(rows)
    n_batches = (n + batch_size - 1) // batch_size
    with torch.no_grad():
        for s in tqdm(range(0, n, batch_size), total=n_batches, desc='  zms encoder'):
            chunk_rows = rows[s:s+batch_size]
            sos_list, fa_list, mask_list, spec_ids = [], [], [], []
            for row in chunk_rows:
                spec_id = row['spec']
                p = os.path.join(subform_dir, f'{spec_id}.json')
                if not os.path.exists(p):
                    continue
                try:
                    with open(p) as f:
                        data = json.load(f)
                except Exception:
                    continue
                output_tbl = data.get('output_tbl')
                if output_tbl is None:
                    continue
                per_peak_formulas = output_tbl.get('formula', []) or []
                if not per_peak_formulas:
                    continue
                fa, mk = _encode_peaks_to_formula_array(per_peak_formulas, max_peaks=128)
                # sos
                pre_oh = torch.zeros(2)
                pre_oh[_DENIMS_PRECURSOR[row['ionization']]] = 1.0
                en_oh = torch.zeros(11)
                en_oh[0] = 1.0
                sos = torch.cat([pre_oh, en_oh], dim=0).view(1, -1)
                sos_list.append(sos)
                fa_list.append(fa)
                mask_list.append(mk)
                spec_ids.append(spec_id)

            if not sos_list:
                continue
            sos = torch.stack(sos_list).to(device)
            fa = torch.stack(fa_list).to(device)
            mk = torch.stack(mask_list).to(device)
            ms_emb = m.ms_encoder(sos, fa, mask=mk)   # [B, 512]
            for i, sid in enumerate(spec_ids):
                cache[sid] = ms_emb[i].cpu().to(dtype)

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    torch.save(cache, cache_path)
    print(f'  ✓ saved {len(cache)} entries to {cache_path}')
    return cache


def ensure_cond_emb_cache(stage, align_ckpt_path, smiles_pool=None, msg_root=None,
                          cache_dir='./data/cache', device='cpu', batch_size=64,
                          force_rebuild=False):
    """根据 stage 确保需要的 cache 文件存在。

    Args:
        stage: 'graph2mol' or 'ms2mol'
        align_ckpt_path: DeniMS encoder ckpt
        smiles_pool: graph2mol 阶段必需，list of unique SMILES
        msg_root: ms2mol 阶段必需
        cache_dir: 缓存目录
        force_rebuild: 强制重建

    Returns:
        cache dict（zmol or zms）
    """
    paths = _cache_paths(cache_dir)
    if stage == 'graph2mol':
        path = paths['zmol']
        if os.path.exists(path) and not force_rebuild:
            print(f'[zmol cache] 已存在，加载 {path}')
            try:
                return torch.load(path, weights_only=False)
            except TypeError:
                return torch.load(path)
        if smiles_pool is None:
            raise ValueError('graph2mol 阶段需要 smiles_pool 来构建 zmol cache')
        return build_zmol_cache(align_ckpt_path, smiles_pool, path,
                                 device=device, batch_size=batch_size)
    elif stage == 'ms2mol':
        path = paths['zms']
        if os.path.exists(path) and not force_rebuild:
            print(f'[zms cache] 已存在，加载 {path}')
            try:
                return torch.load(path, weights_only=False)
            except TypeError:
                return torch.load(path)
        if msg_root is None:
            raise ValueError('ms2mol 阶段需要 msg_root 来构建 zms cache')
        return build_zms_cache(align_ckpt_path, msg_root, path,
                                device=device, batch_size=batch_size)
    else:
        raise ValueError(f'stage 必须是 graph2mol/ms2mol，得到 {stage}')
