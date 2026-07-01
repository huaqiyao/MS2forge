import os
import re
import glob
import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Optional
import pickle

# 条件类型定义（与model.py保持一致）
INSTRUMENT_TYPES = ['Orbitrap', 'QTOF', 'NONE']
IONIZATION_TYPES = ['[M+H]+', '[M+Na]+']


def get_instrument_type_idx(instrument_type):
    """获取仪器类型索引"""
    if instrument_type in INSTRUMENT_TYPES:
        return INSTRUMENT_TYPES.index(instrument_type)
    return INSTRUMENT_TYPES.index('NONE')  # 未知仪器归入NONE


def get_ionization_type_idx(ionization_type):
    """获取ionization方式索引"""
    if ionization_type in IONIZATION_TYPES:
        return IONIZATION_TYPES.index(ionization_type)
    return 0  # 默认返回[M+H]+


def parse_collision_energy(ce_str, precursor_mz, charge):
    """解析碰撞能量字符串"""
    if not ce_str or ce_str == 'N/A':
        return None, None
    
    # 尝试提取数值
    numbers = re.findall(r'\d+\.?\d*', str(ce_str))
    if not numbers:
        return None, None
    
    ce = float(numbers[0])
    
    # 计算归一化碰撞能量 (NCE)
    if 'eV' in str(ce_str).lower():
        # 简单的eV到NCE转换（可能需要根据实际情况调整）
        nce = ce / (precursor_mz / 500.0)
    else:
        # 假设已经是NCE或百分比
        nce = ce
    
    return ce, nce


def unify_precursor_type(precursor_type):
    """统一前体离子类型格式"""
    precursor_type = str(precursor_type).strip()
    
    # 标准化格式
    replacements = {
        '(M+H)+': '[M+H]+',
        '(M-H)-': '[M-H]-',
        '(M+Na)+': '[M+Na]+',
        '(2M+H)+': '[2M+H]+',
        '(2M-H)-': '[2M-H]-',
        '(M+2H)2+': '[M+2H]2+',
    }
    
    for old, new in replacements.items():
        precursor_type = precursor_type.replace(old, new)
    
    return precursor_type


def generate_ms_spectrum(mz_array, intensity_array, precursor_mz, resolution=0.2, max_mz=1500, charge=1):
    """
    将原始质谱数据转换为固定分辨率的谱图数组
    
    Args:
        mz_array: m/z 值数组
        intensity_array: 强度数组
        precursor_mz: 前体离子m/z
        resolution: 分辨率
        max_mz: 最大m/z值
        charge: 电荷数
    
    Returns:
        good_spec: 是否为有效谱图
        processed_mz: 处理后的m/z数组
        processed_intensity: 处理后的强度数组  
        spec_array: 固定分辨率的谱图数组
    """
    try:
        # 过滤有效的峰
        valid_mask = (intensity_array > 0) & (mz_array > 0) & (mz_array <= max_mz)
        if not np.any(valid_mask):
            return False, None, None, None
            
        mz_filtered = mz_array[valid_mask]
        intensity_filtered = intensity_array[valid_mask]
        
        # 移除前体离子峰（±1.5 Da）
        precursor_mask = np.abs(mz_filtered - precursor_mz) > 1.5
        mz_filtered = mz_filtered[precursor_mask]
        intensity_filtered = intensity_filtered[precursor_mask]
        
        if len(mz_filtered) < 5:  # 至少需要5个峰
            return False, None, None, None
        
        # 归一化强度
        max_intensity = np.max(intensity_filtered)
        if max_intensity > 0:
            intensity_filtered = intensity_filtered / max_intensity
        
        # 创建固定分辨率的谱图数组
        num_bins = int(max_mz / resolution) + 1
        spec_array = np.zeros((num_bins, 1))
        
        for mz, intensity in zip(mz_filtered, intensity_filtered):
            bin_idx = int(mz / resolution)
            if 0 <= bin_idx < num_bins:
                spec_array[bin_idx, 0] = max(spec_array[bin_idx, 0], intensity)
        
        return True, mz_filtered, intensity_filtered, spec_array
        
    except Exception as e:
        print(f"处理质谱数据时出错: {e}")
        return False, None, None, None


def parse_ms_file(file_path):
    """
    解析.ms文件

    Args:
        file_path: .ms文件路径

    Returns:
        spectrum_data: 包含质谱数据的字典
    """
    spectrum_data = {
        'title': '',
        'precursor_mz': 0.0,
        'precursor_type': '[M+H]+',
        'collision_energy': '',
        'mz_array': [],
        'intensity_array': []
    }

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 解析元数据
        in_peaks = False
        for line in lines:
            line = line.strip()

            if line.startswith('>compound'):
                spectrum_data['title'] = line.split(' ', 1)[1] if len(line.split(' ', 1)) > 1 else ''
            elif line.startswith('>parentmass'):
                try:
                    spectrum_data['precursor_mz'] = float(line.split(' ', 1)[1])
                except:
                    spectrum_data['precursor_mz'] = 0.0
            elif line.startswith('>ionization'):
                spectrum_data['precursor_type'] = unify_precursor_type(line.split(' ', 1)[1])
            elif line.startswith('>ms2peaks'):
                in_peaks = True
                continue
            elif in_peaks and line and not line.startswith('#') and not line.startswith('>'):
                # 解析峰数据
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        if mz > 0 and intensity > 0:
                            spectrum_data['mz_array'].append(mz)
                            spectrum_data['intensity_array'].append(intensity)
                    except ValueError:
                        continue

        # 转换为numpy数组
        spectrum_data['mz_array'] = np.array(spectrum_data['mz_array'])
        spectrum_data['intensity_array'] = np.array(spectrum_data['intensity_array'])

        return spectrum_data

    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
        return None


def parse_raw_spectrum_peaks(file_path, max_peaks=200, normalize=True):
    """
    从MS文件解析原始质谱峰数据（用于origin模式）

    Args:
        file_path: .ms文件路径
        max_peaks: 最大峰数量
        normalize: 是否归一化强度

    Returns:
        peaks: [num_peaks, 2] numpy数组，每行为 (m/z, intensity)
        precursor_mz: 前体离子m/z
    """
    spectrum_data = parse_ms_file(file_path)

    if spectrum_data is None or len(spectrum_data['mz_array']) == 0:
        # 返回空峰列表
        return np.zeros((0, 2), dtype=np.float32), 0.0

    mz_array = spectrum_data['mz_array']
    intensity_array = spectrum_data['intensity_array']
    precursor_mz = spectrum_data['precursor_mz']

    # 归一化强度
    if normalize and len(intensity_array) > 0:
        max_intensity = np.max(intensity_array)
        if max_intensity > 0:
            intensity_array = intensity_array / max_intensity

    # 按强度排序，保留top-k峰
    if len(mz_array) > max_peaks:
        top_indices = np.argsort(intensity_array)[-max_peaks:]
        mz_array = mz_array[top_indices]
        intensity_array = intensity_array[top_indices]

    # 按m/z排序
    sort_indices = np.argsort(mz_array)
    mz_array = mz_array[sort_indices]
    intensity_array = intensity_array[sort_indices]

    # 组合为 [num_peaks, 2]
    peaks = np.stack([mz_array, intensity_array], axis=1).astype(np.float32)

    return peaks, precursor_mz


def build_mol_spectrum_mapping(sdf_dir, spec_dir, instrument_type='Orbitrap'):
    """
    构建分子ID到质谱文件的映射
    
    Args:
        sdf_dir: SDF文件目录
        spec_dir: 质谱文件目录  
        instrument_type: 仪器类型 ('Orbitrap' 或 'QTOF')
    
    Returns:
        mapping: 映射字典 {mol_id: [spec_files]}
    """
    mapping = {}
    
    # 获取所有SDF文件
    sdf_files = glob.glob(os.path.join(sdf_dir, "*.sdf"))
    sdf_mol_ids = set()
    
    for sdf_file in sdf_files:
        mol_id = os.path.basename(sdf_file).replace('.sdf', '')
        sdf_mol_ids.add(mol_id)
    
    # 获取对应仪器类型的质谱文件
    spec_pattern = os.path.join(spec_dir, instrument_type, "*_*.ms")
    spec_files = glob.glob(spec_pattern)
    
    for spec_file in spec_files:
        # 从文件名提取分子ID (例如: 8_1.ms -> 8)
        filename = os.path.basename(spec_file)
        mol_id = filename.split('_')[0]
        
        # 只保留有对应SDF文件的质谱文件
        if mol_id in sdf_mol_ids:
            if mol_id not in mapping:
                mapping[mol_id] = []
            mapping[mol_id].append(spec_file)
    
    print(f"找到 {len(mapping)} 个分子有对应的 {instrument_type} 质谱数据")
    print(f"总共 {sum(len(files) for files in mapping.values())} 个质谱文件")
    
    return mapping


def load_and_process_spectrum(spec_file, config):
    """
    加载和处理单个质谱文件
    
    Args:
        spec_file: 质谱文件路径
        config: 配置参数
    
    Returns:
        processed_data: 处理后的质谱数据字典
    """
    # 解析文件
    spectrum_data = parse_ms_file(spec_file)
    if spectrum_data is None:
        return None
    
    # 提取配置参数
    resolution = config.get('resolution', 0.2)
    max_mz = config.get('max_mz', 1500)
    type2charge = config.get('type2charge', {})
    precursor_type_mapping = config.get('precursor_type', {})
    
    # 获取电荷
    precursor_type = spectrum_data['precursor_type']
    charge_str = type2charge.get(precursor_type, '+1')
    charge = int(charge_str.replace('+', '').replace('-', ''))
    if charge_str.startswith('-'):
        charge = -charge
    
    # 处理质谱数据
    good_spec, _, _, spec_array = generate_ms_spectrum(
        spectrum_data['mz_array'],
        spectrum_data['intensity_array'],
        spectrum_data['precursor_mz'],
        resolution=resolution,
        max_mz=max_mz,
        charge=charge
    )
    
    if not good_spec:
        return None
    
    # 解析碰撞能量
    ce, nce = parse_collision_energy(
        spectrum_data['collision_energy'],
        spectrum_data['precursor_mz'],
        abs(charge)
    )
    
    # 如果没有碰撞能量信息，使用默认值
    if ce is None and nce is None:
        nce = 25.0  # 使用默认的标准化碰撞能量
    
    # 构建环境数据 [precursor_mz, nce, precursor_type_idx]
    precursor_type_idx = precursor_type_mapping.get(precursor_type, 0)
    env_array = np.array([
        spectrum_data['precursor_mz'],
        nce if nce is not None else 25.0,
        precursor_type_idx
    ], dtype=np.float32)
    
    return {
        'title': spectrum_data['title'],
        'precursor_type': precursor_type,
        'spec': spec_array,
        'env': env_array,
        'precursor_mz': spectrum_data['precursor_mz'],
        'collision_energy': spectrum_data['collision_energy']
    }


def create_spectrum_cache(mol_spec_mapping, config, cache_file=None):
    """
    创建质谱数据缓存
    
    Args:
        mol_spec_mapping: 分子-质谱映射
        config: 处理配置
        cache_file: 缓存文件路径
    
    Returns:
        spectrum_cache: 质谱数据缓存字典 {mol_id: [processed_spec_data]}
    """
    spectrum_cache = {}
    
    for mol_id, spec_files in mol_spec_mapping.items():
        processed_specs = []
        
        for spec_file in spec_files:
            processed_data = load_and_process_spectrum(spec_file, config)
            if processed_data is not None:
                processed_specs.append(processed_data)
        
        if processed_specs:
            spectrum_cache[mol_id] = processed_specs
    
    # 保存缓存
    if cache_file:
        with open(cache_file, 'wb') as f:
            pickle.dump(spectrum_cache, f)
        print(f"质谱缓存已保存到: {cache_file}")
    
    print(f"成功处理 {len(spectrum_cache)} 个分子的质谱数据")
    return spectrum_cache


def load_spectrum_cache(cache_file):
    """加载质谱数据缓存"""
    with open(cache_file, 'rb') as f:
        spectrum_cache = pickle.load(f)
    print(f"从缓存加载了 {len(spectrum_cache)} 个分子的质谱数据")
    return spectrum_cache


def get_spectrum_for_molecule(mol_id, spectrum_cache, mode='random'):
    """
    为分子获取质谱数据
    
    Args:
        mol_id: 分子ID
        spectrum_cache: 质谱缓存
        mode: 选择模式 ('random', 'first', 'average')
    
    Returns:
        spectrum_data: 质谱数据或None
    """
    if mol_id not in spectrum_cache:
        return None
    
    spec_list = spectrum_cache[mol_id]
    if not spec_list:
        return None
    
    if mode == 'random':
        return random.choice(spec_list)
    elif mode == 'first':
        return spec_list[0]
    elif mode == 'average' and len(spec_list) > 1:
        # 简单的谱图平均（可以改进）
        avg_spec = np.mean([spec['spec'] for spec in spec_list], axis=0)
        avg_env = np.mean([spec['env'] for spec in spec_list], axis=0)
        
        result = spec_list[0].copy()
        result['spec'] = avg_spec
        result['env'] = avg_env
        result['title'] = f"Averaged_{mol_id}"
        return result
    else:
        return spec_list[0]


# 预训练特征相关函数
def load_pretrained_embeddings(embeddings_path):
    """
    加载DreaMS预训练质谱特征
    
    Args:
        embeddings_path: batch_embeddings.pkl文件路径
    
    Returns:
        embeddings_dict: {filename: embedding_array} 字典
    """
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"加载了 {len(embeddings_dict)} 个质谱文件的预训练特征")
        return embeddings_dict
    except Exception as e:
        print(f"加载预训练特征时出错: {e}")
        return {}


def build_mol_pretrained_mapping(sdf_dir, spec_dir, embeddings_dict, instrument_type='Orbitrap'):
    """
    构建分子ID到预训练质谱特征的映射
    
    Args:
        sdf_dir: SDF文件目录
        spec_dir: 质谱文件目录
        embeddings_dict: 预训练特征字典 {filename: embedding}
        instrument_type: 仪器类型
    
    Returns:
        mapping: {mol_id: [embedding_arrays]} 映射字典
    """
    mapping = {}
    
    # 获取所有SDF文件
    sdf_files = glob.glob(os.path.join(sdf_dir, "*.sdf"))
    sdf_mol_ids = set()
    
    for sdf_file in sdf_files:
        mol_id = os.path.basename(sdf_file).replace('.sdf', '')
        sdf_mol_ids.add(mol_id)
    
    print(f"[DEBUG] 找到 {len(sdf_mol_ids)} 个SDF文件")
    print(f"[DEBUG] SDF文件ID范围: {min(sdf_mol_ids, key=int)} - {max(sdf_mol_ids, key=int)}")
    print(f"[DEBUG] 预训练特征数量: {len(embeddings_dict)}")
    
    # 分析预训练特征的ID分布
    pretrained_mol_ids = set()
    feature_key_patterns = {}
    
    for filename in embeddings_dict.keys():
        if '_' in filename:
            mol_id = filename.split('_')[0]
            pattern = 'mol_id_spec_idx'
        else:
            mol_id = filename
            pattern = 'mol_id_only'
        
        pretrained_mol_ids.add(mol_id)
        if pattern not in feature_key_patterns:
            feature_key_patterns[pattern] = 0
        feature_key_patterns[pattern] += 1
    
    print(f"[DEBUG] 预训练特征ID范围: {min(pretrained_mol_ids, key=int)} - {max(pretrained_mol_ids, key=int)}")
    print(f"[DEBUG] 预训练特征键模式分布: {feature_key_patterns}")
    
    # 计算SDF与预训练特征的交集
    intersection = sdf_mol_ids.intersection(pretrained_mol_ids)
    print(f"[DEBUG] SDF与预训练特征的交集: {len(intersection)} 个分子")
    print(f"[DEBUG] 交集分子ID样例: {sorted(list(intersection), key=int)[:10]}")
    
    if len(intersection) == 0:
        print(f"[WARNING] 没有找到SDF文件与预训练特征的对应关系!")
        print(f"[WARNING] SDF样例: {sorted(list(sdf_mol_ids), key=int)[:10]}")
        print(f"[WARNING] 预训练特征样例: {sorted(list(pretrained_mol_ids), key=int)[:10]}")
        return {}
    
    # 遍历预训练特征，建立映射
    matched_count = 0
    for filename, embedding in embeddings_dict.items():
        # 从文件名提取分子ID
        if '_' in filename:
            mol_id = filename.split('_')[0]
        else:
            mol_id = filename
        
        # 只保留有对应SDF文件的分子
        if mol_id in sdf_mol_ids:
            if mol_id not in mapping:
                mapping[mol_id] = []
            mapping[mol_id].append(embedding)
            matched_count += 1
    
    print(f"[DEBUG] 成功匹配 {len(mapping)} 个分子的预训练质谱特征")
    print(f"[DEBUG] 总共匹配 {matched_count} 个预训练特征")
    print(f"[DEBUG] 平均每个分子 {matched_count/len(mapping):.1f} 个预训练特征")
    
    # 显示详细匹配信息
    if len(mapping) > 0:
        sample_mappings = sorted(list(mapping.items()), key=lambda x: int(x[0]))[:5]
        for mol_id, embs in sample_mappings:
            print(f"[DEBUG] 分子 {mol_id}: {len(embs)} 个预训练特征，特征形状: {embs[0].shape}")
    
    return mapping


def get_pretrained_embedding_for_molecule(mol_id, pretrained_mapping, mode='random'):
    """
    为分子获取预训练质谱特征
    
    Args:
        mol_id: 分子ID
        pretrained_mapping: 预训练特征映射
        mode: 选择模式 ('random', 'first', 'average')
    
    Returns:
        embedding: [1024] 预训练特征向量或None
    """
    if mol_id not in pretrained_mapping:
        return None
    
    embeddings_list = pretrained_mapping[mol_id]
    if not embeddings_list:
        return None
    
    if mode == 'random':
        selected = random.choice(embeddings_list)
    elif mode == 'first':
        selected = embeddings_list[0]
    elif mode == 'average' and len(embeddings_list) > 1:
        # 平均多个特征
        selected = np.mean(embeddings_list, axis=0)
    else:
        selected = embeddings_list[0]
    
    # 确保返回正确的形状 [1024]
    if isinstance(selected, np.ndarray):
        if selected.ndim == 2 and selected.shape[0] == 1:
            selected = selected[0]  # [1, 1024] -> [1024]
        elif selected.ndim == 1:
            pass  # [1024] - 已经是正确形状
        else:
            print(f"Warning: Unexpected embedding shape: {selected.shape}")
            return None
    else:
        print(f"Warning: Embedding is not numpy array: {type(selected)}")
        return None
    
    # 最终验证
    if selected.shape != (1024,):
        print(f"Warning: Final embedding shape mismatch: {selected.shape}, expected (1024,)")
        return None
    
    return selected


def create_pretrained_spectrum_cache(sdf_dir, spec_dir, embeddings_path, instrument_type='Orbitrap', cache_file=None):
    """
    创建基于预训练特征的质谱缓存
    
    Args:
        sdf_dir: SDF文件目录
        spec_dir: 质谱文件目录
        embeddings_path: 预训练特征文件路径
        instrument_type: 仪器类型
        cache_file: 缓存文件保存路径
    
    Returns:
        pretrained_cache: {mol_id: [pretrained_embeddings]} 缓存字典
    """
    # 加载预训练特征
    embeddings_dict = load_pretrained_embeddings(embeddings_path)
    if not embeddings_dict:
        return {}
    
    # 建立映射
    pretrained_mapping = build_mol_pretrained_mapping(sdf_dir, spec_dir, embeddings_dict, instrument_type)
    
    # 保存缓存
    if cache_file:
        with open(cache_file, 'wb') as f:
            pickle.dump(pretrained_mapping, f)
        print(f"预训练特征缓存已保存到: {cache_file}")
    
    return pretrained_mapping


# 默认配置
DEFAULT_SPECTRUM_CONFIG = {
    'resolution': 0.2,
    'max_mz': 1500,
    'precursor_type': {
        '[M+H]+': 1,
        '[M-H]-': 2,
        '[M+Na]+': 3,
        '[M+2H]2+': 4,
        '[2M+H]+': 5,
        '[2M-H]-': 6,
    },
    'type2charge': {
        '[M+H]+': '+1',
        '[M-H]-': '-1',
        '[M+Na]+': '+1',
        '[M+2H]2+': '+2',
        '[2M+H]+': '+1',
        '[2M-H]-': '-1',
        '[M+H-H2O]+': '+1',
        '[M-H2O+H]+': '+1',
        '[M+H-2H2O]+': '+1',
        '[M+H-NH3]+': '+1',
        '[M+H+NH3]+': '+1',
        '[M+NH4]+': '+1',
        '[M+H-CH2O2]+': '+1',
        '[M+H-CH4O2]+': '+1',
        '[M-H-CO2]-': '-1',
        '[M-CHO2]-': '-1',
        '[M-H-H2O]-': '-1',
        '[M-H2O-H]+': '-1',
    }
}


def parse_ms_file_metadata(file_path):
    """
    从ms文件解析元数据（ionization方式等）

    Args:
        file_path: .ms文件路径

    Returns:
        metadata: 包含ionization等信息的字典
    """
    metadata = {
        'ionization': '[M+H]+',  # 默认值
        'formula': None,
        'compound': None,
    }

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>ionization'):
                    ionization = line.split(' ', 1)[1] if ' ' in line else '[M+H]+'
                    metadata['ionization'] = unify_precursor_type(ionization)
                elif line.startswith('>formula'):
                    metadata['formula'] = line.split(' ', 1)[1] if ' ' in line else None
                elif line.startswith('>compound'):
                    metadata['compound'] = line.split(' ', 1)[1] if ' ' in line else None
    except Exception as e:
        print(f"解析文件元数据时出错 {file_path}: {e}")

    return metadata


def create_multi_instrument_pretrained_cache(sdf_dir, spec_dir, embeddings_paths, cache_file=None):
    """
    创建支持多仪器类型的预训练特征缓存（统一格式）

    Args:
        sdf_dir: SDF文件目录
        spec_dir: 质谱文件目录（包含Orbitrap和QTOF子目录）
        embeddings_paths: 预训练特征文件路径字典 {'Orbitrap': path1, 'QTOF': path2}
        cache_file: 缓存文件保存路径

    Returns:
        pretrained_cache: {mol_id: [{'embedding': array, 'instrument_type': str, 'ionization': str, ...}]}
    """
    from tqdm import tqdm

    # 获取所有SDF文件
    print("[INFO] 扫描SDF文件...")
    sdf_files = glob.glob(os.path.join(sdf_dir, "*.sdf"))
    sdf_mol_ids = set()

    for sdf_file in tqdm(sdf_files, desc="读取SDF文件列表"):
        mol_id = os.path.basename(sdf_file).replace('.sdf', '')
        sdf_mol_ids.add(mol_id)

    print(f"[INFO] 找到 {len(sdf_mol_ids)} 个SDF文件")

    # 主数据结构：存储所有信息
    pretrained_cache = {}  # {mol_id: [feature_entries]}
    mol_spectrum_mapping = {}  # {mol_id: [spectrum_info]}
    instrument_stats = {}

    total_features = 0

    for instrument_type, embeddings_path in embeddings_paths.items():
        if not os.path.exists(embeddings_path):
            print(f"[WARNING] 预训练特征文件不存在: {embeddings_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[INFO] 处理 {instrument_type} 预训练特征...")
        print(f"{'='*60}")

        # 加载预训练特征
        print(f"[INFO] 加载预训练特征文件: {embeddings_path}")
        embeddings_dict = load_pretrained_embeddings(embeddings_path)
        if not embeddings_dict:
            continue

        # 获取对应仪器类型的ms文件目录
        ms_dir = os.path.join(spec_dir, instrument_type)
        if not os.path.exists(ms_dir):
            print(f"[WARNING] MS文件目录不存在: {ms_dir}")
            continue

        # 建立文件名到ionization的映射
        print(f"[INFO] 解析 {instrument_type} MS文件的ionization信息...")
        ms_files = glob.glob(os.path.join(ms_dir, "*.ms"))
        filename_to_metadata = {}

        for ms_file in tqdm(ms_files, desc=f"解析 {instrument_type} MS文件"):
            filename = os.path.basename(ms_file).replace('.ms', '')
            metadata = parse_ms_file_metadata(ms_file)
            metadata['ms_file_path'] = ms_file
            filename_to_metadata[filename] = metadata

        print(f"[INFO] 解析了 {len(filename_to_metadata)} 个 {instrument_type} MS文件")

        # 统计ionization分布
        ionization_counts = {}
        for meta in filename_to_metadata.values():
            ion = meta.get('ionization', '[M+H]+')
            ionization_counts[ion] = ionization_counts.get(ion, 0) + 1
        print(f"[INFO] {instrument_type} Ionization分布:")
        for ion, count in sorted(ionization_counts.items(), key=lambda x: -x[1]):
            print(f"       - {ion}: {count}")

        # 遍历预训练特征
        print(f"[INFO] 匹配 {instrument_type} 预训练特征与SDF分子...")
        matched_count = 0
        skipped_no_sdf = 0
        skipped_bad_shape = 0

        for filename, embedding in tqdm(embeddings_dict.items(), desc=f"处理 {instrument_type} 特征"):
            # 从文件名提取分子ID
            if '_' in filename:
                mol_id = filename.split('_')[0]
            else:
                mol_id = filename

            # 只保留有对应SDF文件的分子
            if mol_id not in sdf_mol_ids:
                skipped_no_sdf += 1
                continue

            # 获取元数据
            metadata = filename_to_metadata.get(filename, {})
            ionization = metadata.get('ionization', '[M+H]+')
            ms_file_path = metadata.get('ms_file_path', '')

            # 确保embedding是正确的形状
            if isinstance(embedding, np.ndarray):
                if embedding.ndim == 2 and embedding.shape[0] == 1:
                    embedding = embedding[0]
                elif embedding.ndim != 1:
                    skipped_bad_shape += 1
                    continue
            else:
                skipped_bad_shape += 1
                continue

            if embedding.shape != (1024,):
                skipped_bad_shape += 1
                continue

            # 创建特征条目（包含所有信息）
            feature_entry = {
                'embedding': embedding,
                'instrument_type': instrument_type,
                'ionization': ionization,
                'instrument_type_idx': get_instrument_type_idx(instrument_type),
                'ionization_type_idx': get_ionization_type_idx(ionization),
                'ms_filename': filename,
                'ms_file_path': ms_file_path,
                'sdf_file_path': os.path.join(sdf_dir, f"{mol_id}.sdf"),
            }

            if mol_id not in pretrained_cache:
                pretrained_cache[mol_id] = []
            pretrained_cache[mol_id].append(feature_entry)
            matched_count += 1
            total_features += 1

            # 记录分子-质谱映射（简化版，用于快速查询）
            if mol_id not in mol_spectrum_mapping:
                mol_spectrum_mapping[mol_id] = []
            mol_spectrum_mapping[mol_id].append({
                'ms_file': ms_file_path,
                'instrument': instrument_type,
                'ionization': ionization,
            })

        # 记录仪器统计
        instrument_stats[instrument_type] = {
            'total_embeddings': len(embeddings_dict),
            'matched': matched_count,
            'skipped_no_sdf': skipped_no_sdf,
            'skipped_bad_shape': skipped_bad_shape,
            'ionization_distribution': ionization_counts,
        }

        print(f"\n[INFO] {instrument_type} 处理结果:")
        print(f"       - 总预训练特征数: {len(embeddings_dict)}")
        print(f"       - 成功匹配: {matched_count}")
        print(f"       - 跳过(无SDF): {skipped_no_sdf}")
        print(f"       - 跳过(形状错误): {skipped_bad_shape}")

    # 统计每种仪器的分子覆盖
    mol_by_instrument = {inst: set() for inst in embeddings_paths.keys()}
    for mol_id, features in pretrained_cache.items():
        for feat in features:
            mol_by_instrument[feat['instrument_type']].add(mol_id)

    # 打印总体统计
    print(f"\n{'='*60}")
    print("[INFO] 多仪器预训练特征缓存创建完成")
    print(f"{'='*60}")
    print(f"[INFO] 总共匹配 {len(pretrained_cache)} 个分子的 {total_features} 个预训练特征")

    if len(pretrained_cache) > 0:
        print(f"[INFO] 平均每个分子 {total_features/len(pretrained_cache):.2f} 个预训练特征")

        print(f"\n[INFO] 各仪器类型分子覆盖:")
        for inst, mols in mol_by_instrument.items():
            print(f"       - {inst}: {len(mols)} 个分子")

        # 统计同时有两种仪器数据的分子
        if len(mol_by_instrument) >= 2:
            instruments = list(mol_by_instrument.keys())
            overlap = mol_by_instrument[instruments[0]].intersection(mol_by_instrument[instruments[1]])
            print(f"       - 同时有 {instruments[0]} 和 {instruments[1]} 数据: {len(overlap)} 个分子")

    # 保存为统一的缓存文件
    if cache_file:
        unified_cache = {
            'version': '2.0',
            'pretrained_features': pretrained_cache,  # 主数据
            'mol_spectrum_mapping': mol_spectrum_mapping,  # 分子-质谱映射
            'stats': {
                'instrument_stats': instrument_stats,
                'total_molecules': len(pretrained_cache),
                'total_features': total_features,
                'mol_by_instrument': {k: len(v) for k, v in mol_by_instrument.items()},
            },
            'metadata': {
                'sdf_dir': sdf_dir,
                'spec_dir': spec_dir,
                'embeddings_paths': embeddings_paths,
                'instrument_types': INSTRUMENT_TYPES,
                'ionization_types': IONIZATION_TYPES,
            }
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(unified_cache, f)
        print(f"\n[INFO] 统一缓存文件已保存到: {cache_file}")

        # 删除旧的分散文件（如果存在）
        old_files = [
            cache_file.replace('.pkl', '_mol_spectrum_mapping.pkl'),
            cache_file.replace('.pkl', '_stats.pkl'),
        ]
        for old_file in old_files:
            if os.path.exists(old_file):
                os.remove(old_file)
                print(f"[INFO] 已删除旧文件: {old_file}")

    return pretrained_cache


def get_pretrained_embedding_with_conditions(mol_id, pretrained_cache, mode='random'):
    """
    为分子获取预训练质谱特征及其条件信息

    Args:
        mol_id: 分子ID
        pretrained_cache: 预训练特征缓存（多仪器格式）
        mode: 选择模式 ('random', 'first')

    Returns:
        dict: {'embedding': array, 'instrument_type_idx': int, 'ionization_type_idx': int} 或 None
    """
    if mol_id not in pretrained_cache:
        return None

    features_list = pretrained_cache[mol_id]
    if not features_list:
        return None

    if mode == 'random':
        selected = random.choice(features_list)
    else:
        selected = features_list[0]

    return {
        'embedding': selected['embedding'],
        'instrument_type_idx': selected['instrument_type_idx'],
        'ionization_type_idx': selected['ionization_type_idx'],
        'instrument_type': selected['instrument_type'],
        'ionization': selected['ionization'],
    }