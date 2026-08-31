
import os
import re
import glob
import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Optional
import pickle


INSTRUMENT_TYPES = ['Orbitrap', 'QTOF', 'NONE']
IONIZATION_TYPES = ['[M+H]+', '[M+Na]+']


def get_instrument_type_idx(instrument_type):

    if instrument_type in INSTRUMENT_TYPES:
        return INSTRUMENT_TYPES.index(instrument_type)
    return INSTRUMENT_TYPES.index('NONE')


def get_ionization_type_idx(ionization_type):

    if ionization_type in IONIZATION_TYPES:
        return IONIZATION_TYPES.index(ionization_type)
    return 0


def parse_collision_energy(ce_str, precursor_mz, charge):

    if not ce_str or ce_str == 'N/A':
        return None, None
    

    numbers = re.findall(r'\d+\.?\d*', str(ce_str))
    if not numbers:
        return None, None
    
    ce = float(numbers[0])
    

    if 'eV' in str(ce_str).lower():

        nce = ce / (precursor_mz / 500.0)
    else:

        nce = ce
    
    return ce, nce


def unify_precursor_type(precursor_type):

    precursor_type = str(precursor_type).strip()
    

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

















    try:

        valid_mask = (intensity_array > 0) & (mz_array > 0) & (mz_array <= max_mz)
        if not np.any(valid_mask):
            return False, None, None, None
            
        mz_filtered = mz_array[valid_mask]
        intensity_filtered = intensity_array[valid_mask]
        

        precursor_mask = np.abs(mz_filtered - precursor_mz) > 1.5
        mz_filtered = mz_filtered[precursor_mask]
        intensity_filtered = intensity_filtered[precursor_mask]
        
        if len(mz_filtered) < 5:
            return False, None, None, None
        

        max_intensity = np.max(intensity_filtered)
        if max_intensity > 0:
            intensity_filtered = intensity_filtered / max_intensity
        

        num_bins = int(max_mz / resolution) + 1
        spec_array = np.zeros((num_bins, 1))
        
        for mz, intensity in zip(mz_filtered, intensity_filtered):
            bin_idx = int(mz / resolution)
            if 0 <= bin_idx < num_bins:
                spec_array[bin_idx, 0] = max(spec_array[bin_idx, 0], intensity)
        
        return True, mz_filtered, intensity_filtered, spec_array
        
    except Exception as e:
        print(f"processingspectrum processing error: {e}")
        return False, None, None, None


def parse_ms_file(file_path):









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


        spectrum_data['mz_array'] = np.array(spectrum_data['mz_array'])
        spectrum_data['intensity_array'] = np.array(spectrum_data['intensity_array'])

        return spectrum_data

    except Exception as e:
        print(f"Error parsing file  {file_path} : : {e}")
        return None


def parse_raw_spectrum_peaks(file_path, max_peaks=200, normalize=True):












    spectrum_data = parse_ms_file(file_path)

    if spectrum_data is None or len(spectrum_data['mz_array']) == 0:

        return np.zeros((0, 2), dtype=np.float32), 0.0

    mz_array = spectrum_data['mz_array']
    intensity_array = spectrum_data['intensity_array']
    precursor_mz = spectrum_data['precursor_mz']


    if normalize and len(intensity_array) > 0:
        max_intensity = np.max(intensity_array)
        if max_intensity > 0:
            intensity_array = intensity_array / max_intensity


    if len(mz_array) > max_peaks:
        top_indices = np.argsort(intensity_array)[-max_peaks:]
        mz_array = mz_array[top_indices]
        intensity_array = intensity_array[top_indices]


    sort_indices = np.argsort(mz_array)
    mz_array = mz_array[sort_indices]
    intensity_array = intensity_array[sort_indices]


    peaks = np.stack([mz_array, intensity_array], axis=1).astype(np.float32)

    return peaks, precursor_mz


def build_mol_spectrum_mapping(sdf_dir, spec_dir, instrument_type='Orbitrap'):











    mapping = {}
    

    sdf_files = glob.glob(os.path.join(sdf_dir, "*.sdf"))
    sdf_mol_ids = set()
    
    for sdf_file in sdf_files:
        mol_id = os.path.basename(sdf_file).replace('.sdf', '')
        sdf_mol_ids.add(mol_id)
    

    spec_pattern = os.path.join(spec_dir, instrument_type, "*_*.ms")
    spec_files = glob.glob(spec_pattern)
    
    for spec_file in spec_files:

        filename = os.path.basename(spec_file)
        mol_id = filename.split('_')[0]
        

        if mol_id in sdf_mol_ids:
            if mol_id not in mapping:
                mapping[mol_id] = []
            mapping[mol_id].append(spec_file)
    
    print(f"found {len(mapping)}  molecules have associated  {instrument_type} mass spectrumdata")
    print(f"total {sum(len(files) for files in mapping.values())}  spectrum files")
    
    return mapping


def load_and_process_spectrum(spec_file, config):











    spectrum_data = parse_ms_file(spec_file)
    if spectrum_data is None:
        return None
    

    resolution = config.get('resolution', 0.2)
    max_mz = config.get('max_mz', 1500)
    type2charge = config.get('type2charge', {})
    precursor_type_mapping = config.get('precursor_type', {})
    

    precursor_type = spectrum_data['precursor_type']
    charge_str = type2charge.get(precursor_type, '+1')
    charge = int(charge_str.replace('+', '').replace('-', ''))
    if charge_str.startswith('-'):
        charge = -charge
    

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
    

    ce, nce = parse_collision_energy(
        spectrum_data['collision_energy'],
        spectrum_data['precursor_mz'],
        abs(charge)
    )
    

    if ce is None and nce is None:
        nce = 25.0
    

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











    spectrum_cache = {}
    
    for mol_id, spec_files in mol_spec_mapping.items():
        processed_specs = []
        
        for spec_file in spec_files:
            processed_data = load_and_process_spectrum(spec_file, config)
            if processed_data is not None:
                processed_specs.append(processed_data)
        
        if processed_specs:
            spectrum_cache[mol_id] = processed_specs
    

    if cache_file:
        with open(cache_file, 'wb') as f:
            pickle.dump(spectrum_cache, f)
        print(f"Spectrum cache saved to: : {cache_file}")
    
    print(f"processed successfully {len(spectrum_cache)}  molecules with spectrum data")
    return spectrum_cache


def load_spectrum_cache(cache_file):

    with open(cache_file, 'rb') as f:
        spectrum_cache = pickle.load(f)
    print(f"from cacheloading {len(spectrum_cache)}  molecules with spectrum data")
    return spectrum_cache


def get_spectrum_for_molecule(mol_id, spectrum_cache, mode='random'):











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

        avg_spec = np.mean([spec['spec'] for spec in spec_list], axis=0)
        avg_env = np.mean([spec['env'] for spec in spec_list], axis=0)
        
        result = spec_list[0].copy()
        result['spec'] = avg_spec
        result['env'] = avg_env
        result['title'] = f"Averaged_{mol_id}"
        return result
    else:
        return spec_list[0]



def load_pretrained_embeddings(embeddings_path):









    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"loading {len(embeddings_dict)}  spectrum files with pretrained features")
        return embeddings_dict
    except Exception as e:
        print(f"loadingpretrained features: : {e}")
        return {}


def build_mol_pretrained_mapping(sdf_dir, spec_dir, embeddings_dict, instrument_type='Orbitrap'):












    mapping = {}
    

    sdf_files = glob.glob(os.path.join(sdf_dir, "*.sdf"))
    sdf_mol_ids = set()
    
    for sdf_file in sdf_files:
        mol_id = os.path.basename(sdf_file).replace('.sdf', '')
        sdf_mol_ids.add(mol_id)
    
    print(f"[DEBUG] found {len(sdf_mol_ids)}  SDFfile")
    print(f"[DEBUG] SDFfileIDrange: {min(sdf_mol_ids, key=int)} - {max(sdf_mol_ids, key=int)}")
    print(f"[DEBUG] pretrained featurescount: {len(embeddings_dict)}")
    

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
    
    print(f"[DEBUG] pretrained featuresIDrange: {min(pretrained_mol_ids, key=int)} - {max(pretrained_mol_ids, key=int)}")
    print(f"[DEBUG] pretrained featureskeysmodedistribution: {feature_key_patterns}")
    

    intersection = sdf_mol_ids.intersection(pretrained_mol_ids)
    print(f"[DEBUG] SDFintersection with pretrained features: {len(intersection)}  molecules")
    print(f"[DEBUG] intersectionmoleculeIDexamples: {sorted(list(intersection), key=int)[:10]}")
    
    if len(intersection) == 0:
        print(f"[WARNING] no with foundSDFcorrespondence between SDF files and pretrained features!")
        print(f"[WARNING] SDFexamples: {sorted(list(sdf_mol_ids), key=int)[:10]}")
        print(f"[WARNING] pretrained featuresexamples: {sorted(list(pretrained_mol_ids), key=int)[:10]}")
        return {}
    

    matched_count = 0
    for filename, embedding in embeddings_dict.items():

        if '_' in filename:
            mol_id = filename.split('_')[0]
        else:
            mol_id = filename
        

        if mol_id in sdf_mol_ids:
            if mol_id not in mapping:
                mapping[mol_id] = []
            mapping[mol_id].append(embedding)
            matched_count += 1
    
    print(f"[DEBUG] successfulmatching {len(mapping)}  molecules with pretrained spectrum features")
    print(f"[DEBUG] Matched  {matched_count}  pretrained features")
    print(f"[DEBUG] Average per molecule:  {matched_count/len(mapping):.1f}  pretrained features")
    

    if len(mapping) > 0:
        sample_mappings = sorted(list(mapping.items()), key=lambda x: int(x[0]))[:5]
        for mol_id, embs in sample_mappings:
            print(f"[DEBUG] molecule {mol_id}: {len(embs)}  pretrained features, feature shape: {embs[0].shape}")
    
    return mapping


def get_pretrained_embedding_for_molecule(mol_id, pretrained_mapping, mode='random'):











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

        selected = np.mean(embeddings_list, axis=0)
    else:
        selected = embeddings_list[0]
    

    if isinstance(selected, np.ndarray):
        if selected.ndim == 2 and selected.shape[0] == 1:
            selected = selected[0]  # [1, 1024] -> [1024]
        elif selected.ndim == 1:
            pass
        else:
            print(f"Warning: Unexpected embedding shape: {selected.shape}")
            return None
    else:
        print(f"Warning: Embedding is not numpy array: {type(selected)}")
        return None
    

    if selected.shape != (1024,):
        print(f"Warning: Final embedding shape mismatch: {selected.shape}, expected (1024,)")
        return None
    
    return selected


def create_pretrained_spectrum_cache(sdf_dir, spec_dir, embeddings_path, instrument_type='Orbitrap', cache_file=None):














    embeddings_dict = load_pretrained_embeddings(embeddings_path)
    if not embeddings_dict:
        return {}
    

    pretrained_mapping = build_mol_pretrained_mapping(sdf_dir, spec_dir, embeddings_dict, instrument_type)
    

    if cache_file:
        with open(cache_file, 'wb') as f:
            pickle.dump(pretrained_mapping, f)
        print(f"Pretrained-feature cache saved to: : {cache_file}")
    
    return pretrained_mapping



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









    metadata = {
        'ionization': '[M+H]+',
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
        print(f"metadata parsing error for file  {file_path}: {e}")

    return metadata


def create_multi_instrument_pretrained_cache(sdf_dir, spec_dir, embeddings_paths, cache_file=None):












    from tqdm import tqdm


    print("[INFO] Scanning SDF files...")
    sdf_files = glob.glob(os.path.join(sdf_dir, "*.sdf"))
    sdf_mol_ids = set()

    for sdf_file in tqdm(sdf_files, desc="Reading SDF file list"):
        mol_id = os.path.basename(sdf_file).replace('.sdf', '')
        sdf_mol_ids.add(mol_id)

    print(f"[INFO] found {len(sdf_mol_ids)}  SDFfile")


    pretrained_cache = {}  # {mol_id: [feature_entries]}
    mol_spectrum_mapping = {}  # {mol_id: [spectrum_info]}
    instrument_stats = {}

    total_features = 0

    for instrument_type, embeddings_path in embeddings_paths.items():
        if not os.path.exists(embeddings_path):
            print(f"[WARNING] Pretrained-feature file does not exist: {embeddings_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[INFO] processing {instrument_type} pretrained features...")
        print(f"{'='*60}")


        print(f"[INFO] Loading pretrained-feature file: {embeddings_path}")
        embeddings_dict = load_pretrained_embeddings(embeddings_path)
        if not embeddings_dict:
            continue


        ms_dir = os.path.join(spec_dir, instrument_type)
        if not os.path.exists(ms_dir):
            print(f"[WARNING] MSdirectory does not exist: {ms_dir}")
            continue


        print(f"[INFO] parsing {instrument_type} MSfile ionizationinformation...")
        ms_files = glob.glob(os.path.join(ms_dir, "*.ms"))
        filename_to_metadata = {}

        for ms_file in tqdm(ms_files, desc=f"parsing {instrument_type} MSfile"):
            filename = os.path.basename(ms_file).replace('.ms', '')
            metadata = parse_ms_file_metadata(ms_file)
            metadata['ms_file_path'] = ms_file
            filename_to_metadata[filename] = metadata

        print(f"[INFO] parsing {len(filename_to_metadata)}   {instrument_type} MSfile")


        ionization_counts = {}
        for meta in filename_to_metadata.values():
            ion = meta.get('ionization', '[M+H]+')
            ionization_counts[ion] = ionization_counts.get(ion, 0) + 1
        print(f"[INFO] {instrument_type} Ionizationdistribution:")
        for ion, count in sorted(ionization_counts.items(), key=lambda x: -x[1]):
            print(f"       - {ion}: {count}")


        print(f"[INFO] matching {instrument_type} pretrained features and SDFmolecule...")
        matched_count = 0
        skipped_no_sdf = 0
        skipped_bad_shape = 0

        for filename, embedding in tqdm(embeddings_dict.items(), desc=f"processing {instrument_type} features"):

            if '_' in filename:
                mol_id = filename.split('_')[0]
            else:
                mol_id = filename


            if mol_id not in sdf_mol_ids:
                skipped_no_sdf += 1
                continue


            metadata = filename_to_metadata.get(filename, {})
            ionization = metadata.get('ionization', '[M+H]+')
            ms_file_path = metadata.get('ms_file_path', '')


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


            if mol_id not in mol_spectrum_mapping:
                mol_spectrum_mapping[mol_id] = []
            mol_spectrum_mapping[mol_id].append({
                'ms_file': ms_file_path,
                'instrument': instrument_type,
                'ionization': ionization,
            })


        instrument_stats[instrument_type] = {
            'total_embeddings': len(embeddings_dict),
            'matched': matched_count,
            'skipped_no_sdf': skipped_no_sdf,
            'skipped_bad_shape': skipped_bad_shape,
            'ionization_distribution': ionization_counts,
        }

        print(f"\n[INFO] {instrument_type} processing results:")
        print(f"       - total pretrained features: {len(embeddings_dict)}")
        print(f"       - successfulmatching: {matched_count}")
        print(f"       - skipped(without SDF): {skipped_no_sdf}")
        print(f"       - skipped(shapeERROR): {skipped_bad_shape}")


    mol_by_instrument = {inst: set() for inst in embeddings_paths.keys()}
    for mol_id, features in pretrained_cache.items():
        for feat in features:
            mol_by_instrument[feat['instrument_type']].add(mol_id)


    print(f"\n{'='*60}")
    print("[INFO] Multi-instrument pretrained-feature cache created")
    print(f"{'='*60}")
    print(f"[INFO] Matched  {len(pretrained_cache)}  molecules  {total_features}  pretrained features")

    if len(pretrained_cache) > 0:
        print(f"[INFO] Average per molecule:  {total_features/len(pretrained_cache):.2f}  pretrained features")

        print(f"\n[INFO] Molecule coverage by instrument type:")
        for inst, mols in mol_by_instrument.items():
            print(f"       - {inst}: {len(mols)}  molecules")


        if len(mol_by_instrument) >= 2:
            instruments = list(mol_by_instrument.keys())
            overlap = mol_by_instrument[instruments[0]].intersection(mol_by_instrument[instruments[1]])
            print(f"       - with both  {instruments[0]}  and  {instruments[1]} data: {len(overlap)}  molecules")


    if cache_file:
        unified_cache = {
            'version': '2.0',
            'pretrained_features': pretrained_cache,
            'mol_spectrum_mapping': mol_spectrum_mapping,
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
        print(f"\n[INFO] Unified cache saved to: : {cache_file}")


        old_files = [
            cache_file.replace('.pkl', '_mol_spectrum_mapping.pkl'),
            cache_file.replace('.pkl', '_stats.pkl'),
        ]
        for old_file in old_files:
            if os.path.exists(old_file):
                os.remove(old_file)
                print(f"[INFO] Deleted obsolete file: {old_file}")

    return pretrained_cache


def get_pretrained_embedding_with_conditions(mol_id, pretrained_cache, mode='random'):











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