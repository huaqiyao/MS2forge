

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
from collections import Counter
RDLogger.DisableLog('rdApp.*')
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


    data_subset_ratio = getattr(config, 'data_subset_ratio', 1.0)


    data_split_mode = getattr(config, 'data_split_mode', 'natms')


    use_spectrum = getattr(config, 'use_spectrum', False)
    instrument_type = getattr(config, 'instrument_type', 'all')
    spectrum_config = getattr(config, 'spectrum_config', None)


    use_spectrum_in_training = getattr(config, 'use_spectrum_in_training', use_spectrum)

    use_pretrained_embeddings = getattr(config, 'use_pretrained_embeddings', False)
    pretrained_embeddings_path = getattr(config, 'pretrained_embeddings_path', None)


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

        dataset = MSFileDataset(
            root,
            path_dict=getattr(config, 'path_dict', None),
            data_subset_ratio=data_subset_ratio,
            instrument_type=instrument_type,
            data_split_mode=data_split_mode,
            *args,
            **kwargs
        )
    elif name == 'msg_diffms':


        dataset = DiffMSMSGDataset(
            root,
            data_subset_ratio=data_subset_ratio,
            instrument_type=instrument_type,
            data_split_mode=data_split_mode,
            max_peaks=getattr(config, 'max_peaks', 128),
        )
        return dataset, dataset.subsets
    elif name == 'smiles':

        atomic_numbers = list(getattr(config, 'atomic_numbers', []))
        if not atomic_numbers:
            raise ValueError("dataset.name='smiles' requires in  dataset specified in the configuration atomic_numbers list")
        dataset = SmilesDataset(
            root=root,
            smiles_file=config.smiles_file,
            atomic_numbers=atomic_numbers,
            data_subset_ratio=data_subset_ratio,
            max_atoms=getattr(config, 'max_atoms', None),
            split_seed=getattr(config, 'split_seed', 2026),
            split_ratio=getattr(config, 'split_ratio', (0.95, 0.025, 0.025)),
        )

        return dataset, dataset.subsets
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:

        split_by_molid = torch.load(os.path.join(root, config.split))
        split = {
            k: [dataset.molid2idx[mol_id] for mol_id in mol_id_list if mol_id in dataset.molid2idx]
            for k, mol_id_list in split_by_molid.items()
        }
        subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
        print('Num of samples:', *{(k, len(v)) for k,v in split.items()})
        return dataset, subsets
    else:

        if name in ['natgen', 'msg', 'msfile']:

            if name == 'msfile' and hasattr(dataset, 'smiles2indices'):
                if data_split_mode == 'diffms':

                    print('=== usingDiffMSdata-split mode(by split_diffms.tsvpredefined split)===')


                    instrument_type = getattr(config, 'instrument_type', 'all')
                    cache_file = os.path.join(root, f'split_indices_{instrument_type}_diffms.pt')


                    if os.path.exists(cache_file):
                        print(f'  Loading data split from cache: {cache_file}')
                        split_indices = torch.load(cache_file)
                        train_indices = split_indices['train']
                        val_indices = split_indices['val']
                        test_indices = split_indices['test']
                    else:

                        split_file = os.path.join(root, 'split_diffms.tsv')
                        if not os.path.exists(split_file):
                            raise FileNotFoundError(f"DiffMSmode requires split_diffms.tsvfile,  but not found: {split_file}")

                        split_df = pd.read_csv(split_file, sep='\t')
                        gymid_to_split = dict(zip(split_df['name'], split_df['split']))
                        print(f'  from split_diffms.tsvloading {len(gymid_to_split)}  GymID split information')


                        print(f'  Splitting dataset(the first run may be slow, results will be cached)...')
                        train_indices = []
                        val_indices = []
                        test_indices = []
                        missing_gymids = []

                        for idx in tqdm(range(len(dataset)), desc='  splitdata'):

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
                            print(f'  WARNING: {len(missing_gymids)}  samples have no matching splitlabels')


                        split_indices = {
                            'train': train_indices,
                            'val': val_indices,
                            'test': test_indices
                        }
                        torch.save(split_indices, cache_file)
                        print(f'  Data split cached at: : {cache_file}')

                    subsets = {
                        'train': Subset(dataset, train_indices),
                        'val': Subset(dataset, val_indices),
                        'test': Subset(dataset, test_indices)
                    }

                    print(f'  Train: {len(train_indices)} samplethis  ({len(train_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Val: {len(val_indices)} samplethis  ({len(val_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Test: {len(test_indices)} samplethis  ({len(test_indices)/len(dataset)*100:.2f}%)')

                    return dataset, subsets

                elif data_split_mode == 'split':

                    print('=== usingSplitdata-split mode(by split.tsvpredefined split)===')


                    instrument_type = getattr(config, 'instrument_type', 'all')
                    cache_file = os.path.join(root, f'split_indices_{instrument_type}_split.pt')


                    if os.path.exists(cache_file):
                        print(f'  Loading data split from cache: {cache_file}')
                        split_indices = torch.load(cache_file)
                        train_indices = split_indices['train']
                        val_indices = split_indices['val']
                        test_indices = split_indices['test']
                    else:

                        split_file = os.path.join(root, 'split.tsv')
                        if not os.path.exists(split_file):
                            raise FileNotFoundError(f"Splitmode requires split.tsvfile,  but not found: {split_file}")

                        split_df = pd.read_csv(split_file, sep='\t')
                        gymid_to_split = dict(zip(split_df['name'], split_df['split']))
                        print(f'  from split.tsvloading {len(gymid_to_split)}  GymID split information')


                        print(f'  Splitting dataset(the first run may be slow, results will be cached)...')
                        train_indices = []
                        val_indices = []
                        test_indices = []
                        missing_gymids = []

                        for idx in tqdm(range(len(dataset)), desc='  splitdata'):
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
                            print(f'  WARNING: {len(missing_gymids)}  samples have no matching splitlabels')


                        split_indices = {
                            'train': train_indices,
                            'val': val_indices,
                            'test': test_indices
                        }
                        torch.save(split_indices, cache_file)
                        print(f'  Data split cached at: : {cache_file}')

                    subsets = {
                        'train': Subset(dataset, train_indices),
                        'val': Subset(dataset, val_indices),
                        'test': Subset(dataset, test_indices)
                    }

                    print(f'  Train: {len(train_indices)} samplethis  ({len(train_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Val: {len(val_indices)} samplethis  ({len(val_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Test: {len(test_indices)} samplethis  ({len(test_indices)/len(dataset)*100:.2f}%)')

                    return dataset, subsets

                else:

                    print('=== usingNatMSdata-split mode(by split_natms.tsvpredefined split)===')


                    instrument_type = getattr(config, 'instrument_type', 'all')
                    cache_file = os.path.join(root, f'split_indices_natms_{instrument_type.lower()}.pt')


                    if os.path.exists(cache_file):
                        print(f'  Loading data split from cache: {cache_file}')
                        split_indices = torch.load(cache_file)
                        train_indices = split_indices['train']
                        val_indices = split_indices['val']
                        test_indices = split_indices['test']
                    else:

                        split_file = os.path.join(root, 'split_natms.tsv')
                        if not os.path.exists(split_file):
                            raise FileNotFoundError(f"NatMSmode requires split_natms.tsvfile,  but not found: {split_file}")

                        split_df = pd.read_csv(split_file, sep='\t')
                        gymid_to_split = dict(zip(split_df['name'], split_df['split']))
                        print(f'  from split_natms.tsvloading {len(gymid_to_split)}  GymID split information')


                        print(f'  Splitting dataset(the first run may be slow, results will be cached)...')
                        train_indices = []
                        val_indices = []
                        test_indices = []
                        missing_gymids = []

                        for idx in range(len(dataset)):
                            try:

                                sample_info = dataset.get_raw(idx)
                                ms_file_path = sample_info.get('ms_file_path', None)

                                if ms_file_path and os.path.exists(ms_file_path):

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
                            print(f'  WARNING: {len(missing_gymids)}  samples have no matching splitlabels')


                        split_indices = {
                            'train': train_indices,
                            'val': val_indices,
                            'test': test_indices
                        }
                        torch.save(split_indices, cache_file)
                        print(f'  Data split cached at: : {cache_file}')

                    subsets = {
                        'train': Subset(dataset, train_indices),
                        'val': Subset(dataset, val_indices),
                        'test': Subset(dataset, test_indices)
                    }

                    print(f'  Train: {len(train_indices)} samplethis  ({len(train_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Val: {len(val_indices)} samplethis  ({len(val_indices)/len(dataset)*100:.2f}%)')
                    print(f'  Test: {len(test_indices)} samplethis  ({len(test_indices)/len(dataset)*100:.2f}%)')

                    return dataset, subsets
            else:

                total_size = len(dataset)
                indices = list(range(total_size))
                np.random.seed(2023)
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
    """MolecularDataset implementation."""

    def __init__(self, root, path_dict, dataset_name='natgen', transform=None, data_subset_ratio=1.0,
                 use_spectrum=False, instrument_type='all', spectrum_config=None,
                 use_spectrum_in_training=None,
                 use_pretrained_embeddings=False, pretrained_embeddings_path=None,
                 pretrained_embeddings_paths=None):
        super().__init__()
        self.root = root
        self.dataset_name = dataset_name
        self.sdf_path = os.path.join(root, path_dict['sdf'])
        self.data_subset_ratio = data_subset_ratio


        self.use_spectrum = use_spectrum
        self.instrument_type = instrument_type
        self.spectrum_config = spectrum_config if spectrum_config else DEFAULT_SPECTRUM_CONFIG
        self.use_spectrum_in_training = use_spectrum_in_training if use_spectrum_in_training is not None else use_spectrum


        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.pretrained_embeddings_path = pretrained_embeddings_path
        self.pretrained_embeddings_paths = pretrained_embeddings_paths
        self.use_multi_instrument = pretrained_embeddings_paths is not None


        print(f"[DEBUG Dataset Init] dataset_name: {self.dataset_name}")
        print(f"[DEBUG Dataset Init] use_spectrum: {self.use_spectrum}")
        print(f"[DEBUG Dataset Init] use_spectrum_in_training: {self.use_spectrum_in_training}")
        print(f"[DEBUG Dataset Init] By default, only molecules with pretrained features are processed")
        print(f"[DEBUG Dataset Init] instrument_type: {self.instrument_type}")
        print(f"[DEBUG Dataset Init] use_pretrained_embeddings: {self.use_pretrained_embeddings}")
        print(f"[DEBUG Dataset Init] use_multi_instrument: {self.use_multi_instrument}")


        suffix_parts = [self.dataset_name]
        if data_subset_ratio < 1.0:
            suffix_parts.append(f"{int(data_subset_ratio * 100)}pct")

        if self.use_multi_instrument:
            suffix_parts.append("spectrum_multi_instrument")
        else:
            suffix_parts.append(f"spectrum_{instrument_type.lower()}")
        if use_pretrained_embeddings:
            suffix_parts.append("pretrained")

        suffix = "_" + "_".join(suffix_parts)


        if self.dataset_name == 'msg':

            msg_data_dir = os.path.join(root, 'msg_data')
            os.makedirs(msg_data_dir, exist_ok=True)
            self.processed_path = os.path.join(msg_data_dir, path_dict['processed'].replace('.lmdb', f'{suffix}.lmdb'))
        else:

            self.processed_path = os.path.join(root, path_dict['processed'].replace('.lmdb', f'{suffix}.lmdb'))

        self.molid2idx_path = self.processed_path[:self.processed_path.find('.lmdb')]+'_molid2idx.pt'

        self.transform = transform
        self.db = None
        self.keys = None


        self.spectrum_cache = None
        self.pretrained_cache = None
        self.mol_spectrum_mapping = {}
        self.cache_stats = {}
        self.cache_metadata = {}


        print("Loading the pretrained-feature cache to select molecules with available features...")
        self._setup_pretrained_embeddings()

        if (not os.path.exists(self.processed_path)) or (not os.path.exists(self.molid2idx_path)):
            self._process()
            self._precompute_molid2idx()
        self.molid2idx = torch.load(self.molid2idx_path)

    def _setup_spectrum_data(self):
        """_setup_spectrum_data implementation."""

        if self.dataset_name == 'natgen':
            spectrum_cache_path = os.path.join(self.root, f'spectrum_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')
        elif self.dataset_name == 'msg':

            msg_data_dir = os.path.join(self.root, 'msg_data')
            os.makedirs(msg_data_dir, exist_ok=True)
            spectrum_cache_path = os.path.join(msg_data_dir, f'spectrum_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')
        else:
            spectrum_cache_path = os.path.join(self.root, f'spectrum_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')

        if os.path.exists(spectrum_cache_path):
            print(f"Loading existing mass spectrumcache: {spectrum_cache_path}")
            self.spectrum_cache = load_spectrum_cache(spectrum_cache_path)
        else:
            print(f"Creating new mass spectrumcache...")

            if self.dataset_name == 'natgen':
                spec_dir = os.path.join(self.root, 'natgen_matched_spec_files')
            elif self.dataset_name == 'msg':
                spec_dir = os.path.join(self.root, 'msg_matched_spec_files')
            else:
                raise ValueError(f"unsupported datasettype: {self.dataset_name}")

            if os.path.exists(spec_dir):
                mol_spec_mapping = build_mol_spectrum_mapping(
                    self.sdf_path,
                    spec_dir,
                    self.instrument_type
                )

                self.spectrum_cache = create_spectrum_cache(
                    mol_spec_mapping,
                    self.spectrum_config,
                    spectrum_cache_path
                )
            else:
                print(f"WARNING: Spectrum directory does not exist: {spec_dir}")
                print("Training will continue without spectrum conditioning")
                self.use_spectrum = False
                self.spectrum_cache = None


        self.spectrum_mol_ids = None
        if self.spectrum_cache:
            self.spectrum_mol_ids = set(self.spectrum_cache.keys())
            print(f"found {len(self.spectrum_mol_ids)}  with mass spectrumdata molecule")
            print("Only molecules with mass spectra will be processed")

    def _setup_pretrained_embeddings(self):
        """_setup_pretrained_embeddings implementation."""

        if self.use_multi_instrument and self.pretrained_embeddings_paths:
            print("[INFO] Loading pretrained features in multi-instrument mode...")


            if self.dataset_name == 'msg':
                msg_data_dir = os.path.join(self.root, 'msg_data')
                os.makedirs(msg_data_dir, exist_ok=True)
                pretrained_cache_path = os.path.join(msg_data_dir, f'pretrained_cache_{self.dataset_name}_multi_instrument.pkl')
            else:
                pretrained_cache_path = os.path.join(self.root, f'pretrained_cache_{self.dataset_name}_multi_instrument.pkl')

            if os.path.exists(pretrained_cache_path):
                print(f"Loading existing multi-instrumentpretrained featurescache: {pretrained_cache_path}")
                with open(pretrained_cache_path, 'rb') as f:
                    loaded_cache = pickle.load(f)


                if isinstance(loaded_cache, dict) and loaded_cache.get('version') == '2.0':
                    print(f"[INFO] Detected unified cache format v2.0")
                    self.pretrained_cache = loaded_cache['pretrained_features']
                    self.mol_spectrum_mapping = loaded_cache.get('mol_spectrum_mapping', {})
                    self.cache_stats = loaded_cache.get('stats', {})
                    self.cache_metadata = loaded_cache.get('metadata', {})
                    print(f"[INFO] Cache statistics: {self.cache_stats.get('total_molecules', 0)}  molecules, {self.cache_stats.get('total_features', 0)}  features")
                else:

                    self.pretrained_cache = loaded_cache
                    self.mol_spectrum_mapping = {}
                    self.cache_stats = {}
                    self.cache_metadata = {}
            else:
                print(f"Creating new multi-instrumentpretrained featurescache...")

                if self.dataset_name == 'natgen':
                    spec_dir = os.path.join(self.root, 'natgen_matched_spec_files')
                elif self.dataset_name == 'msg':
                    spec_dir = os.path.join(self.root, 'msg_matched_spec_files')
                else:
                    raise ValueError(f"unsupported datasettype: {self.dataset_name}")

                if os.path.exists(spec_dir):
                    self.pretrained_cache = create_multi_instrument_pretrained_cache(
                        self.sdf_path,
                        spec_dir,
                        self.pretrained_embeddings_paths,
                        pretrained_cache_path
                    )
                else:
                    print(f"WARNING: Spectrum directory does not exist: {spec_dir}")
                    self.use_spectrum = False
                    self.use_pretrained_embeddings = False
                    self.pretrained_cache = None


        elif self.pretrained_embeddings_path and os.path.exists(self.pretrained_embeddings_path):
            print("[INFO] Loading pretrained features in single-instrument mode...")


            if self.dataset_name == 'natgen':
                pretrained_cache_path = os.path.join(self.root, f'pretrained_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')
            elif self.dataset_name == 'msg':
                msg_data_dir = os.path.join(self.root, 'msg_data')
                os.makedirs(msg_data_dir, exist_ok=True)
                pretrained_cache_path = os.path.join(msg_data_dir, f'pretrained_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')
            else:
                pretrained_cache_path = os.path.join(self.root, f'pretrained_cache_{self.dataset_name}_{self.instrument_type.lower()}.pkl')

            if os.path.exists(pretrained_cache_path):
                print(f"Loading existing pretrained featurescache: {pretrained_cache_path}")
                with open(pretrained_cache_path, 'rb') as f:
                    self.pretrained_cache = pickle.load(f)
            else:
                print(f"Creating new pretrained featurescache...")
                if self.dataset_name == 'natgen':
                    spec_dir = os.path.join(self.root, 'natgen_matched_spec_files')
                elif self.dataset_name == 'msg':
                    spec_dir = os.path.join(self.root, 'msg_matched_spec_files')
                else:
                    raise ValueError(f"unsupported datasettype: {self.dataset_name}")

                if os.path.exists(spec_dir):
                    self.pretrained_cache = create_pretrained_spectrum_cache(
                        self.sdf_path,
                        spec_dir,
                        self.pretrained_embeddings_path,
                        self.instrument_type,
                        pretrained_cache_path
                    )
                else:
                    print(f"WARNING: Spectrum directory does not exist: {spec_dir}")
                    self.use_spectrum = False
                    self.use_pretrained_embeddings = False
                    self.pretrained_cache = None
        else:
            print(f"WARNING: Pretrained-feature file does not exist")
            print("Training will continue without spectrum conditioning")
            self.use_spectrum = False
            self.use_pretrained_embeddings = False
            self.pretrained_cache = None


        self.spectrum_mol_ids = None
        if self.pretrained_cache:
            self.spectrum_mol_ids = set(self.pretrained_cache.keys())
            print(f"found {len(self.spectrum_mol_ids)}  entries with pretrained featuresmolecule")
            print("Only molecules with pretrained features will be processed")
        else:
            print("WARNING: no with pretrained featurescache, will processingall with molecule")

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


        sdf_files = glob.glob(os.path.join(self.sdf_path, "*.sdf"))
        if len(sdf_files) == 0:
            raise ValueError(f"No SDF files found in {self.sdf_path}")


        if self.data_subset_ratio < 1.0:
            np.random.seed(2023)
            np.random.shuffle(sdf_files)
            num_files_to_process = int(len(sdf_files) * self.data_subset_ratio)
            sdf_files = sdf_files[:num_files_to_process]
            print(f"Selected {num_files_to_process} out of {len(glob.glob(os.path.join(self.sdf_path, '*.sdf')))} SDF files for processing")
        else:
            print(f"Processing all {len(sdf_files)} SDF files")


        print("\n[INFO] First pass: Automatically detecting atom types in the dataset...")
        all_elements = set()
        valid_mol_ids = set()

        for sdf_file in tqdm(sdf_files, desc='Scanning atom types'):
            mol_name = os.path.splitext(os.path.basename(sdf_file))[0]
            mol_id = mol_name


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


        atomic_num_to_symbol = {
            1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 33: 'As', 34: 'Se', 35: 'Br', 53: 'I'
        }


        supported_elements = sorted(list(all_elements))
        element_symbols = [atomic_num_to_symbol.get(z, f'Z{z}') for z in supported_elements]

        print(f"\n[INFO] Detected  {len(supported_elements)}  atom types:")
        print(f"       atomic numbers: {supported_elements}")
        print(f"       element symbols: {element_symbols}")
        print(f"       valid molecules: {len(valid_mol_ids)}")


        self.detected_atomic_numbers = supported_elements
        supported_elements_set = set(supported_elements)


        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )

        num_skipped = 0
        num_processed = 0

        print("\n[INFO] Second pass: Processing molecular data...")
        with db.begin(write=True, buffers=True) as txn:
            for sdf_file in tqdm(sdf_files, desc='Processing SDF files'):
                try:

                    mol_name = os.path.splitext(os.path.basename(sdf_file))[0]
                    mol_id = mol_name


                    if self.spectrum_mol_ids and mol_id not in self.spectrum_mol_ids:
                        num_skipped += 1
                        continue


                    suppl = Chem.SDMolSupplier(sdf_file)


                    if len(suppl) == 0:
                        num_skipped += 1
                        continue


                    mol = suppl[0]
                    if mol is None:
                        num_skipped += 1
                        continue


                    mol = Chem.RemoveAllHs(mol)


                    smiles = Chem.MolToSmiles(mol)


                    if mol.GetNumAtoms() == 0:
                        num_skipped += 1
                        continue


                    mol_elements = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
                    if not mol_elements.issubset(supported_elements_set):
                        unsupported = mol_elements - supported_elements_set
                        num_skipped += 1
                        continue


                    if mol.GetNumAtoms() > 100:
                        num_skipped += 1
                        continue

                    if mol.GetNumAtoms() < 5:
                        num_skipped += 1
                        continue


                    confs_list = [mol]


                    ligand_dict = parse_conf_list(confs_list, smiles=smiles)
                    if ligand_dict['num_confs'] == 0:
                        print(f"Warning: No valid conformers found in {sdf_file}")
                        num_skipped += 1
                        continue


                    ligand_dict = torchify_dict(ligand_dict)
                    data = Drug3DData.from_drug3d_dicts(ligand_dict)


                    data.smiles = smiles
                    data.mol_id = mol_id
                    data.source_file = sdf_file


                    if self.use_spectrum_in_training:
                        if self.use_pretrained_embeddings and self.pretrained_cache is not None and mol_id in self.pretrained_cache:

                            pretrained_list = self.pretrained_cache[mol_id]


                            for feat_idx, feature_entry in enumerate(pretrained_list):

                                data_with_pretrained = Drug3DData.from_drug3d_dicts(ligand_dict)
                                data_with_pretrained.smiles = smiles
                                data_with_pretrained.mol_id = f"{mol_id}_feat{feat_idx}"
                                data_with_pretrained.original_mol_id = mol_id
                                data_with_pretrained.source_file = sdf_file


                                data_with_pretrained.has_spectrum = True
                                data_with_pretrained.spec_data = None
                                data_with_pretrained.spec_env = None
                                data_with_pretrained.feature_index = feat_idx


                                if self.use_multi_instrument and isinstance(feature_entry, dict):
                                    data_with_pretrained.pretrained_embedding = torch.from_numpy(feature_entry['embedding']).float()
                                    data_with_pretrained.instrument_type = feature_entry['instrument_type']
                                    data_with_pretrained.ionization = feature_entry['ionization']
                                    data_with_pretrained.instrument_type_idx = feature_entry['instrument_type_idx']
                                    data_with_pretrained.ionization_type_idx = feature_entry['ionization_type_idx']
                                else:

                                    data_with_pretrained.pretrained_embedding = torch.from_numpy(feature_entry).float()
                                    data_with_pretrained.instrument_type = self.instrument_type
                                    data_with_pretrained.ionization = '[M+H]+'
                                    data_with_pretrained.instrument_type_idx = INSTRUMENT_TYPES.index(self.instrument_type) if self.instrument_type in INSTRUMENT_TYPES else INSTRUMENT_TYPES.index('NONE')
                                    data_with_pretrained.ionization_type_idx = 0


                                unique_key = f"{mol_id}_feat{feat_idx}"
                                txn.put(
                                    key=unique_key.encode('utf-8'),
                                    value=pickle.dumps(data_with_pretrained)
                                )
                                num_processed += 1

                        elif not self.use_pretrained_embeddings and self.spectrum_cache is not None and mol_id in self.spectrum_cache:

                            spec_list = self.spectrum_cache[mol_id]
                            print(f"[DEBUG Preprocessing] Found {len(spec_list)} spectra for mol_id: {mol_id}")


                            for spec_idx, spectrum_data in enumerate(spec_list):

                                data_with_spec = Drug3DData.from_drug3d_dicts(ligand_dict)
                                data_with_spec.smiles = smiles
                                data_with_spec.mol_id = f"{mol_id}_spec{spec_idx}"
                                data_with_spec.original_mol_id = mol_id
                                data_with_spec.source_file = sdf_file


                                data_with_spec.has_spectrum = True
                                data_with_spec.spec_data = torch.from_numpy(spectrum_data['spec'][:, 0]).float()
                                data_with_spec.spec_env = torch.from_numpy(spectrum_data['env']).float()
                                data_with_spec.instrument_type = self.instrument_type
                                data_with_spec.spectrum_index = spec_idx


                                unique_key = f"{mol_id}_spec{spec_idx}"
                                txn.put(
                                    key=unique_key.encode('utf-8'),
                                    value=pickle.dumps(data_with_spec)
                                )
                                num_processed += 1
                                print(f"[DEBUG Preprocessing] Stored sample with spectrum: {unique_key}")

                            print(f"[DEBUG Preprocessing] Created {len(spec_list)} training samples for molecule {mol_id}")
                        else:

                            data.has_spectrum = False
                            data.spec_data = None
                            data.spec_env = None
                            data.pretrained_embedding = None
                            data.instrument_type = None


                            txn.put(
                                key=str(mol_id).encode('utf-8'),
                                value=pickle.dumps(data)
                            )
                            num_processed += 1
                            print(f"[DEBUG Preprocessing] Stored sample without spectrum: {mol_id}")
                    else:

                        data.has_spectrum = False
                        data.spec_data = None
                        data.spec_env = None
                        data.pretrained_embedding = None
                        data.instrument_type = None


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


        if self.use_spectrum_in_training:
            if self.use_pretrained_embeddings and self.pretrained_cache:
                total_molecules = len([f for f in sdf_files if os.path.splitext(os.path.basename(f))[0] in self.pretrained_cache])
                total_features = sum(len(feat_list) for feat_list in self.pretrained_cache.values())
                print(f'=== Pretrained-feature augmentation statistics ===')
                print(f'molecules with pretrained features: {total_molecules}')
                print(f'total pretrained features: {total_features}')
                print(f'average pretrained features per molecule: {total_features/total_molecules if total_molecules > 0 else 0:.2f}')
                print(f'augmentation factor: {total_features/total_molecules if total_molecules > 0 else 1:.2f}x')
            elif not self.use_pretrained_embeddings and self.spectrum_cache:
                total_molecules = len([f for f in sdf_files if os.path.splitext(os.path.basename(f))[0] in self.spectrum_cache])
                total_spectra = sum(len(spec_list) for spec_list in self.spectrum_cache.values())
                print(f'=== Spectrum-data augmentation statistics ===')
                print(f'molecules with spectra: {total_molecules}')
                print(f'total spectra: {total_spectra}')
                print(f'average spectra per molecule: {total_spectra/total_molecules if total_molecules > 0 else 0:.2f}')
                print(f'augmentation factor: {total_spectra/total_molecules if total_molecules > 0 else 1:.2f}x')

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


        global debug_getitem_counter
        if not 'debug_getitem_counter' in globals():
            debug_getitem_counter = 0
        debug_getitem_counter += 1


        should_debug = debug_getitem_counter <= 10

        if should_debug:
            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] Processing sample {idx}")
            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] use_pretrained_embeddings: {self.use_pretrained_embeddings}")
            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] Loaded data.has_spectrum: {getattr(data, 'has_spectrum', 'NOT SET')}")


        if self.use_spectrum_in_training:
            if self.use_pretrained_embeddings:

                mol_id = str(data.mol_id) if hasattr(data, 'mol_id') else str(data.original_mol_id) if hasattr(data, 'original_mol_id') else None

                if should_debug:
                    print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] pretrained-feature mode - mol_id: {mol_id}")
                    print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] pretrained_cache available: {self.pretrained_cache is not None}")
                    if self.pretrained_cache:
                        print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] pretrained_cachesize: {len(self.pretrained_cache)}")
                        print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] mol_idin cache in : {mol_id in self.pretrained_cache if mol_id else False}")

                if mol_id and self.pretrained_cache and mol_id in self.pretrained_cache:

                    pretrained_embedding = get_pretrained_embedding_for_molecule(mol_id, self.pretrained_cache, mode='random')
                    if pretrained_embedding is not None:
                        data.has_spectrum = True
                        data.pretrained_embedding = torch.from_numpy(pretrained_embedding).float()  # [1024]
                        data.spec_data = None
                        data.spec_env = None

                        if should_debug:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] successfully attached pretrained feature: shape={data.pretrained_embedding.shape}")
                    else:
                        data.has_spectrum = False
                        data.pretrained_embedding = None
                        data.spec_data = None
                        data.spec_env = None

                        if should_debug:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] failed to retrieve pretrained feature")
                else:
                    data.has_spectrum = False
                    data.pretrained_embedding = None
                    data.spec_data = None
                    data.spec_env = None

                    if should_debug:
                        if not mol_id:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] mol_id is empty")
                        elif not self.pretrained_cache:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] pretrained_cache is empty")
                        elif mol_id not in self.pretrained_cache:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] mol_id {mol_id}  not in pretrained_cache in ")
            else:

                if hasattr(data, 'spec_data'):
                    if should_debug:
                        print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] spec_data is not None: {data.spec_data is not None}")
                        if data.spec_data is not None:
                            print(f"[DEBUG Dataset __getitem__ {debug_getitem_counter}] spec_data shape: {data.spec_data.shape}")
                    data.pretrained_embedding = None
                else:
                    data.has_spectrum = False
                    data.spec_data = None
                    data.spec_env = None
                    data.pretrained_embedding = None
        else:

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
    """collate_with_pretrained_features implementation."""
    from torch_geometric.data import Batch


    has_pretrained = any(
        hasattr(data, 'pretrained_embedding') and data.pretrained_embedding is not None
        for data in batch
    )


    spectrum_info = []
    cleaned_data_list = []

    for data in batch:

        info = {
            'has_spectrum': getattr(data, 'has_spectrum', False),
            'pretrained_embedding': getattr(data, 'pretrained_embedding', None),
            'spec_data': getattr(data, 'spec_data', None),
            'spec_env': getattr(data, 'spec_env', None),
            'instrument_type_idx': getattr(data, 'instrument_type_idx', 2),
            'ionization_type_idx': getattr(data, 'ionization_type_idx', 0),
        }
        spectrum_info.append(info)


        data_copy = data.clone()


        attrs_to_remove = ['pretrained_embedding', 'spec_data', 'spec_env', 'has_spectrum',
                          'instrument_type', 'ionization', 'instrument_type_idx', 'ionization_type_idx',
                          'spectrum_index', 'original_mol_id', 'feature_index']
        for attr in attrs_to_remove:
            if hasattr(data_copy, attr):
                delattr(data_copy, attr)

        cleaned_data_list.append(data_copy)


    follow_batch = ['node_type', 'halfedge_type']
    exclude_keys = ['orig_keys', 'pos_all_confs', 'smiles', 'num_confs', 'i_conf_list',
                   'bond_index', 'bond_type', 'num_bonds', 'num_atoms']

    batched_data = Batch.from_data_list(cleaned_data_list, follow_batch=follow_batch, exclude_keys=exclude_keys)


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

    return batched_data


def collate_mol2d(batch):
    """collate_mol2d implementation."""
    from torch_geometric.data import Batch


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


        data_copy = data.clone()
        attrs_to_remove = ['pretrained_embedding', 'has_spectrum', 'instrument_type_idx',
                          'ionization_type_idx', 'smiles', 'mol_id']
        for attr in attrs_to_remove:
            if hasattr(data_copy, attr):
                delattr(data_copy, attr)
        cleaned_data_list.append(data_copy)


    follow_batch = ['node_type', 'halfedge_type']
    batched_data = Batch.from_data_list(cleaned_data_list, follow_batch=follow_batch)


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
    """MSFileDataset implementation."""

    def __init__(self, root, path_dict=None, transform=None, data_subset_ratio=1.0,
                 instrument_type='all', data_split_mode='natms', num_workers=8):
        """__init__ implementation."""
        super().__init__()
        self.root = root
        self.transform = transform
        self.data_subset_ratio = data_subset_ratio
        self.instrument_type = instrument_type
        self.data_split_mode = data_split_mode
        self.num_workers = num_workers


        self.ms_base_dir = os.path.join(root, 'msg_processed')


        self.embedding_paths = {
            'Orbitrap': os.path.join(self.ms_base_dir, 'Orbitrap_embedding', 'batch_embeddings.pkl'),
            'QTOF': os.path.join(self.ms_base_dir, 'QTOF_embedding', 'batch_embeddings.pkl'),
            'NONE': os.path.join(self.ms_base_dir, 'NONE_embedding', 'batch_embeddings.pkl')
        }


        suffix = f"msfile_{instrument_type.lower()}_{data_split_mode}"
        if data_subset_ratio < 1.0:
            suffix += f"_{int(data_subset_ratio * 100)}pct"
        self.processed_path = os.path.join(root, f'processed_{suffix}.lmdb')
        self.molid2idx_path = self.processed_path.replace('.lmdb', '_molid2idx.pt')

        self.db = None
        self.keys = None


        print(f"[MSFileDataset] loadingpretrained features...")
        self.pretrained_embeddings = {}
        self._load_pretrained_embeddings()


        self.smiles2indices_path = self.molid2idx_path.replace('_molid2idx.pt', '_smiles2indices.pt')
        if not os.path.exists(self.processed_path) or not os.path.exists(self.molid2idx_path) or not os.path.exists(self.smiles2indices_path):
            self._process_fast()
            self._precompute_molid2idx()

        self.molid2idx = torch.load(self.molid2idx_path)
        self.smiles2indices = torch.load(self.smiles2indices_path)

    def _load_pretrained_embeddings(self):
        """_load_pretrained_embeddings implementation."""
        if self.instrument_type == 'all':
            instruments_to_load = ['Orbitrap', 'QTOF', 'NONE']
        else:
            instruments_to_load = [self.instrument_type]

        for inst in instruments_to_load:
            path = self.embedding_paths.get(inst)
            if path and os.path.exists(path):
                print(f"  loading {inst} pretrained features: {path}")
                with open(path, 'rb') as f:
                    embeddings = pickle.load(f)
                self.pretrained_embeddings[inst] = embeddings
                print(f"    loading {len(embeddings)}  features")
            else:
                print(f"  WARNING: {inst} Pretrained-feature file does not exist: {path}")

    def _parse_ms_file_fast(self, ms_file_path):
        """_parse_ms_file_fast implementation."""
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
        """_smiles_to_graph implementation."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.RemoveAllHs(mol)
            num_atoms = mol.GetNumAtoms()
            if num_atoms == 0:
                return None


            node_type = [atom.GetAtomicNum() for atom in mol.GetAtoms()]


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
        """_process_fast implementation."""
        from torch_geometric.data import Data

        print(f"[MSFileDataset] Processing MSfile...")
        print(f"  instrument type: {self.instrument_type}")
        print(f"  Data fraction: {self.data_subset_ratio:.1%}")


        if self.instrument_type == 'all':
            instruments_to_process = ['Orbitrap', 'QTOF', 'NONE']
        else:
            instruments_to_process = [self.instrument_type]


        entries_to_process = []
        for inst in instruments_to_process:
            if inst not in self.pretrained_embeddings:
                continue
            inst_dir = os.path.join(self.ms_base_dir, inst)
            if not os.path.exists(inst_dir):
                continue


            for emb_key in self.pretrained_embeddings[inst].keys():
                ms_file = os.path.join(inst_dir, f"{emb_key}.ms")
                if os.path.exists(ms_file):
                    entries_to_process.append((ms_file, inst, emb_key))

        print(f"  found {len(entries_to_process)}  entries with pretrained featuresMSfile")


        if self.data_subset_ratio < 1.0:
            np.random.seed(2023)
            np.random.shuffle(entries_to_process)
            num_files = int(len(entries_to_process) * self.data_subset_ratio)
            entries_to_process = entries_to_process[:num_files]
            print(f"  select {num_files}  files selected for processing")


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

        print("\n[INFO] Processing molecular data(single pass)...")
        with db.begin(write=True, buffers=True) as txn:
            for ms_file, inst, emb_key in tqdm(entries_to_process, desc='processingmolecule'):
                try:

                    smiles, ionization = self._parse_ms_file_fast(ms_file)
                    if not smiles:
                        num_skipped += 1
                        continue


                    graph_data = self._smiles_to_graph(smiles)
                    if graph_data is None:
                        num_skipped += 1
                        continue


                    all_elements.update(graph_data['node_type'].tolist())


                    embedding = self.pretrained_embeddings[inst][emb_key]
                    if embedding.ndim == 2:
                        embedding = embedding.squeeze(0)


                    data = Data(
                        node_type=torch.from_numpy(graph_data['node_type']),
                        edge_index=torch.from_numpy(graph_data['edge_index']),
                        edge_type=torch.from_numpy(graph_data['edge_type']),
                        num_nodes=graph_data['num_atoms'],
                    )


                    data.smiles = graph_data['smiles']
                    data.mol_id = f"{inst}_{emb_key}"
                    data.ms_file_path = ms_file
                    data.has_spectrum = True
                    data.pretrained_embedding = torch.from_numpy(embedding).float()


                    data.instrument_type_idx = INSTRUMENT_TYPES.index(inst) if inst in INSTRUMENT_TYPES else INSTRUMENT_TYPES.index('NONE')
                    ionization_clean = ionization if ionization else '[M+H]+'
                    data.ionization_type_idx = IONIZATION_TYPES.index(ionization_clean) if ionization_clean in IONIZATION_TYPES else 0


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


        atomic_num_to_symbol = {
            1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 33: 'As', 34: 'Se', 35: 'Br', 53: 'I'
        }
        supported_elements = sorted(list(all_elements))
        element_symbols = [atomic_num_to_symbol.get(z, f'Z{z}') for z in supported_elements]

        print(f"\n[INFO] Processing complete:")
        print(f"       processed successfully: {num_processed}")
        print(f"       skipped: {num_skipped}")
        print(f"       Detected  {len(supported_elements)}  atom types: {element_symbols}")

    def _connect_db(self):
        """_connect_db implementation."""
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
        """_precompute_molid2idx implementation."""
        self._connect_db()
        molid2idx = {}
        smiles2indices = {}

        for i, key in enumerate(self.keys):
            data = pickle.loads(self.db.begin().get(key))
            if data is None:
                continue
            mol_id = data.mol_id
            molid2idx[mol_id] = i


            smiles = getattr(data, 'smiles', None)
            if smiles:
                if smiles not in smiles2indices:
                    smiles2indices[smiles] = []
                smiles2indices[smiles].append(i)

        torch.save(molid2idx, self.molid2idx_path)

        smiles2indices_path = self.molid2idx_path.replace('_molid2idx.pt', '_smiles2indices.pt')
        torch.save(smiles2indices, smiles2indices_path)
        self._close_db()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def get_raw(self, idx):
        """get_raw implementation."""
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))


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
    """SmilesDataset implementation."""

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
        self.detected_atomic_numbers = list(self.atomic_numbers)
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


        if not os.path.isfile(smiles_file):
            print(f"[SmilesDataset] {smiles_file}  is unavailable; automatically building pretraining  SMILES csv...")
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


        self.subsets = self._build_subsets(split_seed=split_seed, split_ratio=split_ratio)


    def _read_smiles_file(self):
        """_read_smiles_file implementation."""
        path = self.smiles_file
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SMILES file does not exist: {path}")
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
                    f"in  {path}  does not contain  SMILES/inchi column.expected one of the following columns: {self.SMILES_COLUMN_CANDIDATES}"
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
        """_smi_to_canonical implementation."""
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
        """_smiles_to_graph implementation."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.RemoveAllHs(mol)


            try:
                smiles_canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                mol_c = Chem.MolFromSmiles(smiles_canon)
                if mol_c is not None:
                    mol = Chem.RemoveAllHs(mol_c)
            except Exception:
                pass
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


    def _process(self):
        from torch_geometric.data import Data

        print(f"[SmilesDataset] processing SMILES file: {self.smiles_file}")
        raw_smiles, raw_sources, raw_splits, raw_mol_ids = self._read_smiles_file()
        print(f"  raw entries: {len(raw_smiles)}")


        canonical_list = list(zip(raw_smiles, raw_sources, raw_splits, raw_mol_ids))

        if self.data_subset_ratio < 1.0:
            np.random.seed(2026)
            rng_idx = np.arange(len(canonical_list))
            np.random.shuffle(rng_idx)
            n_keep = int(len(canonical_list) * self.data_subset_ratio)
            canonical_list = [canonical_list[k] for k in rng_idx[:n_keep]]
            print(f"  subsampled at  {self.data_subset_ratio:.1%} select : {n_keep}")


        none_idx = INSTRUMENT_TYPES.index('NONE') if 'NONE' in INSTRUMENT_TYPES else 0
        mhplus_idx = IONIZATION_TYPES.index('[M+H]+') if '[M+H]+' in IONIZATION_TYPES else 0

        os.makedirs(os.path.dirname(self.processed_path) or '.', exist_ok=True)
        db = lmdb.open(
            self.processed_path,
            map_size=200 * (1024 ** 3),
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
            for i, (smi, src, sp, mid) in enumerate(tqdm(canonical_list, desc='  writingLMDB')):
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
        print(f"[SmilesDataset] complete: retained {num_kept}, skipped(unsupported atom type) {num_skipped_atom}, skipped(size) {num_skipped_size}")


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


    def _build_subsets(self, split_seed=2026, split_ratio=(0.95, 0.025, 0.025)):
        n = len(self.keys)

        if any(s is not None for s in self._splits_in_order):
            train_idx, val_idx, test_idx = [], [], []
            for i, sp in enumerate(self._splits_in_order):
                if sp == 'val':
                    val_idx.append(i)
                elif sp == 'test':
                    test_idx.append(i)
                else:
                    train_idx.append(i)
            print(f"[SmilesDataset] split (by  csv column): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        else:
            rng = np.random.default_rng(split_seed)
            perm = rng.permutation(n)
            r_train, r_val, _ = split_ratio
            n_train = int(n * r_train)
            n_val = int(n * r_val)
            train_idx = perm[:n_train].tolist()
            val_idx = perm[n_train:n_train + n_val].tolist()
            test_idx = perm[n_train + n_val:].tolist()
            print(f"[SmilesDataset] split (random fractions): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        return {
            'train': Subset(self, train_idx),
            'val':   Subset(self, val_idx),
            'test':  Subset(self, test_idx),
        }


# =====================================================================






# =====================================================================

DEFAULT_PRETRAIN_SOURCES = {
    'hmdb':    'https://hmdb.ca/system/downloads/current/structures.zip',
    'dsstox':  'https://clowder.edap-cluster.com/api/files/6616d8d7e4b063812d70fc95/blob',
    'coconut': 'https://coconut.s3.uni-jena.de/prod/downloads/2025-03/coconut_csv-03-2025.zip',
    'moses':   'https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv',
}

DEFAULT_FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P', 'Br', 'I', 'B', 'Si', 'Se'}


def _download_file(url, dst):
    """_download_file implementation."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"  Existing file; skipping download: {dst} ({size_mb:.1f} MB)")
        return dst
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    print(f"  download {url} -> {dst}")
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get('Content-Length') or 0)
            chunk = 1 << 20  # 1 MB
            with open(dst, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True, unit_divisor=1024,
                desc=f"  download {os.path.basename(dst)}", leave=False,
            ) as pbar:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    pbar.update(len(buf))
        return dst
    except Exception as e:
        print(f"  [WARNING] Download failed {url}: {e}")
        return None


def _filter_mol_for_pretrain(mol, max_mw=1500.0, allowed_atoms=None):
    """_filter_mol_for_pretrain implementation."""
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
    """_read_sdf_smiles implementation."""
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
    """_collect_msg_smiles_by_split implementation."""
    out = {'train': [], 'val': [], 'test': []}
    if not msg_split_file or not os.path.isfile(msg_split_file):
        print(f"  [WARNING] MSG split file not found: {msg_split_file}, skipped MSG merge/exclude")
        return out

    try:
        df = pd.read_csv(msg_split_file, sep='\t')
    except Exception as e:
        print(f"  [WARNING] Failed to read the MSG split: {e}")
        return out

    gymid_to_split = dict(zip(df['name'].astype(str), df['split'].astype(str)))

    base_root = os.path.dirname(msg_split_file)
    msg_processed_dir = os.path.join(base_root, 'msg_processed')
    if not os.path.isdir(msg_processed_dir):
        print(f"  [WARNING] not found {msg_processed_dir}/, skipped MSG merge")
        return out


    ms_files = []
    for inst in os.listdir(msg_processed_dir):
        inst_dir = os.path.join(msg_processed_dir, inst)
        if not os.path.isdir(inst_dir) or inst.endswith('_embedding'):
            continue
        for fn in os.listdir(inst_dir):
            if fn.endswith('.ms'):
                ms_files.append(os.path.join(inst_dir, fn))

    for ms_path in tqdm(ms_files, desc='  scanning MSG .ms', leave=False):
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
                        out[split_label].append((cano, inchi, gymid))
                        break
        except Exception:
            continue
    print(f"  MSG split collected(one row per spectrum): train={len(out['train'])}, val={len(out['val'])}, test={len(out['test'])}")
    return out


def build_pretrain_smiles_csv(
    output_csv,
    cache_dir=None,
    msg_split_file=None,
    sources=None,
    allowed_atoms=None,
    max_mw=1500.0,
    max_atoms=None,
    include_mol_id=False,
):
    """build_pretrain_smiles_csv implementation."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(output_csv) or '.', 'raw')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

    sources = sources or DEFAULT_PRETRAIN_SOURCES
    allowed_atoms = allowed_atoms or DEFAULT_FILTER_ATOMS

    print("=" * 60)
    print("[build_pretrain_smiles_csv] Building pretraining  SMILES csv")
    print(f"  Cache directory: {cache_dir}")
    print(f"  Output: {output_csv}")
    print("=" * 60)


    msg_by_split = _collect_msg_smiles_by_split(msg_split_file) if msg_split_file else {
        'train': [], 'val': [], 'test': []
    }
    excluded_inchis = set(inchi for _, inchi, _ in msg_by_split.get('val', []))
    excluded_inchis.update(inchi for _, inchi, _ in msg_by_split.get('test', []))
    print(f"  MSG val+test leakage-exclusion InChI count(after deduplication): {len(excluded_inchis)}")



    seen_inchi = set()
    rows = []  # (smiles, source, split, mol_id)

    for split_label in ('train', 'val', 'test'):
        for cano, inchi, gymid in msg_by_split.get(split_label, []):
            rows.append((cano, 'msg', split_label, gymid))
            if inchi:
                seen_inchi.add(inchi)
    print(f"  MSG merged(one row per spectrum): {len(rows)}  entries")

    def _ingest_smiles_iterable(iter_smis, source_name, total=None):
        kept = 0
        skipped = 0
        pbar = tqdm(iter_smis, desc=f'  {source_name} normalization', total=total, leave=False)
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
            rows.append((cano, source_name, 'train', ''))
            kept += 1
            if (kept + skipped) % 5000 == 0:
                pbar.set_postfix(kept=kept, skipped=skipped)
        return kept, skipped


    if 'hmdb' in sources:
        print("\n[HMDB]")
        sdf_candidates = (
            glob.glob(os.path.join(cache_dir, 'hmdb*.sdf'))
            + glob.glob(os.path.join(cache_dir, 'structures*.sdf'))
        )
        if not sdf_candidates:
            print("  [skipped] not found HMDB sdf file.Download the file manually and place it under  raw/  under : ")
            print(f"    {sources['hmdb']}")
            print(f"    expected filename: {cache_dir}/hmdb.sdf  or  {cache_dir}/structures.sdf")
        else:
            smis = []
            for sdf in sdf_candidates:
                smis.extend(_read_sdf_smiles(sdf))
            print(f"  [HMDB] raw molecules: {len(smis)}")
            before = len(rows)
            kept, skipped = _ingest_smiles_iterable(smis, 'hmdb', total=len(smis))
            print(f"  [HMDB] retained {kept} / skipped {skipped} (csv cumulative +{len(rows) - before})")


    if 'dsstox' in sources:
        print("\n[DSSTox]")
        xlsx_files = (
            glob.glob(os.path.join(cache_dir, 'DSSToxDump*.xlsx'))
            + glob.glob(os.path.join(cache_dir, 'DSSTox', 'DSSToxDump*.xlsx'))
            + glob.glob(os.path.join(cache_dir, 'DSSTox', '*.xlsx'))
        )
        if not xlsx_files:
            print("  [skipped] not found DSSTox xlsx file.Download the file manually and place it under  raw/  under : ")
            print(f"    {sources['dsstox']}")
            print(f"    expected filename: {cache_dir}/DSSTox/DSSToxDump*.xlsx")
        else:
            dss_smis = []
            for fp in tqdm(xlsx_files, desc='  DSSTox xlsx loading'):
                try:
                    df = pd.read_excel(fp)
                    if 'SMILES' in df.columns:
                        dss_smis.extend(df['SMILES'].dropna().astype(str).tolist())
                except Exception as e:
                    print(f"  [WARNING] loading {fp} failed: {e}")
            print(f"  [DSSTox] raw molecules: {len(dss_smis)}")
            before = len(rows)
            kept, skipped = _ingest_smiles_iterable(dss_smis, 'dsstox', total=len(dss_smis))
            print(f"  [DSSTox] retained {kept} / skipped {skipped} (csv cumulative +{len(rows) - before})")


    if 'coconut' in sources:
        print("\n[COCONUT]")
        csv_candidates = glob.glob(os.path.join(cache_dir, 'coconut_csv*.csv'))
        if not csv_candidates:
            print("  [skipped] not found COCONUT csv.Download the file manually and place it under  raw/  under : ")
            print(f"    {sources['coconut']}")
            print(f"    expected filename: {cache_dir}/coconut_csv*.csv")
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
                        print(f"  [WARNING] {fp} without  SMILES/smiles/canonical_smiles column, skipped")
                        continue
                    coconut_smis.extend(df[col].dropna().astype(str).tolist())
                except Exception as e:
                    print(f"  [WARNING] loading {fp} failed: {e}")
            print(f"  [COCONUT] raw molecules: {len(coconut_smis)}")
            before = len(rows)
            kept, skipped = _ingest_smiles_iterable(coconut_smis, 'coconut', total=len(coconut_smis))
            print(f"  [COCONUT] retained {kept} / skipped {skipped} (csv cumulative +{len(rows) - before})")


    if 'moses' in sources:
        print("\n[MOSES]")
        moses_paths = [
            os.path.join(cache_dir, 'moses.csv'),
            os.path.join(cache_dir, 'dataset_v1.csv'),
        ]
        moses_path = next((p for p in moses_paths if os.path.isfile(p)), None)
        if moses_path is None:
            print("  [skipped] not found MOSES csv.Download the file manually and place it under  raw/  under : ")
            print(f"    {sources['moses']}")
            print(f"    expected filename: {cache_dir}/moses.csv  or  {cache_dir}/dataset_v1.csv")
        else:
            try:
                df = pd.read_csv(moses_path)
                col = 'SMILES' if 'SMILES' in df.columns else ('smiles' if 'smiles' in df.columns else None)
                moses_smis = df[col].dropna().astype(str).tolist() if col is not None else []
            except Exception as e:
                print(f"  [WARNING] loading MOSES failed: {e}")
                moses_smis = []
            print(f"  [MOSES] raw molecules (from  {os.path.basename(moses_path)}): {len(moses_smis)}")
            before = len(rows)
            kept, skipped = _ingest_smiles_iterable(moses_smis, 'moses', total=len(moses_smis))
            print(f"  [MOSES] retained {kept} / skipped {skipped} (csv cumulative +{len(rows) - before})")


    print("\n[output]")
    if not rows:
        raise RuntimeError("The pretraining dataset is empty, verify that the download completed successfully")
    df_out = pd.DataFrame(rows, columns=['smiles', 'source', 'split', 'mol_id'])
    if not include_mol_id:
        df_out = df_out[['smiles', 'source', 'split']]
    df_out.to_csv(output_csv, index=False)
    print(f"  total {len(df_out)}  entries  ->  {output_csv}")
    print(f"  source distribution: {df_out['source'].value_counts().to_dict()}")
    print(f"  split distribution: {df_out['split'].value_counts().to_dict()}")
    return output_csv


# ============================================================================


#                                   + labels.tsv + split.tsv)

# ============================================================================


_MS2FORGE_ELEMENTS = ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I"]
_MS2FORGE_PRECURSOR = {
    '[M+H]+':  0,
    '[M-H]-':  1,
    '[M+Na]+': 0,
}


_DIFFMS_ATOM_TYPES = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Si': 5, 'P': 6, 'S': 7,
                       'Cl': 8, 'Br': 9, 'I': 10, 'H': 11}


class _MS2ForgePositionalEncoding:
    """_MS2ForgePositionalEncoding implementation."""
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
    """_formula_str_to_array implementation."""
    counts = np.zeros(len(_MS2FORGE_ELEMENTS), dtype=int)
    for sym, num in re.findall(r'([A-Z][a-z]?)(\d*)', formula_str):
        if sym in _MS2FORGE_ELEMENTS:
            counts[_MS2FORGE_ELEMENTS.index(sym)] += int(num) if num else 1
    return counts


def _encode_peaks_to_formula_array(per_peak_formulas, max_peaks=128):
    """_encode_peaks_to_formula_array implementation."""
    pos_enc = _MS2ForgePositionalEncoding()
    total_dim = len(_MS2FORGE_ELEMENTS) * 16   # 144
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


def _smi_to_ms2forge_graph(smiles):
    """_smi_to_ms2forge_graph implementation."""
    from rdkit.Chem.rdchem import BondType as BT
    from torch_geometric.utils import subgraph
    BOND_TYPES_DEN = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}



    mol_raw = Chem.MolFromSmiles(smiles)
    if mol_raw is None:
        return None
    try:
        smiles_canon = Chem.MolToSmiles(mol_raw, canonical=True, isomericSmiles=False)
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles_canon)
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
    x = x[to_keep][:, :-1]
    if x.size(0) == 0:
        return None
    return x, edge_index, edge_attr


def _zmol_smi_to_graph_worker(smi):
    """_zmol_smi_to_graph_worker implementation."""
    try:
        from rdkit import Chem
        mol_raw = Chem.MolFromSmiles(smi)
        if mol_raw is None:
            return None
        cano = Chem.MolToSmiles(mol_raw, canonical=True, isomericSmiles=False)
    except Exception:
        return None
    res = _smi_to_ms2forge_graph(cano)
    if res is None:
        return None
    x, edge_index, edge_attr = res
    return (cano, x, edge_index, edge_attr)


class DiffMSMSGDataset(Dataset):
    """DiffMSMSGDataset implementation."""

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


        self.spec_files_dir = os.path.join(root, 'spec_files')
        self.subformulae_dir = os.path.join(root, 'subformulae', 'default_subformulae')
        self.labels_path = os.path.join(root, 'labels.tsv')
        self.split_path = os.path.join(root, 'split.tsv')

        for p in (self.spec_files_dir, self.subformulae_dir, self.labels_path, self.split_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"DiffMSMSGDataset path is missing: {p}")

        print(f"[DiffMSMSGDataset] Loading labels and split ...")
        labels_df = pd.read_csv(self.labels_path, sep='\t')
        split_df = pd.read_csv(self.split_path, sep='\t')
        spec2split = dict(zip(split_df['name'], split_df['split']))
        labels_df['split'] = labels_df['spec'].map(spec2split)


        labels_df = labels_df[labels_df['ionization'].isin(_MS2FORGE_PRECURSOR.keys())].copy()

        from models.model import INSTRUMENT_TYPES
        labels_df['instrument_type_idx'] = labels_df['instrument'].map(
            lambda x: INSTRUMENT_TYPES.index(x) if x in INSTRUMENT_TYPES else INSTRUMENT_TYPES.index('NONE')
        )

        labels_df['ionization_type_idx'] = labels_df['ionization'].map(
            lambda x: 0 if x == '[M+H]+' else (1 if x == '[M+Na]+' else 0)
        )


        if data_subset_ratio < 1.0:
            n_keep = int(len(labels_df) * data_subset_ratio)
            labels_df = labels_df.sample(n=n_keep, random_state=2026).reset_index(drop=True)
        else:
            labels_df = labels_df.reset_index(drop=True)

        self.labels_df = labels_df
        print(f"[DiffMSMSGDataset] Total samples: {len(labels_df)}, "
              f"unique smiles: {labels_df['smiles'].nunique()}")
        print(f"[DiffMSMSGDataset] split distribution: {labels_df['split'].value_counts().to_dict()}")


        self._graph_cache = {}


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
        formula = row['formula']


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
        precursor_oh[_MS2FORGE_PRECURSOR[row['ionization']]] = 1.0
        energy_oh = torch.zeros(11)
        energy_oh[0] = 1.0
        sos = torch.cat([precursor_oh, energy_oh], dim=0).view(1, -1)  # [1, 13]

        # ---- 2. SMILES  ->  MS2Forge align graph ----
        if smiles in self._graph_cache:
            x, edge_index, edge_attr = self._graph_cache[smiles]
        else:
            res = _smi_to_ms2forge_graph(smiles)
            if res is None:

                x = torch.zeros(1, 11)
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                edge_attr = torch.zeros(0, 5)
            else:
                x, edge_index, edge_attr = res
            self._graph_cache[smiles] = (x, edge_index, edge_attr)


        from torch_geometric.data import Data as PygData
        d = PygData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=int(x.size(0)),
        )
        d.smiles = smiles
        d.mol_id = spec_id
        d.formula = formula
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


# ============================================================================


_CACHE_VERSION = 'v1'


def _cache_paths(cache_dir, version=_CACHE_VERSION):
    return {
        'zmol': os.path.join(cache_dir, f'zmol_{version}.pt'),
        'zms':  os.path.join(cache_dir, f'zms_{version}.pt'),
        'meta': os.path.join(cache_dir, f'meta_{version}.json'),
    }


def build_zmol_cache(align_ckpt_path, smiles_list, cache_path, device='cpu',
                      batch_size=64, dtype=torch.float16):
    """build_zmol_cache implementation."""
    from models.model import GraphEncoder
    from utils.transforms import _BFN2DIFFMS

    print(f'[zmol cache] building: {len(smiles_list)} unique SMILES  ->  {cache_path}')


    graph_encoder = GraphEncoder(n_layers=4, output_dims={'X': 512}).to(device)
    try:
        sd = torch.load(align_ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(align_ckpt_path, map_location=device)
    sd = sd['model'] if 'model' in sd else sd
    ge_state = {k[len('graph_encoder.'):]: v for k, v in sd.items() if k.startswith('graph_encoder.')}
    miss, unex = graph_encoder.load_state_dict(ge_state, strict=False)
    print(f'  graph_encoder ckpt: loaded={len(ge_state)}, missing={len(miss)}, unexpected={len(unex)}')
    graph_encoder.eval()
    m = type('Wrapper', (), {'graph_encoder': graph_encoder})()



    from torch_geometric.data import Data as PygData

    valid_smis = []      # canonical SMILES(cache key)
    pyg_data_list = []
    n_dup = 0
    seen = set()
    for smi in tqdm(smiles_list, desc='  RDKit Parsing SMILES  ->  graph', unit='mol', mininterval=2.0):
        try:
            mol_raw = Chem.MolFromSmiles(smi)
            if mol_raw is None:
                continue
            cano = Chem.MolToSmiles(mol_raw, canonical=True, isomericSmiles=False)
        except Exception:
            continue
        if cano in seen:
            n_dup += 1
            continue
        seen.add(cano)
        res = _smi_to_ms2forge_graph(cano)
        if res is None:
            continue
        x, edge_index, edge_attr = res
        pyg_data_list.append(PygData(x=x, edge_index=edge_index, edge_attr=edge_attr))
        valid_smis.append(cano)
    print(f'  RDKit successful: {len(valid_smis)} / {len(smiles_list)} (canonical after deduplication, {n_dup}  entries canonical duplicate)')


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
    """build_zms_cache implementation."""
    from models.model import MSEncoder

    print(f'[zms cache] building: scan  {msg_root}  ->  {cache_path}')


    ms_encoder = MSEncoder(
        dim_sos=13, dim_formula=144, hidden_dim=512,
        num_transformer_layers=3, nhead=8, output_dim=512,
        dropout=0.0, input_dropout=0.0, max_len=129,
    ).to(device)
    try:
        sd = torch.load(align_ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(align_ckpt_path, map_location=device)
    sd = sd['model'] if 'model' in sd else sd
    ms_state = {k[len('ms_encoder.'):]: v for k, v in sd.items() if k.startswith('ms_encoder.')}
    miss, unex = ms_encoder.load_state_dict(ms_state, strict=False)
    print(f'  ms_encoder ckpt: loaded={len(ms_state)}, missing={len(miss)}, unexpected={len(unex)}')
    ms_encoder.eval()
    m = type('Wrapper', (), {'ms_encoder': ms_encoder})()


    labels_df = pd.read_csv(os.path.join(msg_root, 'labels.tsv'), sep='\t')
    labels_df = labels_df[labels_df['ionization'].isin(_MS2FORGE_PRECURSOR.keys())].copy()
    print(f'  MSG samplethis : {len(labels_df)} (only  [M+H]+/[M-H]-)')

    subform_dir = os.path.join(msg_root, 'subformulae', 'default_subformulae')
    cache = {}


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
                pre_oh[_MS2FORGE_PRECURSOR[row['ionization']]] = 1.0
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
    """ensure_cond_emb_cache implementation."""
    paths = _cache_paths(cache_dir)
    if stage == 'graph2mol':
        path = paths['zmol']
        if os.path.exists(path) and not force_rebuild:
            print(f'[zmol cache] already exists, loading {path}')
            try:
                return torch.load(path, weights_only=False)
            except TypeError:
                return torch.load(path)
        if smiles_pool is None:
            raise ValueError('graph2mol stage requires  smiles_pool  to build  zmol cache')
        return build_zmol_cache(align_ckpt_path, smiles_pool, path,
                                 device=device, batch_size=batch_size)
    elif stage == 'ms2mol':
        path = paths['zms']
        if os.path.exists(path) and not force_rebuild:
            print(f'[zms cache] already exists, loading {path}')
            try:
                return torch.load(path, weights_only=False)
            except TypeError:
                return torch.load(path)
        if msg_root is None:
            raise ValueError('ms2mol stage requires  msg_root  to build  zms cache')
        return build_zms_cache(align_ckpt_path, msg_root, path,
                                device=device, batch_size=batch_size)
    else:
        raise ValueError(f'stage must be one of  graph2mol/ms2mol, ; received  {stage}')
