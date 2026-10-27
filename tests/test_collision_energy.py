"""Collision-energy parsing and train/export consistency on small CPU fixtures."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from utils.collision_energy import collision_energy_bin, energy_bin
from encoder.config import load
from encoder.data import Inputs
from encoder.inputs import metadata, spectrum_batch
from encoder.prepare import prepare_finetune
from encoder.export import export_caches
from encoder.model import build
from encoder.state import export, seed_all
from encoder.pretrain_data import one_hot_encode_energy


class CollisionEnergyTests(unittest.TestCase):
    def test_boundaries_and_missing(self):
        cases = [(None, 0), ('', 0), ('nan', 0), (float('inf'), 0), (-1, 0),
                 ('unknown', 0), ('20-40', 0), (0, 0), (19.99, 0), (20, 1),
                 (39.99, 1), (40, 2), (199.9, 9), (200, 10), (1000, 10),
                 ('NCE=65', 3), ('45 eV', 2), ('75%', 3)]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(energy_bin(value), expected)
                token = one_hot_encode_energy(value)
                self.assertEqual(token.argmax().item(), expected)
                self.assertEqual(token.sum().item(), 1)

    def test_field_precedence_and_ms_fallback(self):
        self.assertEqual(collision_energy_bin({'Collision_Energy_NCE': '65', 'ce': 180}), 3)
        self.assertEqual(collision_energy_bin({'nce': 'nan', 'collision_energy': '45 eV'}), 2)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'spec_files'
            directory.mkdir()
            file = directory / 'a.ms'
            file.write_text('>collisionenergy 85\n>ms2peaks\n100 1\n', encoding='utf-8')
            self.assertEqual(collision_energy_bin({}, tmp, 'a'), 4)
            self.assertEqual(collision_energy_bin({'ce': 0}, tmp, 'a'), 0)
            self.assertEqual(collision_energy_bin({}, tmp, 'missing'), 0)
            file.write_text('#collision_energy_NCE: 125\n>collisionenergy 85\n', encoding='utf-8')
            self.assertEqual(collision_energy_bin({}, tmp, 'a'), 6)
            file.write_text('>collisionenergy 20\n>collisionenergy 40\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'multiple collision energies'):
                collision_energy_bin({}, tmp, 'a')
            with self.assertRaisesRegex(ValueError, 'Unsafe spectrum ID'):
                collision_energy_bin({}, tmp, '../outside')

    def test_preparation_and_embedding_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_all(42, 2)
            checkpoint = root / 'encoder.pth'
            export(checkpoint, build())
            for dataset in ('msg', 'nplib1'):
                with self.subTest(dataset=dataset):
                    data = root / dataset / 'data'
                    annotations = data / 'subformulae/default_subformulae'
                    annotations.mkdir(parents=True)
                    (data / 'spec_files').mkdir()
                    rows = []
                    values = [('', ''), ('', '45'), ('65', '180'), ('nan', 'bad'), ('', ''), ('210', '')]
                    for i, (nce, ce) in enumerate(values):
                        sid = f's{i}'
                        rows.append(dict(spec=sid, smiles='CCO', formula='C2H6O',
                                         ionization='[M+H]+', instrument='Orbitrap',
                                         collision_energy_NCE=nce, collision_energy=ce))
                        (annotations / (sid + '.json')).write_text(json.dumps({'output_tbl': {'formula': ['CH3O']}}))
                    with (data / 'labels.tsv').open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
                        writer.writeheader(); writer.writerows(rows)
                    (data / 'split.tsv').write_text('name\tsplit\n' + ''.join(
                        f's{i}\t' + ('train' if i < 2 else 'val' if i < 4 else 'test') + '\n' for i in range(6)))
                    (data / 'spec_files/s4.ms').write_text('>collisionenergy 85\n>ms2peaks\n100 1\n')
                    cfg = load('configs/encoder.yml', dataset=dataset)
                    prepared = root / dataset / 'prepared'
                    cfg['datasets'][dataset].update(data_root=str(data), prepared_dir=str(prepared))
                    prepare_finetune(cfg)
                    np.testing.assert_array_equal(np.load(prepared / 'energy_bins.npy'), [0, 2, 3, 0])
                    records = metadata(data)
                    self.assertEqual([r['energy_bin'] for r in records], [0, 2, 3, 0, 4, 10])
                    self.assertTrue(all('smiles' not in r for r in records))
                    live = spectrum_batch(records, data, 'cpu')
                    trained = Inputs(prepared, 'cpu').batch([0, 1, 2, 3])
                    for x, y in zip(trained[:3], live):
                        torch.testing.assert_close(x, y[:4], rtol=0, atol=0)
                    self.assertEqual(live[0][:, 0, 2:].argmax(1).tolist(), [0, 2, 3, 0, 4, 10])
                    out = root / dataset / 'embeddings'
                    export_caches(cfg, checkpoint, out, prepared / 'manifest.tsv', batch_size=2)
                    cached = np.load(out / 'zms/features.npy')
                    manifest = json.loads((out / 'zms/CACHE_MANIFEST.json').read_text())
                    self.assertEqual(manifest['source']['energy_bins'], {f's{i}': b for i, b in enumerate([0, 2, 3, 0, 4, 10])})
                    np.testing.assert_array_equal(cached[0], cached[3])
                    self.assertGreater(np.max(np.abs(cached[0] - cached[1])), 1e-5)
                    (prepared / 'energy_bins.npy').unlink()
                    with self.assertRaisesRegex(FileNotFoundError, 'rerun encoder.prepare'):
                        Inputs(prepared, 'cpu')


if __name__ == '__main__':
    unittest.main()
