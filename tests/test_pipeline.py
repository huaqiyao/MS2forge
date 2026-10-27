"""CPU integration: raw annotations -> encoder -> caches -> decoders -> sampling."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml
from encoder.config import load
from encoder.prepare import prepare_finetune
from encoder.inputs import metadata, spectrum_batch, graph_batch
from encoder.data import Inputs
from encoder.export import export_caches
from encoder.model import build
from encoder.state import export, seed_all
from encoder.train import run_finetune
from utils.artifacts import resolve, sha256, require_encoder
from utils.checkpoint import read_weights
from utils.prepare import pipeline

REPO=Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_raw_inputs_to_sampling_and_identity_checks(self):
        with tempfile.TemporaryDirectory(prefix='pipeline_test_') as tmp, patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':''}):
            root=Path(tmp)
            with patch.dict(os.environ, {'MS2FORGE_ASSETS':str(root)}):
                data=root/'msg/data'; annotations=data/'subformulae/default_subformulae'
                annotations.mkdir(parents=True)
                rows=[]
                for i in range(7):
                    sid=f's{i}'; split='train' if i<4 else ('val' if i<6 else 'test')
                    rows.append(dict(spec=sid,split=split,smiles='CCO',formula='C2H6O',ionization='[M+H]+',instrument='Orbitrap',
                                     collision_energy='' if i == 0 else str(i * 25)))
                    (annotations/(sid+'.json')).write_text(json.dumps({'output_tbl':{'formula':['CH3O','C2H5']}}))
                with (data/'labels.tsv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,delimiter='\t',fieldnames=[k for k in rows[0] if k!='split'])
                    writer.writeheader();writer.writerows({k:v for k,v in r.items() if k!='split'} for r in rows)
                (data/'split.tsv').write_text('name\tsplit\n'+''.join(r['spec']+'\t'+r['split']+'\n' for r in rows))
                (data/'graph2mol.tsv').write_text('sample_id\tsplit\tsmiles\nG1\ttrain\tCCO\nG2\ttrain\tCCO\nG3\tval\tCCO\n')
                cfg=load('configs/encoder.yml')
                cfg['output_root']=str(root/'encoder_runs')
                cfg['runtime'].update(device='cpu',threads=2)
                initial=root/'initial.pth';seed_all(42,2);export(initial,build())
                cfg['finetune'].update(initial_weights=str(initial),batch_size=2,warmup_steps=1,monitor_every=1,monitor_ratio=1.0)
                cfg['datasets']['msg']['steps']=2
                config=root/'encoder.yml';config.write_text(yaml.safe_dump(cfg))
                prepared=prepare_finetune(cfg)
                self.assertEqual(len(np.load(prepared/'train_indices.npy')),4)
                self.assertEqual(len(np.load(prepared/'val_indices.npy')),2)
                self.assertFalse((prepared/'test_indices.npy').exists())
                inputs=Inputs(prepared,'cpu').batch([0,1])
                for expected, actual in zip(inputs[:3],spectrum_batch(metadata(data)[:2],data,'cpu')):
                    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
                for expected,actual in zip(inputs[3:],graph_batch(['CCO','CCO'],'cpu')):
                    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
                # Spectrum preparation does not use the test reference structure.
                labels=(data/'labels.tsv').read_text();(data/'labels.tsv').write_text(labels.replace('s6\tCCO','s6\tINVALID_SMILES'))
                self.assertNotIn('smiles',metadata(data)[-1])
                run_finetune(cfg,torch.device('cpu'),[])
                selected=root/'msg/checkpoints/encoder/latest.json'
                frozen=resolve(selected,'encoder')
                export_caches(cfg,selected,root/'msg/embeddings',data/'graph2mol.tsv',2,'cpu')
                manifest=json.loads((root/'msg/embeddings/zms/CACHE_MANIFEST.json').read_text())
                self.assertEqual(manifest['encoder_sha256'],sha256(frozen))
                self.assertEqual(manifest['count'],7)
                with self.assertRaises(FileExistsError):
                    export_caches(cfg,selected,root/'msg/embeddings',data/'graph2mol.tsv',2,'cpu')
                pipeline(SimpleNamespace(encoder_config=str(config),dataset='msg',graph_rows=None))
                for stage in ('graph2mol','ms2mol'):
                    decoder=yaml.safe_load((REPO/'configs'/f'{stage}.yml').read_text())
                    decoder['train'].update(global_batch=2,updates=1,validation_every=1,checkpoint_every=1,max_micro_graphs=1)
                    filename=root/(stage+'.yml');filename.write_text(yaml.safe_dump(decoder))
                    self.command('-m','utils.train','--config',filename,'--device','cpu')
                    self.assertTrue((root/f'msg/output/{stage}/latest.json').exists())
                graph_selected=resolve(root/'shared/checkpoints/graph2mol.json','graph2mol')
                self.assertIn('eq_graph2mol_u1_rank0.pt',str(graph_selected))
                weights=resolve(root/'msg/output/ms2mol/latest.json','ms2mol')
                raw=torch.load(weights,map_location='cpu',weights_only=True)
                self.assertTrue(all(torch.is_tensor(v) for v in raw.values()))
                payload,_=read_weights(weights)
                require_encoder(payload,manifest['encoder_sha256'],'test')
                with self.assertRaises(ValueError):require_encoder(payload,'wrong-encoder','test')
                with self.assertRaises(ValueError):require_encoder({},manifest['encoder_sha256'],'legacy')
                del payload
                sample=yaml.safe_load((REPO/'configs/sample.yml').read_text())
                sample.update(draws=2,steps=2,batch_size=1)
                filename=root/'sample.yml';filename.write_text(yaml.safe_dump(sample))
                self.command('sample.py','--config',filename,'--gpus','1','--device','cpu')
                # Resume an existing run whose manifest/cache use historical names.
                sample_dir=root/'msg/output/rerank_candidates/test'
                run_manifest=sample_dir/'run_rank0.json'
                saved_identity=json.loads(run_manifest.read_text())
                saved_model=saved_identity['config']['model']
                saved_model['flash']=saved_model.pop('ms2forge')
                run_manifest.write_text(json.dumps(saved_identity))
                cache_file=sample_dir/'candidates_rank0.jsonl'
                cache_file.write_text(cache_file.read_text().replace('ms2forge_candidate_cache.v1','flash_candidate_cache.v1'))
                old_manifest,old_cache=run_manifest.read_bytes(),cache_file.read_bytes()
                self.command('sample.py','--config',filename,'--gpus','1','--device','cpu','--resume')
                self.assertEqual(run_manifest.read_bytes(),old_manifest)
                self.assertEqual(cache_file.read_bytes(),old_cache)
                records=(root/'msg/output/rerank_candidates/test/candidates_rank0.jsonl').read_text().splitlines()
                self.assertEqual(len(records),1)
                self.assertEqual(json.loads(records[0])['spec_id'],'s6')
                # NPLIB1 can reuse the shared MSG Graph2Mol with its own fine-tuned encoder.
                shutil.copytree(data,root/'nplib1/data')
                npl=copy.deepcopy(cfg);npl['dataset']='nplib1'
                npl['finetune']['seed']=2027
                npl['datasets']['nplib1']['steps']=2
                prepare_finetune(npl)
                run_finetune(npl,torch.device('cpu'),[])
                npl_selected=root/'nplib1/checkpoints/encoder/latest.json'
                npl_encoder_hash=sha256(resolve(npl_selected,'encoder'))
                self.assertNotEqual(npl_encoder_hash,manifest['encoder_sha256'])
                export_caches(npl,npl_selected,root/'nplib1/embeddings',root/'nplib1/data/graph2mol.tsv',2,'cpu')
                pipeline(SimpleNamespace(encoder_config=str(config),dataset='nplib1',graph_rows=None))
                self.command('-m','utils.train','--config',root/'ms2mol.yml','--dataset','nplib1','--device','cpu')
                npl_payload,_=read_weights(root/'nplib1/output/ms2mol/latest.json')
                require_encoder(npl_payload,npl_encoder_hash,'NPLIB1 student')
                with self.assertRaises(ValueError):
                    require_encoder(npl_payload,manifest['encoder_sha256'],'NPLIB1 student')
                self.assertEqual(resolve(root/'shared/checkpoints/graph2mol.json','graph2mol'),graph_selected)
                self.command('sample.py','nplib1','--config',filename,'--gpus','1','--device','cpu')
                self.assertTrue((root/'nplib1/output/rerank_candidates/test/candidates_rank0.jsonl').exists())
                # Changing cache identity must stop sampling, even with valid checkpoint tensors.
                cached=root/'msg/embeddings/zms/CACHE_MANIFEST.json'
                bad=dict(manifest,encoder_sha256='different');cached.write_text(json.dumps(bad))
                result=self.command('sample.py','--config',filename,'--gpus','1','--device','cpu','--resume',expect_success=False)
                self.assertIn('different encoder',result.stderr)
                # Pointer hashes also reject physically replaced weights.
                pointer=root/'bad.json';record=json.loads(selected.read_text());record['sha256']='bad';pointer.write_text(json.dumps(record))
                with self.assertRaises(ValueError):resolve(pointer,'encoder')

    def command(self,*args,expect_success=True):
        result=subprocess.run([sys.executable,*map(str,args)],cwd=REPO,text=True,capture_output=True,timeout=180)
        if expect_success:self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        else:self.assertNotEqual(result.returncode,0)
        return result


if __name__=='__main__':unittest.main()
