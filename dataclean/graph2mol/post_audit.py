"""Independent exact-identity audit of every excluded record and retained samples."""
import concurrent.futures
from collections import Counter
import csv
import io
import json
from pathlib import Path
from rdkit import Chem, RDLogger
from clean_safe_gpt import read, atomic, digest, stream_lines

ROOT=Path(__file__).resolve().parent
RDLogger.DisableLog('rdApp.*')
REF=None
LOOKUP=None
def initialize():
    global REF,LOOKUP
    REF=read(ROOT/'reference.json')
    LOOKUP={r['inchi'] for r in REF}
def independent_identity(smiles):
    mol=Chem.MolFromSmiles(smiles)
    assert mol is not None
    achiral=Chem.MolFromSmiles(Chem.MolToSmiles(mol,isomericSmiles=False))
    result=Chem.MolToInchi(achiral)
    assert result
    return result
def check(path):
    report=read(path)
    assert report['status']=='verified'
    assert report['input_rows']==report['removed_rows']+report['retained_rows']
    assert sum(report['reference_hits'].values())==report['removed_rows']
    removed=ROOT/report['removed_path'];output=ROOT/report['output_path']
    assert digest(removed)==report['removed_sha256']
    assert output.stat().st_size==report['output_bytes']
    hits=Counter();last=-1
    for row in csv.DictReader((b.decode() for b in stream_lines(removed)),delimiter='\t'):
        idx=int(row['reference_id']);ref=REF[idx]
        key=independent_identity(row['source_smiles'])
        assert key==row['inchi']==ref['inchi']
        assert row['evaluation_membership'].split(',')==ref['groups']
        pos=int(row['source_row_zero_based'])
        assert report['source_row_start']<=pos<report['source_row_stop'] and pos>last
        assert row['source_parquet']==report['source_parquet']
        last=pos;hits[str(idx)]+=1
    assert dict(hits)==report['reference_hits']
    samples=0
    lines=stream_lines(output)
    try:
        for line in lines:
            assert independent_identity(line.decode().rstrip('\n')) not in LOOKUP
            samples+=1
            if samples==10:break
    finally:lines.close()
    return {'split':report['split'],'removed':sum(hits.values()),'samples':samples,
            'input':report['input_rows'],'retained':report['retained_rows'],'hits':dict(hits)}
def main():
    audit=read(ROOT/'audit_results.json');assert audit['complete']
    paths=sorted((ROOT/'shard_reports').glob('*/*.json'))
    assert len(paths)==audit['verified_shards']==1168
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,initializer=initialize) as pool:
        checked=list(pool.map(check,paths))
    total=Counter();hits=Counter()
    for d in checked:
        total.update({k:d[k] for k in ('input','removed','retained','samples')});hits.update(d['hits'])
    assert total['input']==audit['scanned_rows'] and total['removed']==audit['removed_rows'] and total['retained']==audit['retained_rows']
    assert dict(hits)==audit['global_reference_hits']
    for split,expected in audit['splits'].items():
        part=[d for d in checked if d['split']==split]
        assert sum(d['input'] for d in part)==expected['input_rows']
        assert sum(d['retained'] for d in part)==expected['retained_rows']
        assert sum(d['removed'] for d in part)==expected['removed_rows']
    manifest=read(ROOT.parent/'manifest.json')
    for item in manifest['files']:
        assert (ROOT.parent/'raw'/item['path']).stat().st_size==item['size']
    report={'passed':True,'shards_checked':len(paths),
            'removed_rows_rechecked':total['removed'],'retained_rows_sampled':total['samples'],
            'retained_sample_exclusion_matches':0,'aggregate_counts_match':True,
            'raw_parquet_files_present_and_size_checked':len(manifest['files']),
            'source_integrity_note':'Original download SHA256/Parquet verification is in verification_report.json; current post audit checks raw file presence and size.',
            'audit_results_sha256':digest(ROOT/'audit_results.json')}
    atomic(ROOT/'post_audit.json',report)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
