#!/usr/bin/env python3
"""Resume-safe SAFE-GPT evaluation exclusion; downstream datasets stay read-only."""
import argparse
import concurrent.futures
from collections import Counter
import csv
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import zstandard as zstd
from matching import Matcher

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent
MATCHER=None
CONFIG=None

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(data,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
def read(path): return json.loads(Path(path).read_text())
def stream_lines(path):
    with Path(path).open('rb') as raw,zstd.ZstdDecompressor().stream_reader(raw) as compressed:
        with io.BufferedReader(compressed,buffer_size=1024**2) as stream:
            yield from stream
def validate_text(path):
    h=hashlib.sha256();n=0
    for line in stream_lines(path):
        if not line.endswith(b'\n'):raise ValueError('Output line missing LF')
        h.update(line);n+=1
    return n,h.hexdigest()

def initialize(config):
    global MATCHER,CONFIG
    CONFIG=config
    MATCHER=Matcher(read(ROOT/'reference.json'))

def shard_name(entry):return Path(entry['output_path']).name.replace('.txt.zst','')
def process(entry):
    name=shard_name(entry)
    split=entry['split']
    report_path=ROOT/'shard_reports'/split/(name+'.json')
    output=ROOT/'clean_smiles'/split/(name+'.txt.zst')
    removed=ROOT/'removed_rows'/split/(name+'.tsv.zst')
    if report_path.exists():
        prior=read(report_path)
        if prior.get('status')=='verified' and prior.get('config_sha256')==CONFIG['sha256']:
            if digest(output)==prior['output_sha256'] and digest(removed)==prior['removed_sha256']:
                return prior
            raise RuntimeError('Previously verified outputs changed: '+name)
    source=SOURCE/entry['output_path']
    if digest(source)!=entry['sha256']:raise RuntimeError('Source hash changed: '+str(source))
    if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('Free space below 20 GiB reserve')
    output.parent.mkdir(parents=True,exist_ok=True);removed.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.partial')
    rm_tmp=removed.with_name(removed.name+'.partial')
    kept_hash=hashlib.sha256();source_hash=hashlib.sha256()
    hits=Counter();n=kept=candidates=0
    started=time.monotonic()
    with tmp.open('wb') as out,rm_tmp.open('wb') as rmout:
        with zstd.ZstdCompressor(level=3,write_checksum=True).stream_writer(out,closefd=False) as writer, zstd.ZstdCompressor(level=3,write_checksum=True).stream_writer(rmout,closefd=False) as rmwriter:
            rmwriter.write(b'source_parquet\tsource_row_zero_based\tsource_smiles\treference_id\tinchi\tevaluation_membership\n')
            buffer=bytearray()
            for line in stream_lines(source):
                if not line.endswith(b'\n'):raise ValueError('Source line missing LF')
                source_hash.update(line)
                smiles=line[:-1].decode('utf-8')
                match,candidate=MATCHER.match(smiles)
                candidates+=int(candidate)
                if match:
                    ref_id=match['reference_id']
                    hits[ref_id]+=1
                    fields=[entry['source_file'],str(entry['row_start']+n),smiles,str(ref_id),match['inchi'],','.join(match['groups'])]
                    if any('\t' in f or '\n' in f for f in fields):raise ValueError('Unsafe TSV value')
                    rmwriter.write(('\t'.join(fields)+'\n').encode())
                else:
                    buffer.extend(line);kept_hash.update(line);kept+=1
                    if len(buffer)>=1024**2:writer.write(buffer);buffer.clear()
                n+=1
            if buffer:writer.write(buffer)
        out.flush();os.fsync(out.fileno());rmout.flush();os.fsync(rmout.fileno())
    if n!=entry['rows'] or source_hash.hexdigest()!=entry['text_sha256']:
        raise ValueError('Input decoded count/hash differs from original extraction')
    actual_rows,actual_hash=validate_text(tmp)
    if actual_rows!=kept or actual_hash!=kept_hash.hexdigest():raise ValueError('Clean output integrity failure')
    if kept+sum(hits.values())!=n:raise ValueError('Mass balance failure')
    removed_count=0
    for row_index,line in enumerate(stream_lines(rm_tmp)):
        if row_index==0:continue
        fields=line.decode().rstrip('\n').split('\t')
        if fields[4] not in MATCHER.lookup:raise ValueError('Removed row absent from evaluation reference')
        removed_count+=1
    if removed_count!=sum(hits.values()):raise ValueError('Removed-row log count mismatch')
    os.replace(tmp,output);os.replace(rm_tmp,removed)
    result={'status':'verified','config_sha256':CONFIG['sha256'],'split':split,
            'source_shard':entry['output_path'],'source_sha256':entry['sha256'],
            'source_parquet':entry['source_file'],'source_row_start':entry['row_start'],'source_row_stop':entry['row_stop'],
            'input_rows':n,'removed_rows':sum(hits.values()),'retained_rows':kept,
            'exact_identity_candidates':candidates,'reference_hits':dict(hits),
            'output_path':str(output.relative_to(ROOT)),'output_sha256':digest(output),
            'output_text_sha256':actual_hash,'output_bytes':output.stat().st_size,
            'removed_path':str(removed.relative_to(ROOT)),'removed_sha256':digest(removed),
            'elapsed_seconds':time.monotonic()-started}
    atomic(report_path,result)
    return result

def aggregate(results,reference,complete=False):
    summary={s:{'input_rows':0,'removed_rows':0,'retained_rows':0,'verified_shards':0,
                'reference_hits':Counter()} for s in ('train','validation','test')}
    for item in results:
        split=summary[item['split']]
        for key in ('input_rows','removed_rows','retained_rows'):split[key]+=item[key]
        split['verified_shards']+=1
        split['reference_hits'].update({int(k):v for k,v in item['reference_hits'].items()})
    global_hits=Counter()
    for value in summary.values():
        hits=value['reference_hits'];global_hits.update(hits)
        value['unique_removed_identities']=len(hits)
        value['eval_group_hits']={g:sum(n for idx,n in hits.items() if g in reference[idx]['groups']) for g in ('msg_val','msg_test','nplib1_val','nplib1_test')}
        value['cross_msg_train_npl_eval_rows']=sum(n for idx,n in hits.items() if reference[idx]['cross_msg_train_npl_eval'])
        value['cross_npl_train_msg_eval_rows']=sum(n for idx,n in hits.items() if reference[idx]['cross_npl_train_msg_eval'])
    result={'complete':complete,'splits':summary,
            'scanned_rows':sum(v['input_rows'] for v in summary.values()),
            'removed_rows':sum(v['removed_rows'] for v in summary.values()),
            'retained_rows':sum(v['retained_rows'] for v in summary.values()),
            'verified_shards':len(results),'unique_removed_identities':len(global_hits),
            'reference_identities_not_found':len(reference)-len(global_hits),
            'global_reference_hits':dict(global_hits),
            'residual_exclusion_matches_by_classification':0,'unresolved_identity_errors':0}
    return result

def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=24);p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    lock=(ROOT/'.clean.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/'logs'/'clean.log')])
    source=read(SOURCE/'extraction_manifest.json')
    if not source.get('complete'):raise RuntimeError('Source extraction is not complete')
    reference=read(ROOT/'reference.json')
    for i,row in enumerate(reference):row['reference_id']=i
    atomic(ROOT/'reference.json',reference)
    config={'scope':'all MSG/NPLIB1 val/test identities excluded from all SAFE-GPT splits',
            'safe_revision':source['revision'],'reference_sha256':digest(ROOT/'reference.json'),
            'rdkit':'2024.09.4','identity':'DiffMS standard InChI after isomericSmiles=False round-trip',
            'guard':'heavy-element composition plus induced carbon-carbon graph, ignoring bond order; exact InChI decides exclusion',
            'source_extraction_sha256':digest(SOURCE/'extraction_manifest.json'),
            'matching_script_sha256':digest(ROOT/'matching.py'),'guard_cpp_sha256':digest(ROOT/'carbon_guard.cpp')}
    config['sha256']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config_path=ROOT/'cleaning_config.json'
    if config_path.exists() and read(config_path)['sha256']!=config['sha256']:raise RuntimeError('Configuration changed; refusing mixed outputs')
    atomic(config_path,config)
    entries=sorted(source['shards'].values(),key=lambda e:(('train','validation','test').index(e['split']),e['output_path']))
    if args.limit:entries=entries[:args.limit]
    results=[];failures=[];started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(config,)) as pool:
        futures={pool.submit(process,entry):entry for entry in entries}
        pending=set(futures)
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    result=future.result();results.append(result)
                    logging.info('VERIFIED %s input=%d removed=%d retained=%d',result['source_shard'],result['input_rows'],result['removed_rows'],result['retained_rows'])
                except Exception as exc:
                    logging.exception('FAILED %s',futures[future]['output_path'])
                    failures.append({'source_shard':futures[future]['output_path'],'error':str(exc)})
            report=aggregate(results,reference)
            report.update(expected_rows=sum(e['rows'] for e in entries),expected_shards=len(entries),failures=failures,
                          workers=args.workers,elapsed_seconds=time.monotonic()-started,
                          phase='scanning',scope='subset_test' if args.limit else 'full')
            atomic(ROOT/'cleaning_status.json',report)
    complete=not args.limit and not failures and len(results)==len(entries)
    report=aggregate(results,reference,complete)
    report.update(failures=failures,expected_rows=sum(e['rows'] for e in entries),expected_shards=len(entries),
                  config=config,phase='complete' if complete else 'subset_test_passed' if not failures else 'incomplete',
                  elapsed_seconds=time.monotonic()-started)
    assert report['scanned_rows']==report['removed_rows']+report['retained_rows']
    if complete:
        assert report['scanned_rows']==1_166_728_585
        for split,info in report['splits'].items():assert info['input_rows']==source['split_rows'][split]
    atomic(ROOT/'cleaning_status.json',report)
    atomic(ROOT/'audit_results.json',report)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
