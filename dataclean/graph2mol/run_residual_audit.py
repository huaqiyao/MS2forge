"""Read-only full pass over cleaned SAFE-GPT shards for NPLIB1 held-out overlap."""
import concurrent.futures, csv, hashlib, io, json, os, sys, time
from collections import Counter
from pathlib import Path
import zstandard as zstd
from rdkit import Chem,RDLogger
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from matching import Matcher
RDLogger.DisableLog('rdApp.*')
OUT=ROOT/'nplib1_residual_audit'
MATCHER=None
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def init(ref):
    global MATCHER
    MATCHER=Matcher(ref)
def identity(s):
    m=Chem.MolFromSmiles(s);assert m is not None
    m=Chem.MolFromSmiles(Chem.MolToSmiles(m,isomericSmiles=False))
    return Chem.MolToInchi(m)
def work(p):
    r=read(p);dst=OUT/(r['split']+'__'+p.name)
    if dst.exists():
        old=read(dst)
        if old['expected_sha256']==r['output_sha256'] and old['actual_sha256']==sha(ROOT/r['output_path']):return old
    h=hashlib.sha256();n=0;hits=Counter()
    with (ROOT/r['output_path']).open('rb') as raw,zstd.ZstdDecompressor().stream_reader(raw) as reader,io.BufferedReader(reader,1024**2) as lines:
        for line in lines:
            h.update(line);n+=1
            match,_=MATCHER.match(line[:-1].decode())
            if match:
                for g in match['groups']:
                    if g.startswith('nplib1_'):hits[g]+=1
    assert n==r['retained_rows'] and h.hexdigest()==r['output_text_sha256']
    actual=sha(ROOT/r['output_path']);assert actual==r['output_sha256']
    d={'split':r['split'],'rows':n,'nplib1_residual_records':dict(hits),
       'expected_sha256':r['output_sha256'],'actual_sha256':actual,'passed':not hits}
    save(dst,d);return d
def main():
    OUT.mkdir(exist_ok=True)
    labels=list(csv.DictReader((ROOT/'inputs/nplib1/labels.tsv').open(),delimiter='\t'))
    splits={r['name']:r['split'] for r in csv.DictReader((ROOT/'inputs/nplib1/canopus_hplus_100_0.tsv').open(),delimiter='\t')}
    groups={s:set() for s in ('val','test')}
    for row in labels:
        if splits.get(row['spec']) in groups:groups[splits[row['spec']]].add(identity(row['smiles']))
    assert {k:len(v) for k,v in groups.items()}=={'val':399,'test':714}
    ref=[r for r in read(ROOT/'reference.json') if any(g.startswith('nplib1_') for g in r['groups'])]
    assert {r['inchi'] for r in ref}==groups['val']|groups['test']
    paths=sorted((ROOT/'shard_reports').glob('*/*.json'));assert len(paths)==1168
    results=[];start=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=24,initializer=init,initargs=(ref,)) as pool:
        fs=[pool.submit(work,p) for p in paths]
        for f in concurrent.futures.as_completed(fs):
            results.append(f.result())
            if len(results)%20==0:
                status={'complete':False,'shards':len(results),'rows':sum(x['rows'] for x in results),'elapsed_seconds':time.time()-start}
                save(OUT/'status.json',status);print(json.dumps(status),flush=True)
    totals={s:sum(x['rows'] for x in results if x['split']==s) for s in ('train','validation','test')}
    assert sum(totals.values())==1166664975
    d={'complete':True,'passed':all(x['passed'] for x in results),'rows':sum(totals.values()),'shards':len(results),
       'split_rows':totals,'reference_unique':{s:len(g) for s,g in groups.items()},
       'residual_records':{s:{g:sum(x['nplib1_residual_records'].get(g,0) for x in results if x['split']==s) for g in ('nplib1_val','nplib1_test')} for s in totals},
       'method':'Full output stream pass with validated composition/carbon-graph screening followed by exact full InChI comparison; every output compressed and decoded SHA256 checked.',
       'elapsed_seconds':time.time()-start}
    save(OUT/'status.json',d);save(OUT/'residual_audit.json',d);print(json.dumps(d),flush=True)
if __name__=='__main__':main()
