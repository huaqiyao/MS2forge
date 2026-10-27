import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path
from rdkit import Chem, RDLogger, rdBase
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parent
cache={}
def identity(s):
    if s not in cache:
        m=Chem.MolFromSmiles(s)
        assert m is not None
        m=Chem.MolFromSmiles(Chem.MolToSmiles(m,isomericSmiles=False))
        x=Chem.MolToInchi(m)
        assert x.startswith('InChI=1S/')
        cache[s]=x
    return cache[s]
groups={}; rows={}; hashes={}
for ds,fn in [('msg','split.tsv'),('nplib1','canopus_hplus_100_0.tsv')]:
    p=ROOT/'inputs'/ds
    for f in ['labels.tsv',fn]:hashes[ds+'/'+f]=hashlib.sha256((p/f).read_bytes()).hexdigest()
    labels=list(csv.DictReader((p/'labels.tsv').open(),delimiter='\t'))
    splitlist=list(csv.DictReader((p/fn).open(),delimiter='\t'))
    split={r['name']:r['split'] for r in splitlist}
    assert len(split)==len(splitlist)
    assert len({r['spec'] for r in labels})==len(labels)
    assert set(split)<={r['spec'] for r in labels}
    for r in labels:
        s=split.get(r['spec'])
        if s not in ['train','val','test']:continue
        k=ds+'_'+s
        i=identity(r['smiles'])
        groups.setdefault(k,set()).add(i)
        rows.setdefault(k,[]).append(i)
result={'rdkit':rdBase.rdkitVersion,'identity':'Full standard InChI after a non-isomeric SMILES round-trip',
    'source_sha256':hashes,'groups':{k:{'unique_identities':len(v),'spectrum_rows':len(rows[k])} for k,v in groups.items()},
    'msg_train_overlap_nplib1':{s:{'unique_identities':len(groups['msg_train']&groups['nplib1_'+s]),
        'spectrum_rows':sum(i in groups['msg_train'] for i in rows['nplib1_'+s])} for s in ['val','test']}}
(ROOT/'nplib1_cross_reference_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
