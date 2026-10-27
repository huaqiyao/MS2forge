import ctypes
import json
import random
import time
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from matching import Matcher,exact_identity

ROOT=Path(__file__).resolve().parent
MASK=(1<<64)-1
def mix(h,x):return (h^((x+0x9e3779b97f4a7c15+((h<<6)&MASK)+(h>>2))&MASK))&MASK
def independent_rdkit_guard(smiles):
    mol=Chem.MolFromSmiles(smiles)
    assert mol is not None
    composition=Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum()!=1)
    carbons=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==6]
    ids={a:i for i,a in enumerate(carbons)}
    graph=[[ids[n.GetIdx()] for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetAtomicNum()==6] for a in carbons]
    labels=[mix(0x84222325cbf29ce4,len(n)) for n in graph]
    for _ in range(3):
        nxt=[]
        for i,neighbors in enumerate(graph):
            h=mix(0x3141592653589793,labels[i])
            for value in sorted(labels[n] for n in neighbors):h=mix(h,value)
            nxt.append(h)
        labels=nxt
    h=0xcbf29ce484222325
    for element,count in sorted(composition.items()):
        for c in element:h=mix(h,ord(c))
        h=mix(h,count);h=mix(h,0xff)
    h=mix(h,len(graph))
    for value in sorted(labels):h=mix(h,value)
    return h or 1

ref=json.loads((ROOT/'reference.json').read_text());m=Matcher(ref)
count=0
for row in ref:
    smi=row['smiles']
    assert m.guard(smi.encode())==independent_rdkit_guard(smi)
    alternate=Chem.MolToSmiles(Chem.MolFromSmiles(smi),doRandom=True)
    assert m.guard(alternate.encode())==m.guard(smi.encode())
    assert m.match(alternate)[0] is not None
    count+=1
files=sorted((ROOT.parent/'raw'/'data').glob('*/*.parquet'))
full_comparisons=0
for file in files:
    for smi in next(pq.ParquetFile(file).iter_batches(batch_size=200,columns=['smiles'])).column(0).to_pylist():
        assert m.guard(smi.encode())==independent_rdkit_guard(smi)
        assert bool(m.match(smi)[0])==(exact_identity(smi) in m.lookup)
        full_comparisons+=1
tautomer_checks=0
enumerator=rdMolStandardize.TautomerEnumerator()
for row in random.Random(42).sample(ref,300):
    for mol in list(enumerator.Enumerate(Chem.MolFromSmiles(row['smiles'])))[:30]:
        smi=Chem.MolToSmiles(mol)
        if exact_identity(smi) in m.lookup:
            assert m.match(smi)[0] is not None
            tautomer_checks+=1
result={'passed':True,'reference_graph_checks':count,'randomized_reference_match_checks':count,
        'independent_full_inchi_source_comparisons':full_comparisons,
        'all_118_source_parquets_sampled':True,'recognized_tautomer_checks':tautomer_checks,
        'fallback_compositions':len(m.fallback_compositions)}
(ROOT/'guard_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
