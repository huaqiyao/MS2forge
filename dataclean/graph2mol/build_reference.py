import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import time
from rdkit import Chem, RDLogger, rdBase

ROOT=Path(__file__).resolve().parent
RDLogger.DisableLog('rdApp.*')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('SMILES parse failure: '+smiles)
    # Exact identity path used in DiffMS build_fp2mol_datasets.py.
    achiral=Chem.MolToSmiles(mol,isomericSmiles=False)
    mol=Chem.MolFromSmiles(achiral)
    inchi=Chem.MolToInchi(mol)
    if not inchi:
        raise ValueError('Empty InChI: '+smiles)
    composition=tuple(sorted(Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum()!=1).items()))
    return inchi,achiral,composition

def main():
    started=time.time()
    cache={}
    groups={}
    sources={}
    label_rows={}
    joined_rows={}
    for dataset,splitname in [('msg','split.tsv'),('nplib1','canopus_hplus_100_0.tsv')]:
        labelpath=ROOT/'inputs'/dataset/'labels.tsv'
        splitpath=ROOT/'inputs'/dataset/splitname
        labels=list(csv.DictReader(labelpath.open(),delimiter='\t'))
        splits=list(csv.DictReader(splitpath.open(),delimiter='\t'))
        splitmap={r['name']:r['split'] for r in splits}
        assert len(splitmap)==len(splits),'Duplicate split names'
        ids=[r['spec'] for r in labels]
        assert len(set(ids))==len(ids),'Duplicate spectrum labels'
        assert set(splitmap)<=set(ids),'Split names without labels'
        sources[dataset]={'labels_sha256':digest(labelpath),'split_sha256':digest(splitpath),
                          'labels_rows':len(labels),'split_rows':len(splits),
                          'split_value_counts':dict(Counter(splitmap.values())),
                          'labels_not_in_split':sum(s not in splitmap for s in ids)}
        label_rows[dataset]=labels
        joined_rows[dataset]=[]
        for row in labels:
            split=splitmap.get(row['spec'])
            if split not in ('train','val','test'):
                continue
            smiles=row['smiles']
            if smiles not in cache:
                cache[smiles]=identity(smiles)
            inchi,achiral,comp=cache[smiles]
            key=dataset+'_'+split
            groups.setdefault(key,{}).setdefault(inchi,[]).append(row['spec'])
            joined_rows[dataset].append({**row,'split':split,'audit_inchi':inchi})
    eval_names=['msg_val','msg_test','nplib1_val','nplib1_test']
    excluded=set().union(*(set(groups[k]) for k in eval_names))
    cross2=set(groups['msg_train']) & (set(groups['nplib1_val'])|set(groups['nplib1_test']))
    cross3=set(groups['nplib1_train']) & (set(groups['msg_val'])|set(groups['msg_test']))
    representative={}
    compositions={}
    for smi,(inchi,canonical,comp) in cache.items():
        if inchi in excluded:
            representative.setdefault(inchi,{'smiles':smi,'canonical_no_stereo':canonical,'heavy_composition':comp})
            compositions.setdefault(str(comp),set()).add(inchi)
    reference=[]
    for inchi in sorted(excluded):
        membership=[k for k in eval_names if inchi in groups[k]]
        reference.append({'inchi':inchi,'inchikey':Chem.InchiToInchiKey(inchi),**representative[inchi],
                          'groups':membership,'cross_msg_train_npl_eval':inchi in cross2,
                          'cross_npl_train_msg_eval':inchi in cross3})
    report={'rdkit_version':rdBase.rdkitVersion,'sources':sources,
            'identity':'MolFromSmiles -> MolToSmiles(isomericSmiles=False) -> MolFromSmiles -> MolToInchi (standard)',
            'scope':'Remove all four downstream evaluation sets from each SAFE-GPT split; downstream source files unchanged',
            'groups':{k:{'spectrum_rows':sum(map(len,v.values())),'unique_identity_molecules':len(v)} for k,v in groups.items()},
            'unique_exclusion_molecules':len(excluded),'heavy_compositions':len(compositions),
            'cross_msg_train_npl_eval_unique':len(cross2),'cross_npl_train_msg_eval_unique':len(cross3),
            'cross_intersection_unique':len(cross2&cross3),'cross_union_unique':len(cross2|cross3),
            'pairwise_unique_overlaps':{a+'__'+b:len(set(groups[a])&set(groups[b])) for a in groups for b in groups if a<b},
            'invalid_reference_smiles':0,'elapsed_seconds':time.time()-started}
    (ROOT/'reference.json').write_text(json.dumps(reference,indent=2))
    (ROOT/'reference_audit.json').write_text(json.dumps(report,indent=2))
    with (ROOT/'reference_molecules.tsv').open('w',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t')
        writer.writerow(['inchi','inchikey','canonical_no_stereo','eval_membership','cross_msg_train_npl_eval','cross_npl_train_msg_eval'])
        for r in reference:
            writer.writerow([r['inchi'],r['inchikey'],r['canonical_no_stereo'],','.join(r['groups']),int(r['cross_msg_train_npl_eval']),int(r['cross_npl_train_msg_eval'])])
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
