"""Exact DiffMS identity with a necessary heavy-element-composition prefilter."""
import re
from collections import Counter
from rdkit import Chem, RDLogger
import ctypes
from pathlib import Path

RDLogger.DisableLog('rdApp.*')
TOKEN=re.compile(r'\[[^\]]+\]|Cl|Br|[BCNOPSFIbcnops*]')
BRACKET=re.compile(r'\[(?:[0-9]+)?([A-Z][a-z]?|se|as|[bcnops]|\*)')
ALLOWED_GAP=re.compile(r'^[0-9%().=#$:/\\@+\-]*$')

def fast_composition(smiles):
    counts={}
    previous=0
    for match in TOKEN.finditer(smiles):
        if not ALLOWED_GAP.fullmatch(smiles[previous:match.start()]):
            return None
        token=match.group()
        if token.startswith('['):
            atom=BRACKET.match(token)
            if not atom:
                return None
            element=atom.group(1).capitalize()
        else:
            element=token.capitalize()
        if element=='*':
            return None
        if element!='H':
            counts[element]=counts.get(element,0)+1
        previous=match.end()
    if not ALLOWED_GAP.fullmatch(smiles[previous:]) or not counts:
        return None
    return tuple(sorted(counts.items()))

def exact_identity(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('SMILES parsing failed')
    smi=Chem.MolToSmiles(mol,isomericSmiles=False)
    mol=Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError('Achiral SMILES parsing failed')
    key=Chem.MolToInchi(mol)
    if not key:
        raise ValueError('Empty standard InChI')
    return key

class Matcher:
    def __init__(self,reference):
        self.lookup={r['inchi']:r for r in reference}
        self.compositions={tuple((str(e),int(n)) for e,n in r['heavy_composition']) for r in reference}
        self.library=ctypes.CDLL(str(Path(__file__).with_name('carbon_guard.so')))
        self.guard=self.library.screening_hash
        self.guard.argtypes=[ctypes.c_char_p]
        self.guard.restype=ctypes.c_uint64
        self.guards=set()
        for row in reference:
            h=self.guard(row['smiles'].encode())
            reconstructed=Chem.MolToSmiles(Chem.MolFromInchi(row['inchi']),isomericSmiles=False)
            h2=self.guard(reconstructed.encode())
            if not h or h!=h2:
                # If normalization altered connectivity for a reference, always
                # route that whole composition through full InChI conversion.
                self.guards.add(('fallback',tuple(tuple(x) for x in row['heavy_composition'])))
            self.guards.add(h)
        self.fallback_compositions={x[1] for x in self.guards if isinstance(x,tuple)}
    def match(self,smiles):
        h=self.guard(smiles.encode())
        if h and h not in self.guards:
            if not self.fallback_compositions or fast_composition(smiles) not in self.fallback_compositions:
                return None,False
        return self.lookup.get(exact_identity(smiles)),True
