"""Strict neutral formula parsing; never constructs a reference molecule."""
from dataclasses import dataclass
import re

MODEL_ATOMIC_NUMBERS=(5,6,7,8,9,14,15,16,17,33,34,35,53)
# Intersection of the actual frozen spectrum adapter's eleven types and the
# model configuration. As/Se in model YAML do not make the old adapter support them.
ADAPTER_ELEMENTS={'B':5,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,'Cl':17,'Br':35,'I':53}
# Frozen MSG/NP adapters accept these three raw precursor strings. The BFN
# checkpoint has TWO condition embeddings: negative mode historically uses 0.
# This is distinct from the spectrum encoder's precursor one-hot.
BFN_ADDUCT_INDEX={'[M+H]+':0,'[M-H]-':0,'[M+Na]+':1}
ADDUCTS=tuple(BFN_ADDUCT_INDEX)
SPECTRUM_PRECURSOR_INDEX={'[M+H]+':0,'[M-H]-':1,'[M+Na]+':0}

def spectrum_sos(adduct):
    """Frozen formal MSG/NP SOS: precursor(2) + default energy(11).

    Existing Zms caches must retain this upstream representation; this helper
    is for cache builders/fixtures and never changes neutral formula nodes.
    """
    if adduct not in SPECTRUM_PRECURSOR_INDEX:raise InputFailure('UNSUPPORTED_ADDUCT',str(adduct))
    import torch
    sos=torch.zeros(1,13);sos[0,SPECTRUM_PRECURSOR_INDEX[adduct]]=1.;sos[0,2]=1.
    return sos

class InputFailure(ValueError):
    def __init__(self,code,detail):self.code=code; super().__init__(f'{code}: {detail}')

@dataclass(frozen=True)
class FormulaSpec:
    text: str
    counts: tuple
    atomic_numbers: tuple
    node_types: tuple
    occurrence_slots: tuple

def _strict_parse_formula(value, *, adapter='msg', formula_kind='neutral', max_atoms=150,
                  model_atomic_numbers=MODEL_ATOMIC_NUMBERS):
    if adapter not in ('msg','nplib1','graph2mol'):raise InputFailure('UNKNOWN_ADAPTER',adapter)
    if formula_kind!='neutral':raise InputFailure('NON_NEUTRAL_FORMULA','Ion/adduct formula conversion is not enabled')
    if not isinstance(value,str) or not value.strip():raise InputFailure('MISSING_FORMULA','No target-SMILES fallback')
    text=value.strip()
    matches=list(re.finditer(r'([A-Z][a-z]?)([0-9]*)',text))
    if ''.join(m.group(0) for m in matches)!=text:raise InputFailure('FORMULA_SYNTAX',text)
    counts={}
    supported=dict(ADAPTER_ELEMENTS)
    if adapter=='graph2mol':supported.update({'As':33,'Se':34})
    for m in matches:
        symbol,raw=m.groups()
        if raw and (raw.startswith('0') or int(raw)<=0):raise InputFailure('INVALID_COUNT',m.group(0))
        if symbol in counts:raise InputFailure('DUPLICATE_ELEMENT',symbol)
        count=int(raw) if raw else 1
        if count>100000:raise InputFailure('INVALID_COUNT','Unreasonable element count')
        if symbol!='H' and (symbol not in supported or supported[symbol] not in model_atomic_numbers):
            raise InputFailure('UNSUPPORTED_ELEMENT',symbol)
        counts[symbol]=count
    atoms=[]; slots=[]
    for symbol,z in sorted(supported.items(),key=lambda kv:kv[1]):
        for occurrence in range(counts.get(symbol,0)):
            atoms.append(z); slots.append((symbol,occurrence))
    if not atoms:raise InputFailure('NO_HEAVY_ATOMS',text)
    if len(atoms)>max_atoms:raise InputFailure('ATOM_BUDGET',str(len(atoms)))
    return FormulaSpec(text,tuple(sorted(counts.items())),tuple(atoms),
                       tuple(model_atomic_numbers.index(z) for z in atoms),tuple(slots))

def parse_formula(value, **kwargs):
    """Strict neutral formula, atomic-number grouped slots; no reference graph."""
    return _strict_parse_formula(value, **kwargs)

def acquisition_indices(instrument,adduct,adapter):
    if adduct not in ADDUCTS:raise InputFailure('UNSUPPORTED_ADDUCT',str(adduct))
    instrument=str(instrument or 'NONE')
    if adapter=='nplib1':
        if 'orbitrap' in instrument.lower():instrument='Orbitrap'
        elif 'q-tof' in instrument.lower() or 'qtof' in instrument.lower():instrument='QTOF'
    return (['Orbitrap','QTOF','NONE'].index(instrument) if instrument in ['Orbitrap','QTOF','NONE'] else 2,
            BFN_ADDUCT_INDEX[adduct])
