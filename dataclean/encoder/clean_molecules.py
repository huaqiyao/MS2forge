"""Remove joint MSG/NPLIB1 val/test molecular-identity matches from FragHub.

Exclusion uses full Standard InChIKey OR InChIKey-14, not scaffold matching.
Example (run from this folder; replace the input paths):

python clean_molecules.py --input FragHub_filtered.parquet \
    --output fraghub_molecule_clean.parquet \
    --msg-labels msg/labels.tsv --msg-split msg/split.tsv \
    --nplib1-labels nplib1/labels.tsv --nplib1-split nplib1/split.tsv

This script and clean_scaffolds.py share fraghub_common.py. All four held-out
partitions are always included jointly; benchmark training rows are excluded
from the reference. Existing outputs are never overwritten.
"""

from fraghub_common import run


if __name__ == '__main__':
    run('molecules')
