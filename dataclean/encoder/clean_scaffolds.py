"""Remove joint MSG/NPLIB1 val/test scaffold matches from FragHub.

Only nonempty, nonisomeric Bemis-Murcko scaffolds are matched. This step does
not independently exclude InChIKey matches. To obtain a corpus clean under
both molecular-identity and scaffold criteria, use clean_molecules.py first
and pass its output here, with the same four reference files:

python clean_scaffolds.py --input fraghub_molecule_clean.parquet \
    --output fraghub_molecule_scaffold_clean.parquet \
    --msg-labels msg/labels.tsv --msg-split msg/split.tsv \
    --nplib1-labels nplib1/labels.tsv --nplib1-split nplib1/split.tsv

For a scaffold-only control, pass the original FragHub Parquet as --input.
Empty scaffolds never match. All retained spectrum rows, SMILES strings,
columns and row order are preserved; there is no spectrum-row deduplication.
"""

from fraghub_common import run


if __name__ == '__main__':
    run('scaffolds')
