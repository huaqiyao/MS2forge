import json
from pathlib import Path
r=Path(__file__).resolve().parent
d=json.loads((r/'cleaning_status.json').read_text())
print(json.dumps({k:d.get(k) for k in ('complete','phase','scanned_rows','expected_rows','removed_rows','retained_rows','verified_shards','expected_shards','elapsed_seconds','failures')},indent=2))
for key,value in d['splits'].items():
    print(key,{k:value[k] for k in ('input_rows','removed_rows','retained_rows','verified_shards','cross_msg_train_npl_eval_rows','cross_npl_train_msg_eval_rows')})
