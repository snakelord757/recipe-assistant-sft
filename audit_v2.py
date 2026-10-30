"""Integrity checks for model preservation, frozen splits, and review coverage."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'runs'/'v2';DATA=ROOT/'data_v2'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(n):return [json.loads(s) for s in (DATA/(n+'.jsonl')).read_text(encoding='utf-8').splitlines()]
manifest=json.loads((RUN/'data_manifest.json').read_text(encoding='utf-8'))
for name in ['train','validation','fresh_test']:
    assert sha(DATA/(name+'.jsonl'))==manifest[name]['sha256'],name
train,val,fresh=[read(n) for n in ['train','validation','fresh_test']]
families=[{r['family_id'] for r in rows} for rows in [train,val,fresh]]
assert all(not families[i]&families[j] for i,j in [(0,1),(0,2),(1,2)])
keys=lambda rows:{(r['recipe'],r['question']) for r in rows}
assert not keys(train)&keys(fresh)
assert not keys(train)&keys(read('legacy_test'))
assert (ROOT/'data'/'test.jsonl').read_bytes()==(DATA/'legacy_test.jsonl').read_bytes()
v1sha=sha(ROOT/'adapter'/'adapter_model.safetensors')
assert v1sha==sha(ROOT/'runs'/'v1'/'adapter'/'adapter_model.safetensors')
v2sha=sha(RUN/'adapter'/'adapter_model.safetensors')
assert v1sha!=v2sha
reviews=json.loads((RUN/'reviews.json').read_text(encoding='utf-8'))
for suite,label,n in [('fresh','v2_fresh',80),('legacy','v2_legacy',45)]:
    rows=json.loads((RUN/(label+'.json')).read_text(encoding='utf-8'))
    assert len(rows)==n and set(reviews[suite])=={r['id'] for r in rows}
    assert not any(r['truncated'] for r in rows)
assert (RUN/'comparisons.html').read_text(encoding='utf-8').count('<article ')==125
summary={'frozen_hashes_match':True,'family_splits_disjoint':True,'train_test_prompt_overlap':0,'legacy_test_unchanged':True,'v1_adapter_preserved':True,'v1_adapter_sha256':v1sha,'v2_adapter_sha256':v2sha,'reviewed_questions':125,'v2_truncated_answers':0}
(RUN/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
