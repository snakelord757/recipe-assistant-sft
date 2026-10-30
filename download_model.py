"""Direct streamed download fallback for networks with stalled Hub downloads."""
from pathlib import Path
import requests
import hashlib
from tqdm import tqdm
root=Path(__file__).resolve().parent/'models'/'qwen-local'
root.mkdir(parents=True,exist_ok=True)
for name in ['config.json','generation_config.json','tokenizer_config.json','tokenizer.json','merges.txt','vocab.json','model.safetensors']:
    target=root/name
    if target.exists(): continue
    url=f'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/{name}'
    with requests.get(url,stream=True,timeout=(20,60)) as r:
        r.raise_for_status()
        tmp=target.with_suffix(target.suffix+'.part')
        expected=int(r.headers.get('content-length',0))
        with tmp.open('wb') as f,tqdm(total=expected or None,unit='B',unit_scale=True,desc=name) as bar:
            for chunk in r.iter_content(1024*1024):
                f.write(chunk); bar.update(len(chunk))
        if expected and tmp.stat().st_size!=expected: raise IOError('Incomplete download')
        tmp.replace(target)
with (root/'model.safetensors').open('rb') as f:
    actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual=='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b', 'Weights hash mismatch'
