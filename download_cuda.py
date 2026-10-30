"""Parallel HTTP ranges for official CUDA wheel; validates every Content-Range."""
import concurrent.futures, time
from pathlib import Path
import requests
from tqdm import tqdm
URL='https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp312-cp312-win_amd64.whl'
SIZE=3273024349
CHUNK=32*1024*1024
path=Path(__file__).resolve().parent/'.tools'/'torch-2.7.1+cu128-cp312-cp312-win_amd64.whl'
tmp=path.with_suffix('.part')
with tmp.open('wb') as f: f.truncate(SIZE)

def fetch(start):
    end=min(SIZE-1,start+CHUNK-1)
    for attempt in range(4):
        try:
            with requests.get(URL+f'?segment={start}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{SIZE}',r.headers.get('Content-Range')
                count=0
                with tmp.open('r+b') as f:
                    f.seek(start)
                    for b in r.iter_content(1024*1024):f.write(b);count+=len(b)
                assert count==end-start+1
            return count
        except Exception:
            if attempt==3: raise
            time.sleep(2)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool,tqdm(total=SIZE,unit='B',unit_scale=True,mininterval=3,desc='CUDA wheel') as bar:
    for future in concurrent.futures.as_completed([pool.submit(fetch,s) for s in range(0,SIZE,CHUNK)]):
        bar.update(future.result())
tmp.replace(path)
print(path)
