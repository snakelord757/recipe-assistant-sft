"""V1 on new frozen test -> expanded SFT from base -> V2 on both tests."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json, shutil, gc, hashlib
from pathlib import Path
import torch
from peft import PeftModel
from transformers import set_seed
import experiment as E
ROOT=E.ROOT
RUN=ROOT/'runs'/'v2'
DATA=ROOT/'data_v2'
RUN.mkdir(parents=True,exist_ok=True)
E.OUT=RUN

def read(name):return [json.loads(s) for s in (DATA/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()]
E.read=read

def state(stage):
    E.save('status.json',{'stage':stage})
    print('\n'+'='*70+'\n'+stage+'\n'+'='*70,flush=True)

def main():
    archive=ROOT/'runs'/'v1'
    if not archive.exists():
        archive.mkdir(parents=True)
        for name in ['adapter','results','data']:shutil.copytree(ROOT/name,archive/name)
        for name in ['experiment.py','build_dataset.py','review_results.py']:shutil.copy2(ROOT/name,archive/name)
    # Snapshot code/data manifest; V1 results and original adapter are never overwritten.
    for name in ['experiment_v2.py','experiment.py','build_dataset_v2.py']:shutil.copy2(ROOT/name,RUN/name)
    shutil.copy2(DATA/'manifest.json',RUN/'data_manifest.json')
    set_seed(42);torch.set_num_threads(4)
    E.save('config.json',{'initialization':'fresh Qwen3-0.6B, not V1 adapter','epochs':3,'learning_rate':8e-5,'rank':16,'batch_size':2,'accumulation':8,'seed':42,'patience':2,'test_selection':'none; minimum validation loss','old_test_role':'regression: already inspected during V1','fresh_test_role':'held out by recipe family, frozen before this run','prompt':'unchanged from V1','generation':'unchanged from V1: greedy, no thinking, max_new_tokens=160','train_sha256':hashlib.sha256((DATA/'train.jsonl').read_bytes()).hexdigest()})
    if not (RUN/'v1_fresh_metrics.json').exists():
        state('V1: evaluation on 80 NEW held-out questions')
        tok,base=E.load();model=PeftModel.from_pretrained(base,ROOT/'adapter').merge_and_unload()
        E.evaluate(tok,model,'v1_fresh',rows=read('fresh_test'),adapter_applied=True)
        del model,base;gc.collect();torch.cuda.empty_cache()
    else:
        state('Reusing completed V1 fresh baseline; restarting training from base')
    state('V2 SFT: 4360 examples, 3 epochs, batch 2, accumulation 8')
    tok,base=E.load()
    trained=E.train(tok,base,epochs=3,learning_rate=8e-5,adapter_dir=RUN/'adapter',patience=2,batch_size=2,accumulation=8)
    del trained,base;gc.collect();torch.cuda.empty_cache()
    state('V2: load BEST validation checkpoint and evaluate')
    tok,base=E.load();model=PeftModel.from_pretrained(base,RUN/'adapter').merge_and_unload();model.config.use_cache=True
    E.evaluate(tok,model,'v2_fresh',rows=read('fresh_test'),adapter_applied=True)
    E.evaluate(tok,model,'v2_legacy',rows=read('legacy_test'),adapter_applied=True)
    state('GENERATION COMPLETE. Semantic review and comparison report pending.')

if __name__=='__main__':main()
