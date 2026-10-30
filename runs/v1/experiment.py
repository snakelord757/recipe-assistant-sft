"""Reproducible console SFT experiment. Run with .venv/Scripts/python experiment.py."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT/'models'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse, json, math, random, time, subprocess, threading, statistics, hashlib
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from build_dataset import SYSTEM

OUT = ROOT/'results'
OUT.mkdir(exist_ok=True)
MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'

def read(name):
    return [json.loads(s) for s in (ROOT/'data'/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]

def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),encoding='utf-8')

def messages(row):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':row['recipe']+'\n\nВопрос: '+row['question']}]

def prompt(tok,row):
    return tok.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)

def load():
    location=str(ROOT/'models'/'qwen-local') if (ROOT/'models'/'qwen-local'/'model.safetensors').exists() else MODEL
    tok=AutoTokenizer.from_pretrained(location,revision=REVISION)
    model=AutoModelForCausalLM.from_pretrained(location, revision=REVISION, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    return tok, model

class Power:
    """Sample whole GPU board draw; not whole-PC or incremental inference power."""
    def __init__(self): self.samples=[]; self.stop=threading.Event()
    def run(self):
        while not self.stop.is_set():
            try:
                s=subprocess.check_output(['nvidia-smi','--query-gpu=power.draw','--format=csv,noheader,nounits'],creationflags=0x08000000,text=True).strip().splitlines()[0]
                self.samples.append((time.perf_counter(),float(s)))
            except Exception: pass
            self.stop.wait(1)
    def __enter__(self):
        self.start=time.perf_counter(); self.thread=threading.Thread(target=self.run,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.elapsed=time.perf_counter()-self.start; self.stop.set(); self.thread.join(timeout=3)
    def metrics(self):
        mean=statistics.mean(w for _,w in self.samples) if self.samples else None
        return {'elapsed_s':self.elapsed,'power_samples':len(self.samples),'mean_gpu_w':mean,'gpu_energy_wh_estimate':None if mean is None else mean*self.elapsed/3600,'method':'mean 1s nvidia-smi board power samples times elapsed; includes idle/display; excludes rest of PC'}

def evaluate(tok,model,label,rows=None,adapter_applied=None):
    rows=read('test') if rows is None else rows; model.eval()
    # Warm-up excluded from timed test; same input and configuration each run.
    inp=tok('Привет',return_tensors='pt').to('cuda')
    with torch.inference_mode(): model.generate(**inp,max_new_tokens=10,do_sample=False,pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    outputs=[]
    with Power() as power:
        for row in tqdm(rows,desc=f'Test {label}',unit='answer'):
            inp=tok(prompt(tok,row),return_tensors='pt').to('cuda')
            torch.cuda.synchronize(); start=time.perf_counter()
            with torch.inference_mode():
                ids=model.generate(**inp,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id,use_cache=True)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            new=ids[0,inp.input_ids.shape[1]:]
            outputs.append({'id':row['id'],'category':row['category'],'question':row['question'],'reference':row['chosen'],'answer':tok.decode(new,skip_special_tokens=True).strip(),'seconds':elapsed,'output_tokens':len(new),'input_tokens':inp.input_ids.shape[1],'truncated':len(new)==160 and new[-1].item()!=tok.eos_token_id})
            save(label+'.json',outputs)
    lat=[x['seconds'] for x in outputs]
    stats=power.metrics()|{'model':MODEL,'adapter':label=='after' if adapter_applied is None else adapter_applied,'n':len(rows),'mean_latency_s':statistics.mean(lat),'median_latency_s':statistics.median(lat),'p95_latency_s':sorted(lat)[math.ceil(.95*len(lat))-1],'tokens_per_s':sum(x['output_tokens'] for x in outputs)/sum(lat),'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'decoding':'greedy, no thinking, max_new_tokens=160, batch_size=1','timing':'GPU synchronized generation only; excludes tokenization, model load and warmup'}
    save(label+'_metrics.json',stats)
    print(json.dumps(stats,ensure_ascii=False),flush=True)

def encoded(tok,rows):
    result=[]
    for row in rows:
        pre=tok(prompt(tok,row),add_special_tokens=False).input_ids
        ans=tok(row['chosen']+tok.eos_token,add_special_tokens=False).input_ids
        ids=pre+ans
        if len(ids)>1024: raise ValueError(f'Example too long: {row["id"]}, {len(ids)}')
        result.append({'input_ids':ids,'labels':[-100]*len(pre)+ans})
    return result

def collate(batch):
    n=max(len(x['input_ids']) for x in batch)
    return {'input_ids':torch.tensor([x['input_ids']+[151645]*(n-len(x['input_ids'])) for x in batch],device='cuda'),
            'attention_mask':torch.tensor([[1]*len(x['input_ids'])+[0]*(n-len(x['input_ids'])) for x in batch],device='cuda'),
            'labels':torch.tensor([x['labels']+[-100]*(n-len(x['labels'])) for x in batch],device='cuda')}

def val_loss(model,loader):
    model.eval(); total=0; count=0
    with torch.inference_mode():
        for batch in loader:
            loss=model(**batch).loss.item(); n=(batch['labels'][:,1:]!=-100).sum().item()
            total+=loss*n; count+=n
    return total/count

def train(tok,base,epochs=8,learning_rate=2e-4,adapter_dir=None,patience=None):
    set_seed(42)
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    model.config.use_cache=False
    train_data=encoded(tok,read('train')); val_data=encoded(tok,read('validation'))
    loader=DataLoader(train_data,batch_size=4,shuffle=True,collate_fn=collate,num_workers=0)
    valid=DataLoader(val_data,batch_size=4,collate_fn=collate,num_workers=0)
    optim=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=learning_rate,weight_decay=.01)
    accum=4; total_steps=epochs*math.ceil(len(loader)/accum); warmup=max(1,int(total_steps*.05))
    def multiplier(step):
        if step<warmup:return (step+1)/warmup
        return .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
    sched=torch.optim.lr_scheduler.LambdaLR(optim,multiplier)
    history=[]; best=float('inf'); step=0; stale=0
    adapter_dir=Path(adapter_dir) if adapter_dir else ROOT/'adapter'
    initial=val_loss(model,valid); history.append({'epoch':0,'validation_loss':initial})
    print(f'Initial validation loss: {initial:.4f}',flush=True)
    with Power() as power:
        for epoch in range(1,epochs+1):
            model.train(); optim.zero_grad(set_to_none=True); weighted=0; tokens=0
            print(f'Epoch {epoch}/{epochs}: {len(train_data)} examples, {len(loader)} batches, effective batch=16',flush=True)
            bar=tqdm(loader,desc=f'SFT epoch {epoch}/{epochs}',unit='batch',ascii=True,mininterval=1)
            for i,batch in enumerate(bar):
                loss=model(**batch).loss
                n=(batch['labels'][:,1:]!=-100).sum().item()
                weighted+=loss.item()*n; tokens+=n
                group_size=min(accum,len(loader)-(i//accum)*accum)
                (loss/group_size).backward()
                if (i+1)%accum==0 or i+1==len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optim.step(); sched.step(); optim.zero_grad(set_to_none=True); step+=1
                bar.set_postfix(loss=f'{loss.item():.4f}',mean=f'{weighted/tokens:.4f}',step=step,seen=min((i+1)*4,len(train_data)),refresh=False)
            vl=val_loss(model,valid)
            rec={'epoch':epoch,'step':step,'train_loss':weighted/tokens,'validation_loss':vl,'lr':sched.get_last_lr()[0]}
            history.append(rec); save('loss_history.json',history)
            print(json.dumps(rec),flush=True)
            if vl<best:
                best=vl; stale=0; model.save_pretrained(adapter_dir); tok.save_pretrained(adapter_dir)
                save('best_checkpoint.json',rec)
            else:stale+=1
            if patience and stale>=patience:
                print(f'Early stopping: validation did not improve for {patience} epochs.',flush=True);break
    save('training_metrics.json',power.metrics()|{'epochs':epoch,'planned_epochs':epochs,'optimizer_steps':step,'seed':42,'learning_rate':learning_rate,'batch_size':4,'gradient_accumulation':4,'lora_rank':16,'lora_alpha':32,'train_examples':len(train_data),'validation_examples':len(val_data),'loss':'assistant-only causal cross-entropy; epoch means weighted by answer tokens','selection':'minimum validation loss, no test-based selection'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot([r['epoch'] for r in history[1:]],[r['train_loss'] for r in history[1:]],'o-',label='Train')
    plt.plot([r['epoch'] for r in history],[r['validation_loss'] for r in history],'o-',label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Answer token cross-entropy'); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'loss.png',dpi=160)
    return model

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['download','before','train','after','all']); parser.add_argument('--epochs',type=int,default=8); args=parser.parse_args()
    if args.stage=='download':
        from huggingface_hub import snapshot_download
        print(snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE','README.md']))
        return
    assert torch.cuda.is_available(),'CUDA build of PyTorch is required'
    torch.set_num_threads(4); set_seed(42)
    tok,model=load()
    save('environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'model':MODEL,'revision':getattr(model.config,'_commit_hash',None) or 'c1899de289a04d12100db370d81485cdf75e47ca','weights_sha256':'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b','parameters':sum(p.numel() for p in model.parameters()),'data_sha256':{n:hashlib.sha256((ROOT/'data'/f'{n}.jsonl').read_bytes()).hexdigest() for n in ['train','validation','test']}})
    if args.stage in ['before','all']: evaluate(tok,model,'before')
    if args.stage in ['train','all']:
        trained=train(tok,model,args.epochs)
        del trained,model
        import gc
        gc.collect();torch.cuda.empty_cache()
        if args.stage=='all': tok,model=load()
    if args.stage in ['after','all']:
        model=PeftModel.from_pretrained(model,ROOT/'adapter').merge_and_unload()
        model.config.use_cache=True
        evaluate(tok,model,'after')

if __name__=='__main__': main()
