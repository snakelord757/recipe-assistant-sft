"""Console recipe assistant using the trained adapter; no external inference API."""
import argparse, json
import torch
from peft import PeftModel
from experiment import ROOT, load, prompt

def main():
    p=argparse.ArgumentParser(description='Local recipe assistant')
    p.add_argument('--recipe',required=True,help='UTF-8 text file containing the recipe')
    p.add_argument('--base',action='store_true',help='Use the model before SFT')
    p.add_argument('--adapter',default=str(ROOT/'adapter'),help='Adapter directory (V1 by default; V2: runs/v2/adapter)')
    args=p.parse_args()
    recipe=__import__('pathlib').Path(args.recipe).read_text(encoding='utf-8')
    torch.set_num_threads(4)
    tok,model=load()
    if not args.base: model=PeftModel.from_pretrained(model,args.adapter).merge_and_unload()
    model.eval()
    print('Введите вопрос по рецепту. Выход: /exit. Каждый вопрос независим.')
    while True:
        try: q=input('> ').strip()
        except (EOFError,KeyboardInterrupt): break
        if q=='/exit': break
        if not q: continue
        inputs=tok(prompt(tok,{'recipe':recipe,'question':q}),return_tensors='pt').to('cuda')
        if inputs.input_ids.shape[1]>4096:
            print('Рецепт и вопрос слишком длинные: лимит приложения 4096 токенов.'); continue
        with torch.inference_mode():
            ids=model.generate(**inputs,max_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id)
        print(tok.decode(ids[0,inputs.input_ids.shape[1]:],skip_special_tokens=True))

if __name__=='__main__': main()
