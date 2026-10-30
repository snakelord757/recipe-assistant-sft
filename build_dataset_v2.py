"""Assistant-authored recipe families with exact Decimal scaling and family-held-out splits."""
import json, random, re, hashlib
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
from pathlib import Path
from build_dataset import SYSTEM
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data_v2'
D=Decimal

def fmt(x):
    x=D(str(x)).quantize(D('.0001'),rounding=ROUND_HALF_UP)
    return format(x,'f').rstrip('0').rstrip('.') if '.' in format(x,'f') else str(x)

def number(x):return fmt(x).replace('.',',')
def write(name,rows):
    OUT.mkdir(exist_ok=True)
    (OUT/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

# Sixteen genuinely different ingredient combinations in each cooking family.
# Last four of each list are reserved at family level: 2 validation, 2 test.
SALADS=[('редька','яблоко'),('кольраби','морковь'),('сельдерей','груша'),('пекинская капуста','сладкий перец'),('редис','варёное яйцо'),('листовой салат','авокадо'),('руккола','помидоры черри'),('варёная свёкла','чернослив'),('варёный картофель','маринованный огурец'),('варёная морковь','зелёный горошек'),('белокочанная капуста','яблоко'),('огурец','сладкая кукуруза'),('романо','редис'),('варёная фасоль','сладкий перец'),('шпинат','груша'),('фенхель','апельсин')]
PORRIDGES=[('рисовые хлопья','яблоко'),('гречневые хлопья','груша'),('пшённые хлопья','тыква'),('ячменные хлопья','курага'),('ржаные хлопья','изюм'),('овсяные хлопья','персик'),('пшеничные хлопья','банан'),('полбяные хлопья','яблоко'),('мультизлаковые хлопья','чернослив'),('рисовые хлопья','манго'),('гречневые хлопья','абрикос'),('овсяные хлопья','киви'),('ячменные хлопья','финики'),('пшённые хлопья','груша'),('рисовые хлопья','нектарин'),('полбяные хлопья','слива')]
DRINKS=[('манго','банан'),('персик','малина'),('абрикос','банан'),('клубника','груша'),('голубика','яблоко'),('черника','банан'),('нектарин','персик'),('малина','манго'),('ежевика','банан'),('груша','киви'),('клубника','манго'),('вишня без косточек','банан'),('смородина','груша'),('слива','банан'),('ананас','манго'),('крыжовник','банан')]
SOUPS=[('брокколи','картофель'),('цветная капуста','морковь'),('кабачок','картофель'),('батат','морковь'),('корень сельдерея','картофель'),('тыква','цветная капуста'),('морковь','пастернак'),('репа','картофель'),('шампиньоны','картофель'),('зелёный горошек','кабачок'),('брокколи','цветная капуста'),('сладкий перец','морковь'),('пастернак','картофель'),('кольраби','морковь'),('брюква','картофель'),('корень петрушки','морковь')]
BOWLS=[('варёный рис','огурец'),('варёная гречка','помидор'),('варёный булгур','сладкий перец'),('варёная киноа','редис'),('варёная перловка','огурец'),('варёный нут','помидор'),('варёная чечевица','сладкий перец'),('варёная белая фасоль','редис'),('варёная красная фасоль','огурец'),('варёная полба','помидор'),('варёный бурый рис','сладкий перец'),('готовый кускус','редис'),('варёная зелёная чечевица','огурец'),('варёный маш','помидор'),('варёный дикий рис','сладкий перец'),('варёные макароны','редис')]

def family(kind,j,pair,variant):
    rng=random.Random(4100+kind*100+j*7+variant)
    servings=[1,2,3,4,5,6,7,8,10,12][(j+kind*3+variant*4)%10]
    a,b=pair
    dairy=['сметана','натуральный йогурт'][j%2]
    if kind==0:
        title=f'Холодный салат: {a} и {b}'
        ing=[(a,rng.choice([65,80,95,110]),'г'),(b,rng.choice([35,50,70]),'г'),(dairy,rng.choice([12.5,17.5,25,30]),'г')]
        steps=[f'Подготовьте продукты: {a} и {b}; очистите, если требуется.',f'Нарежьте {a}.',f'Нарежьте {b} и смешайте продукты.',f'Добавьте заправку «{dairy}» и перемешайте.']
        replacement='натуральный йогурт' if dairy=='сметана' else 'сметана'
        rule=(2,replacement,'Для холодной заправки подходит замена 1:1. Жирность и вкус могут отличаться.')
        restriction='Нужна заправка без молочных продуктов. Чем заменить её?'
        restricted='несладкий растительный йогурт'
        note='Проверьте состав: молока и молочных белков быть не должно.'
    elif kind==1:
        title=f'Каша из хлопьев: {a}, {b}'
        ing=[(a,rng.choice([30,40,45,50]),'г'),('молоко',rng.choice([120,150,180,200]),'мл'),(b,rng.choice([35,50,65]),'г')]
        steps=['Налейте молоко в кастрюлю и нагрейте.',f'Всыпьте {a} и варите, помешивая, до готовности по упаковке.',f'Подготовьте и мелко нарежьте {b}; для тыквы заранее приготовьте мякоть до мягкости.',f'Добавьте {b} в готовую кашу и перемешайте.']
        rule=(1,'несладкий овсяный напиток','Используйте тот же объём; вкус и густота могут измениться.')
        restriction='Чем заменить молоко при аллергии на молочный белок?'
        restricted='несладкий рисовый напиток'
        note='Проверьте состав на молочные белки. Безлактозное коровье молоко не подходит.'
    elif kind==2:
        title=f'Фруктовый напиток: {a}, {b}'
        ing=[(a,rng.choice([60,75,90]),'г'),(b,rng.choice([40,55,70]),'г'),('кефир',rng.choice([100,125,150,175]),'мл')]
        steps=[f'Вымойте {a} и {b}.','Удалите кожуру, косточки или сердцевины, если они есть.','Нарежьте мякоть и поместите в блендер.','Влейте кефир и взбейте до однородности.']
        rule=(2,'натуральный питьевой йогурт без сахара','Можно использовать тот же объём. Консистенция и кислинка могут отличаться.')
        restriction='Хочу полностью растительный напиток. Чем заменить кефир?'
        restricted='несладкий соевый напиток'
        note='Подходит, если соя допустима; проверьте отсутствие молочных компонентов. Напиток будет менее кислым.'
    elif kind==3:
        title=f'Овощной суп-пюре: {a}, {b}'
        ing=[(a,rng.choice([100,120,150]),'г'),(b,rng.choice([40,60,80]),'г'),('вода',rng.choice([180,220,250]),'мл'),('сливки',rng.choice([20,25,30,40]),'мл')]
        steps=['Вымойте, очистите и нарежьте овощи.',f'Положите {a} и {b} в кастрюлю, добавьте воду.','Варите до полной мягкости овощей.','Измельчите блендером.','Добавьте сливки и осторожно прогрейте, помешивая.']
        rule=(3,'несладкие овсяные сливки','Используйте тот же объём. Суп может стать менее сливочным по вкусу.')
        restriction='Суп должен быть без молочных продуктов. Подойдёт ли обычное коровье молоко вместо сливок?'
        restricted='несладкие растительные сливки'
        note='Нет, обычное коровье молоко не подходит для безмолочного блюда. Проверьте состав замены.'
    else:
        title=f'Зерновой салат: {a}, {b}'
        ing=[(a,rng.choice([110,130,150,180]),'г'),(b,rng.choice([50,75,90]),'г'),('оливковое масло',rng.choice([2.5,5,7.5,12]),'г'),('петрушка',rng.choice([2,3,4,5]),'г')]
        steps=[f'Остудите {a}, если продукт ещё тёплый.',f'Вымойте и нарежьте {b}.','Промойте и измельчите петрушку.','Смешайте подготовленные продукты.','Добавьте оливковое масло и перемешайте.']
        rule=(2,'подсолнечное масло','Используйте ту же массу; вкус заправки будет отличаться.')
        restriction='Можно ли заменить петрушку укропом? Укажи количество.'
        restricted='укроп'
        note='Да, используйте ту же массу или меньше по вкусу. Аромат будет другим.'
    ingredients=[{'name':n,'amount':str(D(str(q))*servings),'unit':u} for n,q,u in ing]
    return {'family':f'f{kind}-{j:02d}','recipe_id':f'f{kind}-{j:02d}-v{variant}','title':title,'servings':servings,'ingredients':ingredients,'steps':steps,'rule':rule,'restriction':restriction,'restricted':restricted,'restriction_note':note,'kind':kind}

def amounts(ing,factor=D(1)):
    return '; '.join(f"{x['name']} — {number(D(x['amount'])*factor)} {x['unit']}" for x in ing)+'.'

def examples(f):
    ing=f['ingredients'];s=f['servings'];steps=f['steps'];kind=f['kind']
    recipe=f"{f['title']}. Порций: {s}.\nИнгредиенты: {amounts(ing)}\n"+'\n'.join(f'{i+1}. {x}' for i,x in enumerate(steps))
    rows=[]
    def add(cat,sub,q,answer,rubric=None):
        rows.append({'id':f"v2-{f['recipe_id']}-{cat}-{sub}",'recipe_id':f['recipe_id'],'family_id':f['family'],'recipe':recipe,'question':q,'category':cat,'subtype':sub,'chosen':answer,'rubric':rubric or {},'teacher':'assistant_authored_with_decimal_arithmetic'})
    # Full lists include fractions, increasing/decreasing counts, and unchanged portions.
    targets=[1,2,3,5,7]
    for j,t in enumerate(targets):
        factor=D(t)/s
        q=[f'Пересчитай все ингредиенты на {t} порций.',f'Теперь нужно {t} порций вместо {s}. Сколько взять каждого продукта?',f'Адаптируй рецепт для {t} порций, сохрани единицы.'][j%3]
        answer=amounts(ing,factor)
        if j%2:answer=f'Коэффициент: {t}/{s}. '+answer
        add('scaling',f'all{t}',q,answer,{'mode':'all','ingredients':ing,'factor':str(factor),'target':t,'source':s})
    for j,t in enumerate([4,6,9]):
        x=ing[j%len(ing)];result=D(x['amount'])*D(t)/s
        add('scaling',f'one{j}',f"Сколько продукта «{x['name']}» нужно на {t} порций? Покажи расчёт.",f"{number(x['amount'])} × {t} / {s} = {number(result)} {x['unit']}.",{'mode':'single','name':x['name'],'amount':str(result),'unit':x['unit']})
    x=ing[0];kg=D(x['amount'])/1000
    add('scaling','unit',f"Сколько килограммов продукта «{x['name']}» указано в исходном рецепте?",f"{number(x['amount'])} г = {number(kg)} кг.",{'mode':'single','amount':str(kg),'unit':'кг','name':x['name']})
    idx,replacement,note=f['rule'];x=ing[idx]
    basic=f"{replacement} — {number(x['amount'])} {x['unit']}. {note}"
    add('substitution','open',f"Нет продукта «{x['name']}». Чем его заменить в этом блюде?",basic,{'replacement':replacement,'amount':x['amount'],'unit':x['unit']})
    add('substitution','specific',f"Можно ли вместо «{x['name']}» взять «{replacement}»? Сколько?",'Да. '+basic,{'replacement':replacement,'amount':x['amount'],'unit':x['unit']})
    ridx=3 if kind==4 else idx
    rx=ing[ridx]
    restrictive=f"{f['restricted']} — {number(rx['amount'])} {rx['unit']}. {f['restriction_note']}"
    add('substitution','restriction',f['restriction'],restrictive,{'replacement':f['restricted'],'constraint':f['restriction'],'amount':rx['amount'],'unit':rx['unit']})
    add('substitution','double',f"Готовлю двойную порцию всего рецепта. Чем заменить «{x['name']}» и сколько замены нужно?",f"{replacement} — {number(D(x['amount'])*2)} {x['unit']}. {note}",{'replacement':replacement,'amount':str(D(x['amount'])*2),'unit':x['unit']})
    add('substitution','flavour',f"Если заменить «{x['name']}» на «{replacement}», вкус точно не изменится?",'Нет, одинаковый вкус гарантировать нельзя. '+basic)
    add('substitution','unspecified','Хочу заменить один из ингредиентов. Подбери замену.','Уточните, какой ингредиент нужно заменить, по какой причине и какие продукты доступны.')
    for n in [1,3,len(steps)]:
        ans=steps[n] if n<len(steps) else 'Рецепт завершён; следующего шага нет.'
        add('attention',f'step{n}',f'Что делать после шага {n}?',ans,{'expected':ans})
    add('attention','missing','При какой температуре духовки готовить это блюдо?','Температура духовки в рецепте не указана. Не могу определить её по переданному тексту.')
    q=['Напиши код игры на Python.','Подбери ноутбук для дизайнера.','Объясни устройство самолёта.','Напиши стихотворение о звёздах.','Реши уравнение 2x + 1 = 9.'][kind]
    add('refusal','outside',q,'Я помогаю только с рецептами и кулинарией. Могу подсказать шаги, замену ингредиентов или пересчёт порций.')
    add('scaling','invalid','Пересчитай на ноль порций.','Количество порций должно быть больше нуля. Уточните нужное число порций.')
    add('scaling','unspecified','Пересчитай рецепт на большее число порций.','На сколько порций нужно пересчитать рецепт?')
    return rows

def build():
    splits={'train':[],'validation':[],'fresh_test':[]};catalog=[]
    for kind,pairs in enumerate([SALADS,PORRIDGES,DRINKS,SOUPS,BOWLS]):
        for j,pair in enumerate(pairs):
            split='train' if j<12 else 'validation' if j<14 else 'fresh_test'
            for variant in range(3 if split=='train' else 1):
                f=family(kind,j,pair,variant);catalog.append(f|{'split':split})
                r=examples(f)
                if split=='fresh_test':
                    # Freeze eight different tasks per new family, before model runs.
                    keep={('scaling','all2'),('scaling','all7'),('scaling','one1'),('scaling','unit'),('substitution','open'),('substitution','specific'),('substitution','restriction'),('attention','step3')}
                    r=[x for x in r if (x['category'],x['subtype']) in keep]
                splits[split].extend(r)
    # Retain old positive answers and refusal diversity; never import old validation/test into train.
    original=[json.loads(x) for x in (ROOT/'data'/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in original:r['family_id']='original-'+r['recipe_id']
    splits['train'].extend(original)
    for key,rows in splits.items():
        assert len({(x['recipe'],x['question']) for x in rows})==len(rows)
        write(key,rows)
    # Preserve old validation only as additional validation (not final test).
    oldval=[json.loads(x) for x in (ROOT/'data'/'validation.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in oldval:r['family_id']='original-'+r['recipe_id']
    splits['validation'].extend(oldval);write('validation',splits['validation'])
    families=[{r['family_id'] for r in rows} for rows in splits.values()]
    assert not(families[0]&families[1] or families[0]&families[2] or families[1]&families[2])
    oldtest=[json.loads(x) for x in (ROOT/'data'/'test.jsonl').read_text(encoding='utf-8').splitlines()]
    assert not {r['recipe'] for r in splits['train']} & {r['recipe'] for r in oldtest}
    write('legacy_test',oldtest);write('catalog',catalog)
    write('sft_train',[{'id':r['id'],'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':r['recipe']+'\n\nВопрос: '+r['question']},{'role':'assistant','content':r['chosen']}]} for r in splits['train']])
    manifest={k:{'examples':len(v),'categories':dict(Counter(r['category'] for r in v)),'families':len({r['family_id'] for r in v}),'recipes':len({r['recipe_id'] for r in v}),'sha256':hashlib.sha256((OUT/(k+'.jsonl')).read_bytes()).hexdigest()} for k,v in splits.items()}
    manifest['notes']=['Synthetic assistant-authored families; portions/quantities vary inside each family.','Splitting is by family, not randomized question rows.','Legacy test is a regression test already inspected; fresh_test is newly frozen.','No teacher API; arithmetic calculated with Decimal, rounded to 0.0001.']
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':build()
