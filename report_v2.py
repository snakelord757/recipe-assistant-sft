"""Render V2 comparison from immutable model outputs and explicit review decisions."""
import json,html,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'/'v2'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def read_rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def grouped(c):return {'steps':'attention','ingredients':'attention','missing':'attention','clarification':'scaling'}.get(c,c)
NAMES={'attention':'Вопросы по рецепту','scaling':'Порции','substitution':'Замены','refusal':'Отказы'}

def main():
    reviews=load(RUN/'reviews.json')
    suites={
        'fresh':(read_rows(ROOT/'data_v2'/'fresh_test.jsonl'),load(RUN/'v1_fresh.json'),load(RUN/'v2_fresh.json')),
        'legacy':(read_rows(ROOT/'data_v2'/'legacy_test.jsonl'),load(ROOT/'runs'/'v1'/'results'/'after.json'),load(RUN/'v2_legacy.json')),
    }
    summary={};blocks=[]
    for suite,(gold,old,new) in suites.items():
        assert [r['id'] for r in gold]==[r['id'] for r in old]==[r['id'] for r in new]
        cats={};ids={r['id'] for r in gold}
        assert ids==set(reviews[suite])
        regress=[];improve=[]
        for g,b,a in zip(gold,old,new):
            r=reviews[suite][g['id']];cat=grouped(g['category'])
            c=cats.setdefault(cat,{'n':0,'v1':0,'v2':0});c['n']+=1;c['v1']+=r['v1'];c['v2']+=r['v2']
            if r['v1'] and not r['v2']:regress.append(g['id'])
            if r['v2'] and not r['v1']:improve.append(g['id'])
            esc=html.escape
            blocks.append(f'<article data-suite="{suite}"><small>{suite} / {esc(g["id"])}</small><h2>{esc(g["question"])}</h2><details><summary>Рецепт</summary><pre>{esc(g["recipe"])}</pre></details><div class="grid"><section><h3>Эталон ассистента</h3><pre>{esc(g["chosen"])}</pre></section><section><h3>V1: {r["v1"]}/1</h3><pre>{esc(b["answer"])}</pre></section><section><h3>V2: {r["v2"]}/1</h3><pre>{esc(a["answer"])}</pre></section></div><p><b>Проверка:</b> {esc(r["reason"])}</p></article>')
        summary[suite]={'n':len(gold),'v1':sum(c['v1'] for c in cats.values()),'v2':sum(c['v2'] for c in cats.values()),'categories':cats,'regressions':regress,'improvements':improve}
    (RUN/'quality_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    fresh=summary['fresh'];legacy=summary['legacy']
    subtypes={}
    for g in suites['fresh'][0]:
        if g['category']!='scaling':continue
        subtype='Все ингредиенты' if g['subtype'].startswith('all') else 'Один ингредиент с расчётом' if g['subtype'].startswith('one') else 'Перевод единиц'
        c=subtypes.setdefault(subtype,{'n':0,'v1':0,'v2':0});r=reviews['fresh'][g['id']]
        c['n']+=1;c['v1']+=r['v1'];c['v2']+=r['v2']
    subtype_table='\n'.join(f"| {k} | {v['v1']}/{v['n']} | {v['v2']}/{v['n']} |" for k,v in subtypes.items())
    met1=load(RUN/'v1_fresh_metrics.json');met2=load(RUN/'v2_fresh_metrics.json')
    train=load(RUN/'training_metrics.json');best=load(RUN/'best_checkpoint.json');history=load(RUN/'loss_history.json')
    def table(s):
        return '\n'.join(f"| {NAMES[k]} | {v['v1']}/{v['n']} | {v['v2']}/{v['n']} |" for k,v in s['categories'].items())
    report=f'''# Расширенное SFT: V1 против V2

Модель в обоих вариантах: Qwen3-0.6B. V1 обучалась на 400 правильных ответах,
V2 — на 4360 (10.9 раза больше). V2 обучалась заново от базовой модели, не от
адаптера V1. Старый адаптер и результаты сохранены в `../v1/` и исходных папках.

## Новый отложенный тест: основной результат

**V1: {fresh['v1']}/{fresh['n']} ({fresh['v1']/fresh['n']:.1%}); V2: {fresh['v2']}/{fresh['n']} ({fresh['v2']/fresh['n']:.1%}).**

| Задача | V1 | V2 |
|---|---:|---:|
{table(fresh)}

Исправлено: {len(fresh['improvements'])}; регрессий: {len(fresh['regressions'])}.
80 вопросов на 10 новых семействах рецептов; набор заморожен до запуска V2.
Чекпойнт выбран по validation loss, а не по этому тесту.

Детализация числовых заданий:

| Подзадача | V1 | V2 |
|---|---:|---:|
{subtype_table}

Метрика означает полное выполнение запроса. У V1 в четырёх вопросах на один
ингредиент правильное итоговое число, но нет явно запрошенного расчёта; это
частичное выполнение и 0 по строгому протоколу. В задачах перевода единиц
повтор исходных граммов не считается выполнением запроса ответить в килограммах.
Все такие случаи пояснены в построчном сравнении, чтобы изменение формата
ответов не было принято за чистое улучшение арифметики.

## Старый тест: регрессии на уже известных задачах

**V1: {legacy['v1']}/{legacy['n']} ({legacy['v1']/legacy['n']:.1%}); V2: {legacy['v2']}/{legacy['n']} ({legacy['v2']/legacy['n']:.1%}).**

| Задача | V1 | V2 |
|---|---:|---:|
{table(legacy)}

Исправлено: {len(legacy['improvements'])}; регрессий: {len(legacy['regressions'])}.
Эти 45 примеров не вошли в обучение, но их ошибки уже были просмотрены при
проектировании V2. Поэтому это проверка регрессий, не независимый новый тест.
Старые оценки V1 сохранены без пересмотра.

Одна оценка пограничная: замена клубники 210 г малины засчитана, хотя подготовка
описана неудачно («нарежьте, слейте»). Если считать и это существенной ошибкой,
результат V2 на старом тесте будет 37/45 (82.2%) вместо 38/45 (84.4%).

## Практический результат и оставшиеся ошибки

На новых отложенных семействах V2 воспроизводит все 80 правильных ответов,
включая 40 числовых задач. Это показывает освоение данного шаблонного
распределения, а не универсальную арифметику или знание всей кулинарии.
Эталоны новых тестов используют те же типы формулировок и правил, что обучение;
поэтому этот тест существенно легче для V2, чем произвольный пользовательский запрос.

На старых рецептах полный пересчёт ингредиентов улучшился с 1/8 до 8/8.
Например, для пудинга на 5 порций V1 давала **750 г чиа**, V2 даёт правильные
**75 г**, вместе с правильными 500 мл молока и 300 г банана.
Следующие шаги и конец рецепта: 6/9 → 9/9. Отказы сохранены: 8/8 → 8/8.
Замены на старом тесте: 1/8 → 4/8 (3/8 при более строгой оценке примера с малиной).

Есть регрессия: в вопросе о первоначальном составе пудинга V1 перечисляла
чиа, молоко и банан правильно, а V2 выдумала сметану и рисовый напиток.
Вопросы о количестве ингредиентов и о пересчёте относятся к разным запросам;
успешная арифметика не гарантирует правильное извлечение состава.

Остались ошибки: «несладкий сахар», «помидорная овсяка», неуместные инструкции
к замене сметаны/горошка, ответ мимо вопроса о калорийности и отсутствие
отклонения нулевого числа порций. Нельзя считать модель готовой к самостоятельным
советам о сложных заменах или аллергиях. Для следующего улучшения нужны
разнообразные естественные вопросы и больше примеров извлечения состава,
а не только новые числовые вариации уже освоенных шаблонов.

## Объём и методика

4360 ответов: 2080 порции, 1180 замены, 820 удержание рецепта, 280 отказы.
200 вариантов рецептов из 80 семейств. Часть примеров — шаблонные вариации порций
и формулировок, а не независимые блюда. Валидация: 245 вопросов на 15 семействах.
Семейства между train, validation и fresh-test не пересекаются, отдельные продукты
и шаблоны вопросов пересекаются. Учитель — ассистент; вычисления эталонов Decimal.

В новом обучении исходные порции варьируются от 1 до 12 (не все целые значения),
есть дробные количества, пересчёт одного продукта и перевод единиц. Многие
количества кратны исходному числу порций; это ограничивает сложность арифметики.
Замены привязаны к типу блюда и ограничениям. Внешний калькулятор при инференсе
не подключался. Никакой коррекции сгенерированных ответов не применялось.

Оба варианта: одинаковый системный промпт, BF16, greedy, thinking выключен,
batch1, предел 160 новых токенов. LoRA сливается с моделью перед измерением.
Различаются не только данные, но и LR/число эпох/число обновлений, поэтому эффект
нельзя полностью приписывать одному увеличению числа примеров.

## Прогресс и loss

Выполнено эпох: {train['epochs']}, обновлений весов: {train['optimizer_steps']},
предъявлений обучающих примеров: {train['epochs']*train['train_examples']}.
Время обучения с валидацией: **{train['elapsed_s']/60:.2f} мин**.
LoRA r16, alpha32, dropout0.05; LR 0.00008, batch2, accumulation8, seed42.
Первый короткий запуск с batch4 остановлен из-за заполнения видеопамяти;
его веса отброшены, обучение начато заново. Его лог сохранён в pilot_batch4.log.
Время и энергия этого прерванного запуска не включены в показатели завершённого SFT.

Начальный validation loss V2: {history[0]['validation_loss']:.4f}.
Лучший: **{best['validation_loss']:.4f}, эпоха {best['epoch']}**.
Train loss на выбранной эпохе: {best['train_loss']:.4f}.
Абсолютные validation loss V1 и V2 нельзя напрямую сравнивать: состав
валидационной выборки изменился. Вычисляется cross-entropy только по ответу.

![Loss V2](loss.png)

## Скорость и оценка стоимости на одном новом тесте

| Метрика | V1 | V2 |
|---|---:|---:|
| Среднее время, с/ответ | {met1['mean_latency_s']:.3f} | {met2['mean_latency_s']:.3f} |
| Медиана, с | {met1['median_latency_s']:.3f} | {met2['median_latency_s']:.3f} |
| p95, с | {met1['p95_latency_s']:.3f} | {met2['p95_latency_s']:.3f} |
| Токенов/с | {met1['tokens_per_s']:.1f} | {met2['tokens_per_s']:.1f} |
| Средняя длина, токенов | {statistics.mean(r['output_tokens'] for r in suites['fresh'][1]):.1f} | {statistics.mean(r['output_tokens'] for r in suites['fresh'][2]):.1f} |
| Энергия GPU, Wh/ответ | {met1['gpu_energy_wh_estimate']/80:.5f} | {met2['gpu_energy_wh_estimate']/80:.5f} |
| RUB/1000 ответов при 6 RUB/kWh | {met1['gpu_energy_wh_estimate']/80*6:.4f} | {met2['gpu_energy_wh_estimate']/80*6:.4f} |

Энергия обучения: примерно {train['gpu_energy_wh_estimate']:.2f} Wh для GPU,
или {train['gpu_energy_wh_estimate']/1000*6:.4f} RUB при указанном условном тарифе.
Это не полная стоимость: не включены остальные компоненты ПК, амортизация,
разработка и подготовка данных. Мощность GPU измерялась выборками nvidia-smi
раз в секунду, включая фон. В latency не включены загрузка модели и токенизация.
Плата за API локальной модели — ноль; скорость и цена учителя-чата недоступны.

## Проверяемость и ограничения

Семантическую оценку выполнил тот же ассистент, который составил эталоны.
Это не независимая слепая экспертиза. Все 125 сравнений и объяснения находятся
в comparisons.html; сырые ответы — v1_fresh.json, v2_fresh.json, v2_legacy.json.
Результат относится к небольшим синтетическим наборам, а не ко всем рецептам.
Требования к полному ответу и допустимому округлению зафиксированы в
../../evaluation_v2_protocol.md. Учитель не получает искусственный балл 100%
за совпадение с собственными ответами.

Запуск V2 в консоли:

```powershell
.\\.venv\\Scripts\\python.exe chat.py --recipe recipe_example.txt --adapter runs/v2/adapter
```

Повторить эксперимент: `.\\run_v2.ps1` (перезапишет V2, но не V1).
Лог: console.log. Модель: adapter/adapter_model.safetensors.
'''
    (RUN/'report.md').write_text(report,encoding='utf-8')
    style='body{font:16px system-ui;margin:24px;background:#edf2f7;color:#172638}article{background:white;padding:24px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}pre{white-space:pre-wrap;font:inherit;line-height:1.5}h2{font-size:20px}section{min-width:0}button{padding:10px;margin:6px}@media(max-width:850px){.grid{grid-template-columns:1fr}}'
    script="function show(s){document.querySelectorAll('article').forEach(x=>x.hidden=s!=='all'&&x.dataset.suite!==s)}"
    page=f'<!doctype html><html lang="ru"><meta charset="utf-8"><title>Рецепты: V1 и V2</title><style>{style}</style><h1>125 сравнений: учитель, V1 и V2</h1><p>Новый тест: {fresh["v1"]}/80 → {fresh["v2"]}/80. Старый тест: {legacy["v1"]}/45 → {legacy["v2"]}/45. Оценка ассистентом.</p><button onclick="show(\'all\')">Все</button><button onclick="show(\'fresh\')">Новый тест</button><button onclick="show(\'legacy\')">Старый тест</button>'+''.join(blocks)+f'<script>{script}</script></html>'
    (RUN/'comparisons.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
