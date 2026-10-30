"""Build a reproducible report from saved outputs and explicit assistant review labels."""
import json, html, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
def load(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def task(c):return {'steps':'attention','ingredients':'attention','missing':'attention','clarification':'scaling'}.get(c,c)
names={'attention':'Вопросы по рецепту','scaling':'Порции','substitution':'Замены','refusal':'Отказы'}

def main():
    before=load('before.json');after=load('after.json');reviews=load('review.json')
    b={r['id']:r for r in before}; a={r['id']:r for r in after}
    assert set(b)==set(a)==set(reviews)
    metrics={k:load(k+'_metrics.json') for k in ['before','after']}
    train=load('training_metrics.json');best=load('best_checkpoint.json');hist=load('loss_history.json')
    rows=[];summary={}
    for category in names:
        ids=[r['id'] for r in before if task(r['category'])==category]
        counts={k:sum(reviews[i][k] for i in ids) for k in ['before','after']}
        summary[category]={'n':len(ids),**counts}
        rows.append(f"| {names[category]} | {counts['before']}/{len(ids)} | {counts['after']}/{len(ids)} |")
    total={k:sum(v[k] for v in reviews.values()) for k in ['before','after']}
    regress=[i for i,v in reviews.items() if v['before'] and not v['after']]
    improvements=[i for i,v in reviews.items() if not v['before'] and v['after']]
    (OUT/'quality_summary.json').write_text(json.dumps({'by_task':summary,'total':total,'n':len(before),'improvements':improvements,'regressions':regress},ensure_ascii=False,indent=2),encoding='utf-8')
    tariff=6
    def cost(m):return m['gpu_energy_wh_estimate']/1000*tariff
    report=f'''# Результаты локального SFT

Ученик: Qwen3-0.6B, LoRA SFT на 400 правильных ответах. GPU: RTX 5060 Ti 16 GB.
Учитель: ассистент этого чата; его заранее составленные ответы служат эталонами.
Неправильные ответы не использованы в SFT. Валидация: 25 вопросов, тест: 45.
Рецепты между выборками не пересекаются.

## Качество

Полностью корректные ответы: **{total['before']}/45 ({total['before']/45:.1%}) → {total['after']}/45 ({total['after']/45:.1%})**.
Исправлено ответов: {len(improvements)}; регрессий: {len(regress)}.

| Задача | До SFT | После SFT |
|---|---:|---:|
{chr(10).join(rows)}

Критерии зафиксированы в evaluation_protocol.md. Каждый ответ оценён ассистентом
по смыслу и проверяемым фактам; все решения с пояснениями находятся в review.json.
Это не независимая слепая оценка: эталоны и оценка принадлежат тому же ассистенту.
Качество учителя отдельно не измерялось: 100% совпадения с собственными эталонами
не является содержательной метрикой. В comparisons.html приведены все три ответа.
Тест маленький и синтетический; несколько вопросов используют один рецепт.
Проценты не доказывают такую же надёжность на произвольных пользовательских данных.

Строгость оценки: один исходный ответ содержал общий отказ без указания кулинарной
области и получил 0 по заранее заданному критерию. Если засчитать такой отказ,
исходный результат будет 12/45 (26.7%), а после SFT останется 24/45 (53.3%).
Для нуля порций модель после SFT дала математически верные нулевые количества,
но не выполнила предусмотренную протоколом проверку допустимости числа порций.
Это прикладная ошибка, а не ошибка умножения.

## Где ученик уже справляется, а где ещё нет

- Перечисление ингредиентов: 8/8 до и после; после встречаются опечатки названий,
  но состав и количества в этих вопросах верны. Улучшения в этой подзадаче нет.
- Отказы: 8/8 после SFT по строгому критерию, 0/8 до (1/8 при мягком критерии).
  На этих запросах ученик выполняет роль ограниченного кулинарного помощника.
- Следующий шаг: 3/9 → 6/9, включая вопрос о завершении рецепта. После последнего
  шага модель всё ещё повторяет рецепт или выдаёт постороннюю часть контекста.
- Пересчёт: 0/8 → 1/8; два дополнительных вопроса на уточнение также не пройдены.
  Надёжно заменять учителя в арифметике нельзя.
- Замены: 0/8 → 1/8 полностью корректных ответов. Иногда названа подходящая
  альтернатива, но следом идёт выдуманная инструкция. Такие ответы не засчитывались.
- Отсутствующие сведения: 0/2 до и после. Нельзя полагаться на распознавание
  недостаточной информации или на советы при аллергиях.

Пример исправления: после первого шага булгура учитель отвечает «Залейте водой
и варите 15 минут». До SFT модель предлагала обжарить булгур, после — точно
воспроизводит нужный следующий шаг.

Пример оставшейся ошибки: на 5 порций пудинга учитель даёт 75 г чиа, 500 мл молока,
300 г банана. До SFT: 135 г / 1500 мл / 900 г. После: **750 г / 500 мл / 300 г**.
Улучшение двух чисел не делает весь ответ правильным.

Пример замены: на вопрос о замене йогурта сметаной учитель предлагает 75 г сметаны.
До SFT модель запрещает замену. После называет 75 г, но добавляет «нарежьте и
слейте» — такой ответ также ошибочный.

Итог: получен рабочий обученный прототип, но не надёжный универсальный помощник.
Для следующего эксперимента полезнее расширить разнообразие рецептов, замен,
пограничных случаев и исходного числа порций, чем повторять те же примеры ещё
больше раз. В текущем обучении исходные рецепты рассчитаны на 2 или 4 порции,
а отложенные — на 3: это существенное ограничение переноса арифметики.
Для практического пересчёта разумно вынести вычисления в проверяемый калькулятор;
это предложение для следующей версии, в данном тесте инструментов у модели нет.

## Обучение и loss

Время этапа обучения: {train['elapsed_s']/60:.2f} мин; {train['optimizer_steps']} шагов оптимизатора.
Начальный validation loss: {hist[0]['validation_loss']:.4f}.
Лучший validation loss: **{best['validation_loss']:.4f}**, эпоха {best['epoch']}.
Train loss на этой эпохе: {best['train_loss']:.4f}.
Loss — cross-entropy по токенам правильного ответа, без токенов вопроса/рецепта.
Выбран чекпойнт с минимальным validation loss, не по результатам теста.
Снижение loss не заменяет проверку фактических ответов.

![Loss](loss.png)

## Скорость и стоимость

| Метрика | До SFT | После SFT |
|---|---:|---:|
| Среднее время генерации, с/запрос | {metrics['before']['mean_latency_s']:.3f} | {metrics['after']['mean_latency_s']:.3f} |
| Медиана, с | {metrics['before']['median_latency_s']:.3f} | {metrics['after']['median_latency_s']:.3f} |
| p95, с | {metrics['before']['p95_latency_s']:.3f} | {metrics['after']['p95_latency_s']:.3f} |
| Выходных токенов/с | {metrics['before']['tokens_per_s']:.1f} | {metrics['after']['tokens_per_s']:.1f} |
| Средняя длина ответа, токены | {statistics.mean(r['output_tokens'] for r in before):.1f} | {statistics.mean(r['output_tokens'] for r in after):.1f} |
| Пик выделенной CUDA-памяти, GB | {metrics['before']['peak_allocated_gb']:.2f} | {metrics['after']['peak_allocated_gb']:.2f} |
| Оценка энергии GPU на запрос, Wh | {metrics['before']['gpu_energy_wh_estimate']/45:.5f} | {metrics['after']['gpu_energy_wh_estimate']/45:.5f} |
| Энергия GPU, RUB/1000 запросов при 6 RUB/kWh | {cost(metrics['before'])/45*1000:.4f} | {cost(metrics['after'])/45*1000:.4f} |

Скорость измерена локально после прогрева, batch size 1, без времени загрузки модели
и токенизации. Длины ответов различаются, поэтому сравнивайте также токены/с.
Оба запуска: BF16, greedy, thinking выключен, предел 160 новых токенов.
LoRA после обучения слит с базовыми весами для измерения.

Энергия — приближённая оценка по выборкам nvidia-smi раз в секунду, включая фон GPU,
но без CPU, памяти и других компонентов ПК. Тариф 6 RUB/kWh — пример расчёта,
не установленный тариф пользователя. Оценка энергии обучения: {train['gpu_energy_wh_estimate']:.3f} Wh,
или {cost(train):.4f} RUB по тому же условному тарифу. Амортизация, труд и загрузки не включены.
Плата за API локальной модели равна нулю. Полную стоимость владения не измеряли.
Для учителя в этом чате время инференса, энергопотребление и цена запроса недоступны;
сопоставимые числа для него не выдумывались.

## Воспроизведение и артефакты

- console.log: полный консольный прогресс.
- loss_history.json, loss.png: loss всех эпох.
- before.json, after.json: реальные ответы и индивидуальные измерения.
- review.json, quality_summary.json: оценки и сводка.
- comparisons.html: рецепт, вопрос, эталон, ответы до/после, причина оценки.
- environment.json: модель, GPU, версия PyTorch и хэши выборок.
- ../adapter/: обученный LoRA-адаптер.

Запуск: `./run_experiment.ps1`. Консольный помощник:
`./.venv/Scripts/python.exe chat.py --recipe recipe_example.txt`.
Для исходной модели добавьте `--base`. Каждый вопрос обрабатывается независимо;
многотуровая память этим экспериментом не проверялась.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    data={r['id']:r for r in [json.loads(s) for s in (ROOT/'data'/'test.jsonl').read_text(encoding='utf-8').splitlines()]}
    esc=html.escape
    blocks=[]
    for i,row in enumerate(before):
        key=row['id']; r=reviews[key]
        blocks.append(f'''<article><h2>{i+1}. {esc(row['question'])}</h2><p>{esc(key)}</p><details><summary>Рецепт</summary><pre>{esc(data[key]['recipe'])}</pre></details><div class="grid"><section><h3>Эталон ассистента</h3><pre>{esc(row['reference'])}</pre></section><section><h3>До SFT: {r['before']}/1</h3><pre>{esc(row['answer'])}</pre></section><section><h3>После SFT: {r['after']}/1</h3><pre>{esc(a[key]['answer'])}</pre></section></div><p><b>Оценка:</b> {esc(r['reason'])}</p></article>''')
    (OUT/'comparisons.html').write_text('''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Сравнение ответов</title><style>body{font:16px system-ui;margin:24px;background:#f3f5f8;color:#172638}article{background:white;padding:24px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}pre{white-space:pre-wrap;font:inherit;line-height:1.5}h2{font-size:20px}section{min-width:0}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><h1>45 тестов: учитель и ученик до/после SFT</h1><p>Оценка ассистентом по зафиксированным критериям. Это не независимая слепая экспертиза.</p>'''+''.join(blocks)+'</html>',encoding='utf-8')
    print(json.dumps({'total':total,'by_task':summary},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
