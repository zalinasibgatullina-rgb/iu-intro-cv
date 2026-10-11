# ДЗ 1 «Фотолаборатория»

`photolab.py` — инструмент командной строки: цветокоррекция (гамма, баланс белого, CLAHE), хромакей,
геометрия (поворот с холстом, масштаб с антиалиасингом, выпрямление по 4 углам) и контактный лист.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # opencv-contrib-python==4.14.0.94
pytest tests -q                                          # 11 passed
bash make_out.sh                                         # повторяет ВСЕ файлы в out/ и пишет out/log.jsonl
```

| Путь | Что там |
|------|---------|
| `photolab.py` | инструмент; 11 функций реализованы |
| `REPORT.md` | отчёт: числа, три ответа, «где сломалось», команды |
| `make_out.sh` | все команды, которыми получены файлы в `out/` |
| `report_numbers.py` | recall / precision мяча и совпадение с `poster-gt.png` на кадрах курса |
| `data/` | мои снимки: `dark`, `bright`, `object`, `bg`, `poster`, `text` (+ `glass` — неудачный хромакей) |
| `out/` | результаты команд и `log.jsonl` |
| `tests/`, `.github/workflows/` | самопроверка и CI (даны) |
