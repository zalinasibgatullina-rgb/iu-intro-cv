#!/usr/bin/env bash
# Все файлы в out/ получены этими командами (из корня репозитория). JSON-строки — в out/log.jsonl.
set -euo pipefail
P="python photolab.py"
mkdir -p out
: > out/log.jsonl
run() { echo "\$ python photolab.py $*" >&2; $P "$@" | tee -a out/log.jsonl; }

# 3.1. Цветокоррекция
run gamma data/dark.jpg --gamma 0.5 -o out/dark_g05.jpg --hist out/dark_g05_hist.png
run gamma data/dark.jpg --gamma 2.0 -o out/dark_g20.jpg --hist out/dark_g20_hist.png
run gamma data/bright.jpg --gamma 2.0 -o out/bright_g20.jpg --hist out/bright_g20_hist.png
run wb data/bright.jpg --method grey-world -o out/bright_gw.jpg --hist out/bright_gw_hist.png
run wb data/bright.jpg --method white-patch -o out/bright_wp.jpg --hist out/bright_wp_hist.png
run clahe data/dark.jpg --clip 2 --tiles 8 -o out/dark_clahe.jpg --compare out/dark_cmp.jpg --hist out/dark_clahe_hist.png

# 3.2. Хромакей
run chromakey data/object.jpg --bg data/bg.jpg --sample 760,700,110,90 -o out/object_on_bg.jpg --mask out/object_mask.png
run chromakey data/object.jpg --bg data/bg.jpg --sample 760,700,110,90 --close 31 -o out/object_on_bg_close31.jpg --mask out/object_mask_close31.png
run chromakey data/glass.jpg --bg data/bg.jpg --sample 650,900,350,250 -o out/glass_on_bg.jpg --mask out/glass_mask.png

# 3.3. Геометрия
run rotate data/poster.jpg --deg 30 -o out/poster_rot30.jpg
run rotate data/poster.jpg --deg -137 -o out/poster_rot-137.jpg
run resize data/text.jpg --scale 0.25 --report -o out/text_x025.jpg
run resize data/text.jpg --scale 0.25 --method naive -o out/text_x025_naive.jpg
run resize data/text.jpg --scale 0.25 --method blur -o out/text_x025_blur.jpg
run rectify data/poster.jpg --corners 1294,1514,335,525,224,1511,1131,534 --aspect 0.707 -o out/poster_front.jpg
run rectify data/poster.jpg --corners 1294,1514,335,525,224,1511,1131,534 -o out/poster_front_sides.jpg
run rectify data/text.jpg --corners 1234,13,418,163,436,1820,1255,1944 -o out/text_front.jpg

# 3.4. Контактный лист (снимки + результаты)
run sheet data/dark.jpg data/bright.jpg data/object.jpg data/bg.jpg data/poster.jpg data/text.jpg \
    out/dark_g05.jpg out/dark_clahe.jpg out/bright_gw.jpg out/bright_wp.jpg out/object_on_bg.jpg out/poster_rot30.jpg \
    out/poster_front.jpg out/text_front.jpg out/text_x025.jpg out/glass_on_bg.jpg --cols 4 -o out/sheet.jpg
