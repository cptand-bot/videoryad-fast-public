# -*- coding: utf-8 -*-
"""СПЛИТ-ЭКРАН «ЗА/ПРОТИВ» — канал МАТРИЦА ВЛАСТИ (25.11.2026).

Решение 25.11 (см. МАТРИЦА-ВЛАСТИ-СОСТОЯНИЕ-ЧИТАТЬ-ПЕРВЫМ.md): карточка
`duel` (статичный текст сверху/снизу) отклонена категорически — «убожество,
таскаешь одну хрень по всему видеоряду». Нужен настоящий ДИНАМИЧЕСКИЙ
сплит-экран: две стороны side-by-side ВЕСЬ РОЛИК, подсветка того, кто
говорит сейчас, аудио-полоса по РЕАЛЬНОЙ амплитуде голоса (не на глаз),
камера в медленном движении, переход между репликами — не хард-кат, а
подсветка, которая переезжает на другую сторону.

ПРАВКА ТОЙ ЖЕ НОЧИ (после первого облачного прогона): первая версия несла
кадр СТЕНОЙ ТЕКСТА (вся реплика целиком) + аудио-полосой — его оценка
дословно: «это что, подкаст?» И он прав: ровно так выглядит любой
генератор подкаст-аудиограмм (две плашки + текст + полоска, без картинки).
Теперь кадр несёт ФОТО активной стороны (реальное, из своей маленькой
библиотеки — решение 25.11: копим раз под повторяющиеся темы, не фетчим
на каждый ролик), а подпись — КОРОТКАЯ фраза-заголовок (не пересказ целиком:
слова и так несёт субтитр внизу всего ролика). Подсветка и волна остались,
но как ИНДИКАТОР «кто говорит», не как замена картинки.

Один вызов = одна реплика ОДНОЙ стороны (как у диалоговой озвучки VoiceGen:
сцена 1:1 с репликой). Переход на экране получается НЕ склейкой внутри
кадра, а тем, что подсветка/полоса у каждого клипа проявляются с его первой
секунды — на стыке клипов это читается как «подсветка перескочила».
"""
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

import style_cards as sc

W, H = sc.W, sc.H
MID_X = W // 2
TOP_STRIP = 140          # дарк-полоса сверху: кикер, имя, волна — ВСЕГДА на
                         # холсте, никогда поверх фото (контраст непредсказуем)
FOOT_SAFE = 800           # ниже этой линии — заголовок уже не идёт, там подвал

# цвет СТОРОНЫ закреплён навечно, как и цвет говорящего у `duel` —
# «цвет имени не меняется весь ролик», то же правило, просто сторон
# ровно две и они известны заранее (не N говорящих).
SIDE_COLOR = {"za": (214, 58, 52), "protiv": (86, 170, 232)}
SIDE_LABEL = {"za": "ЗА", "protiv": "ПРОТИВ"}

_ENV_CACHE = {}
_PHOTO_CACHE = {}
_GRAD_CACHE = {}


def _envelope(path, rate=50):
    """Огибающая громкости файла: один float 0..1 на каждую 1/rate секунды.

    Декод в сырой PCM через ffmpeg — ТОТ ЖЕ приём, что `style_cards.
    _video_frames` уже применяет к видео. Эта функция исполняется внутри
    кадровой рисовалки, то есть ТОЛЬКО на раннере в облаке, никогда на его
    машине — экстракция амплитуды не легче самого рендера, считать его
    локально нельзя по тому же правилу, что и финальную склейку.
    """
    if path in _ENV_CACHE:
        return _ENV_CACHE[path]
    sr = 4000
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
             "-f", "f32le", "-"],
            capture_output=True, check=True).stdout
        pcm = np.frombuffer(raw, dtype=np.float32)
        if len(pcm) == 0:
            raise ValueError("пустой звук")
    except Exception as exc:
        print("огибающая голоса: ffmpeg не прочитал «%s»: %s" % (path, exc),
              file=sys.stderr)
        pcm = np.zeros(sr, dtype=np.float32)
    win = max(1, sr // rate)
    n = max(1, len(pcm) // win)
    pcm = pcm[:n * win].reshape(n, win)
    rms = np.sqrt(np.mean(pcm.astype(np.float64) ** 2, axis=1))
    peak = float(np.percentile(rms, 97)) or 1.0
    env = np.clip(rms / peak, 0.0, 1.0)
    k = 3                                  # сглаживание: сырой RMS по 20 мс дрожит
    if len(env) > k:
        env = np.convolve(env, np.ones(k) / k, mode="same")
    _ENV_CACHE[path] = (env, rate)
    return env, rate


def _amp_at(env, rate, t):
    if env is None or len(env) == 0:
        return 0.0
    i = int(round(t * rate))
    return float(env[max(0, min(len(env) - 1, i))])


def _step_k(t, dur, steps=3):
    """Прогресс 0..1 по ступеням smootherstep — тот же профиль, что держит
    хвост обычных карточек («зум медленными ступенями», 21.10), только
    ведёт камеру ВСЮ сцену, а не последний кусок."""
    if dur <= 0:
        return 0.0
    p = max(0.0, min(1.0, t / dur))
    q = p * steps
    k = min(steps - 1, int(q))
    frac = q - k
    rise = 0.8
    local = min(1.0, frac / rise) if rise > 0 else 1.0
    s = local * local * local * (local * (local * 6 - 15) + 10)   # smootherstep
    return (k + s) / steps


def _step_zoom(t, dur, amount=0.045, steps=3):
    return 1.0 + amount * _step_k(t, dur, steps)


def _zoom_bg(im, scale):
    if scale <= 1.0:
        return im
    w, h = im.size
    nw, nh = int(round(w * scale)), int(round(h * scale))
    big = im.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return big.crop((x0, y0, x0 + w, y0 + h))


def _load_photo(path):
    if path not in _PHOTO_CACHE:
        _PHOTO_CACHE[path] = Image.open(path).convert("RGB")
    return _PHOTO_CACHE[path]


def _photo_frame(path, box_w, box_h, k, zoom_to=1.08):
    img = _load_photo(path)
    return sc._kb_zoom_img(img, box_w, box_h, k, zoom_to=zoom_to)


def _bottom_gradient(box_w, box_h, start=0.3, peak_alpha=235):
    """Тёмная растяжка снизу фото — ЕДИНСТВЕННО ради того, чтобы заголовок
    и подвал читались на любой картинке. Кэш: растяжка одинаковая на каждый
    кадр, пересчитывать её покадрово незачем."""
    key = (box_w, box_h, start, peak_alpha)
    if key in _GRAD_CACHE:
        return _GRAD_CACHE[key]
    h0 = int(box_h * start)
    col = np.zeros((box_h,), dtype=np.float32)
    if box_h > h0:
        col[h0:] = np.linspace(0.0, 1.0, box_h - h0) ** 1.3
    alpha = (col * peak_alpha).astype(np.uint8)
    arr = np.tile(alpha[:, None], (1, box_w))
    a = Image.fromarray(arr, mode="L")
    rgba = Image.merge("RGBA", (Image.new("L", (box_w, box_h), 0),
                                Image.new("L", (box_w, box_h), 0),
                                Image.new("L", (box_w, box_h), 0), a))
    _GRAD_CACHE[key] = rgba
    return rgba


def _wave(d, cx, cy, w, color, amp, t, grow, bars=18, max_h=14):
    """Эквалайзер-ИНДИКАТОР (не хедлайнер): маленькая полоска «кто сейчас
    говорит», столбики зеркально вверх-вниз, каждый со своей фазой — иначе
    это пульсирующий блок, а не бегущая волна."""
    bw = w / bars
    for i in range(bars):
        ph = (i - bars / 2.0) * 0.2
        local = max(0.0, min(1.0, amp + 0.26 * np.sin(5.4 * t + ph)))
        hgt = (2 + local * max_h) * grow
        if hgt < 0.5:
            continue
        x = cx - w / 2 + i * bw + bw * 0.18
        bww = bw * 0.64
        d.rectangle((x, cy - hgt, x + bww, cy + hgt), fill=color)


def dialogue_split(out=None, kick="", za_name="ЗА", protiv_name="ПРОТИВ",
                    active="za", photo="", headline="", accent=None,
                    audio="", source="", watermark="МАТРИЦА ВЛАСТИ",
                    dur=6.0, t=None, sfx=None):
    """Один клип = одна реплика одной стороны.

    active     — "za" | "protiv": кто говорит в ЭТОМ клипе;
    photo      — фото активной стороны (корабль/флаг/лицо — ГЛАВНЫЙ
                 визуальный факт кадра), из своей маленькой библиотеки;
    headline   — КОРОТКАЯ фраза-заголовок (1-2 строки), НЕ пересказ реплики
                 целиком — слова несёт субтитр внизу всего ролика;
    audio      — путь к ОТДЕЛЬНОМУ аудио-файлу ИМЕННО этой реплики (не
                 всего ролика) — огибающая считается от него, без смещений;
    dur        — длительность сцены в секундах, нужна для зума камеры.
    """
    sc._ev(sfx, 0.0, "card")
    im = sc.canvas().convert("RGB")

    tt = 0.0 if t is None else t
    bgscale = _step_zoom(tt, dur, amount=0.03)
    im = _zoom_bg(im, bgscale)

    inactive = "protiv" if active == "za" else "za"
    col = {active: SIDE_COLOR[active], inactive: SIDE_COLOR[inactive]}
    names = {"za": za_name, "protiv": protiv_name}
    halves = {"za": (0, MID_X), "protiv": (MID_X, W)}
    reveal, _ = sc._rv(t, 0.0)
    k = _step_k(tt, dur)

    # --- фото активной стороны — НЕСЁТ кадр, не текст ------------------------
    box_w, box_h = MID_X, H - TOP_STRIP
    ax0, ax1 = halves[active]
    if photo:
        try:
            ph = _photo_frame(photo, box_w, box_h, k)
            im.paste(ph, (ax0, TOP_STRIP))
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            layer.paste(_bottom_gradient(box_w, box_h), (ax0, TOP_STRIP))
            im = Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")
        except Exception as exc:
            print("splitscreen: фото «%s» не открылось: %s: %s"
                  % (photo, type(exc).__name__, exc), file=sys.stderr)
            photo = ""

    # молчащая сторона без своего фото — просто тёмная (настоящий альфа-слой:
    # прямой ImageDraw на RGB-кадре альфу молча игнорирует и красит насмерть
    # — поймано на первом пробном кадре этой же ночи). Говорящая сторона без
    # фото попадает сюда же тем же циклом — одна причина, одна заливка.
    for side, (x0, x1) in halves.items():
        if side == active and photo:
            continue
        im = sc._put(im, lambda d, x0=x0, x1=x1: d.rectangle(
            (x0, TOP_STRIP, x1, H), fill=(0, 0, 0, 255)), 0.42 * reveal, 0)

    # рамка по краю говорящей половины — тонкая, вместо сплошной заливки
    # (заливка перекрывала бы фото)
    bw = 5
    im = sc._put(im, lambda d, x0=ax0, x1=ax1, bw=bw, c=col[active]: (
        d.rectangle((x0, TOP_STRIP, x0 + bw, H), fill=c + (255,)),
        d.rectangle((x1 - bw, TOP_STRIP, x1, H), fill=c + (255,)),
        d.rectangle((x0, H - bw, x1, H), fill=c + (255,))), reveal, 0)

    d = ImageDraw.Draw(im)
    d.line([(MID_X, 20), (MID_X, TOP_STRIP - 10)], fill=(66, 66, 70), width=2)

    if kick:
        sc.kicker(d, 20, kick, dot=True, t=t)

    # --- плашки имён + волна-индикатор, ВСЕГДА на тёмной полосе сверху ------
    # (черта под именем убрана той же ночью — она лезла в ту же полосу, что
    # и волна, и на пробном кадре получилась каша; цвет имени + волна и так
    # достаточно разделяют «кто говорит», черта оказалась лишней)
    f_name = sc._f(sc.F_ALT, 36)
    for side, (x0, x1) in halves.items():
        cx = (x0 + x1) // 2
        label = names[side].upper()
        wtxt = sc.track_w(label, f_name, 4)
        c = col[side] if side == active else sc.GREY_DIM
        sc.track(d, (cx - wtxt // 2, 52), label, f_name, c, 4)

    env, rate = _envelope(audio) if audio else (None, 50)
    amp = _amp_at(env, rate, tt) if env is not None else 0.5
    _wave(d, (ax0 + ax1) // 2, TOP_STRIP - 20, 240, col[active], amp, tt, reveal)

    # --- заголовок реплики — КОРОТКАЯ фраза, якорь у нижнего края фото ------
    if headline:
        from quote_variants import _wrap_keep
        f = sc._f(sc.F_HEAD, 56)
        max_w = box_w - 2 * 80
        lines = _wrap_keep(headline, f, max_w, accent)
        if len(lines) > 2:
            print("splitscreen: заголовок длиннее двух строк (%d) — обрежь "
                  "фразу, карточка не пересказ реплики" % len(lines),
                  file=sys.stderr)
            lines = lines[:2]
        cx = (ax0 + ax1) // 2
        y = FOOT_SAFE - 76 * len(lines)
        # ПЛАШКА-ПОДЛОЖКА ПОД ЗАГОЛОВКОМ (та же ночь): одной мягкой растяжки
        # внизу фото мало — на светлом участке снимка (небо, борт) текст
        # терялся. Растяжка остаётся для общего затемнения низа кадра
        # (подвал должен читаться на ЛЮБОМ фото), а под самим заголовком —
        # гарантированная плашка, не на удачу светлого/тёмного кадра.
        plate_y0, plate_y1 = y - 22, y + 76 * len(lines) - 6
        im = sc._put(im, lambda d, x0=ax0 + 24, x1=ax1 - 24, y0=plate_y0,
                     y1=plate_y1: d.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 255)),
                     0.58 * reveal, 0)
        d = ImageDraw.Draw(im)
        for i, ln in enumerate(lines):
            at = 0.18 + i * 0.12
            av, dxv = sc._rv(t, at)
            if av <= 0:
                y += 76
                continue
            full_w = f.getlength(ln)
            x0ln = cx - full_w / 2
            if accent and accent in ln:
                kk = ln.index(accent)
                pre, post = ln[:kk], ln[kk + len(accent):]
                pre_w, acc_w = f.getlength(pre), f.getlength(accent)
                im = sc._put(im, lambda dr, x0ln=x0ln, pre=pre, post=post,
                             pre_w=pre_w, acc_w=acc_w, y=y: (
                    dr.text((x0ln, y), pre, font=f, fill=sc.WHITE + (255,)),
                    dr.text((x0ln + pre_w, y), accent, font=f, fill=sc.GOLD + (255,)),
                    dr.text((x0ln + pre_w + acc_w, y), post, font=f,
                            fill=sc.WHITE + (255,))), av, dxv)
            else:
                im = sc._put(im, lambda dr, x0ln=x0ln, ln=ln, y=y: dr.text(
                    (x0ln, y), ln, font=f, fill=sc.WHITE + (255,)), av, dxv)
            y += 76
        d = ImageDraw.Draw(im)

    d = ImageDraw.Draw(im)
    sc.footer(d, source, watermark)
    if out:
        im.save(out, quality=94)
        return out
    return im
