# -*- coding: utf-8 -*-
"""СПЛИТ-ЭКРАН «ЗА/ПРОТИВ» — канал МАТРИЦА ВЛАСТИ (25.11.2026).

Решение 25.11 (см. МАТРИЦА-ВЛАСТИ-СОСТОЯНИЕ-ЧИТАТЬ-ПЕРВЫМ.md): карточка
`duel` (статичный текст сверху/снизу) отклонена категорически — «убожество,
таскаешь одну хрень по всему видеоряду». Нужен настоящий ДИНАМИЧЕСКИЙ
сплит-экран: две стороны side-by-side ВЕСЬ РОЛИК, подсветка того, кто
говорит сейчас, аудио-полоса по РЕАЛЬНОЙ амплитуде голоса (не на глаз),
камера в медленном движении, переход между репликами — не хард-кат, а
подсветка, которая переезжает на другую сторону.

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

# цвет СТОРОНЫ закреплён навечно, как и цвет говорящего у `duel` —
# «цвет имени не меняется весь ролик», то же правило, просто сторон
# ровно две и они известны заранее (не N говорящих).
SIDE_COLOR = {"za": (214, 58, 52), "protiv": (86, 170, 232)}
SIDE_LABEL = {"za": "ЗА", "protiv": "ПРОТИВ"}

_ENV_CACHE = {}


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


def _step_zoom(t, dur, amount=0.045, steps=3):
    """Тот же профиль, что `render_anim` держит в хвосте карточек — ступени
    по smootherstep, не ползучий линейный зум (правило 21.10, памятка
    «зум медленными ступенями»). Здесь он ведёт камеру ВСЮ сцену, а не
    только последний кусок: вид обязан двигаться с первого кадра."""
    if dur <= 0:
        return 1.0
    p = max(0.0, min(1.0, t / dur))
    q = p * steps
    k = min(steps - 1, int(q))
    frac = q - k
    rise = 0.8
    local = min(1.0, frac / rise) if rise > 0 else 1.0
    s = local * local * local * (local * (local * 6 - 15) + 10)   # smootherstep
    done = (k + s) / steps
    return 1.0 + amount * done


def _zoom_bg(im, scale):
    if scale <= 1.0:
        return im
    w, h = im.size
    nw, nh = int(round(w * scale)), int(round(h * scale))
    big = im.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return big.crop((x0, y0, x0 + w, y0 + h))


def _wave(d, cx, cy, w, color, amp, t, grow, bars=26):
    """Эквалайзер: столбики зеркально вверх-вниз от центральной линии — тот
    же вид, что на кадре-образце («Верица Обренович», архив пророчеств).
    Каждый столбик берёт СВОЮ фазу — иначе это не бегущая волна, а один
    пульсирующий блок."""
    bw = w / bars
    for i in range(bars):
        ph = (i - bars / 2.0) * 0.17
        local = max(0.0, min(1.0, amp + 0.24 * np.sin(5.4 * t + ph)))
        hgt = (5 + local * 44) * grow
        if hgt < 0.5:
            continue
        x = cx - w / 2 + i * bw + bw * 0.16
        bww = bw * 0.68
        d.rectangle((x, cy - hgt, x + bww, cy + hgt), fill=color)


def dialogue_split(out=None, kick="", za_name="ЗА", protiv_name="ПРОТИВ",
                    active="za", text="", accent=None, audio="", seg_dur=None,
                    source="", watermark="МАТРИЦА ВЛАСТИ", dur=6.0,
                    t=None, sfx=None):
    """Один клип = одна реплика одной стороны.

    active     — "za" | "protiv": кто говорит в ЭТОМ клипе;
    text       — реплика активной стороны (у неё и только у неё есть текст
                 и аудио-полоса в этом клипе — молчащая сторона держит
                 только плашку с именем, притушенную);
    audio      — путь к ОТДЕЛЬНОМУ аудио-файлу ИМЕННО этой реплики (не
                 всего ролика) — огибающая считается от него, без смещений;
    dur        — длительность сцены в секундах, нужна для зума камеры.
    """
    sc._ev(sfx, 0.0, "card")
    im = sc.canvas().convert("RGB")

    bgscale = _step_zoom(0.0 if t is None else t, dur)
    im = _zoom_bg(im, bgscale)

    inactive = "protiv" if active == "za" else "za"
    col = {active: SIDE_COLOR[active], inactive: SIDE_COLOR[inactive]}
    names = {"za": za_name, "protiv": protiv_name}
    halves = {"za": (0, MID_X), "protiv": (MID_X, W)}

    # ВСПЫШКА ПОДСВЕТКИ НА СТЫКЕ РЕПЛИК (решение 25.11): не монтажная
    # склейка, а подсветка, которая «переезжает» — каждый клип начинает со
    # своей вспышки, поэтому на склейке клипов это читается как переход.
    reveal, _ = sc._rv(t, 0.0)

    # половины экрана: тёмный фон у молчащей стороны, мягкая тонировка у
    # говорящей — ровно «активная сторона подсвечена ярче». ОБЯЗАТЕЛЬНО
    # через `_put` (настоящий альфа-слой): прямой `ImageDraw` на RGB-кадре
    # альфу молча игнорирует и красит НАСМЕРТЬ — поймано на первом пробном
    # кадре (вместо лёгкого подсвета вышла сплошная заливка).
    for side, (x0, x1) in halves.items():
        if side == active:
            im = sc._put(im, lambda d, x0=x0, x1=x1, col=col[side]: d.rectangle(
                (x0, 0, x1, H), fill=col + (255,)), 0.16 * reveal, 0)
        else:
            im = sc._put(im, lambda d, x0=x0, x1=x1: d.rectangle(
                (x0, 0, x1, H), fill=(0, 0, 0, 255)), 0.42 * reveal, 0)

    d = ImageDraw.Draw(im)
    d.line([(MID_X, 56), (MID_X, H - 120)], fill=(66, 66, 70), width=2)

    if kick:
        sc.kicker(d, 70, kick, dot=True, t=t)

    # --- плашки имён, по центру своей половины ------------------------------
    f_name = sc._f(sc.F_ALT, 46)
    for side, (x0, x1) in halves.items():
        cx = (x0 + x1) // 2
        label = names[side].upper()
        wtxt = sc.track_w(label, f_name, 4)
        c = col[side] if side == active else sc.GREY_DIM
        y_label = 150
        sc.track(d, (cx - wtxt // 2, y_label), label, f_name, c, 4)
        # черта под именем — отступ от РЕАЛЬНОГО низа букв (bbox), а не на
        # глаз: на пробном кадре черта на глазок легла прямо по буквам,
        # вид читался как зачёркнутое имя.
        bottom = y_label + f_name.getbbox(label)[3]
        bar_w = max(40, wtxt * 0.5)
        d.rectangle((cx - bar_w / 2, bottom + 14, cx + bar_w / 2, bottom + 17),
                    fill=c)

    # --- аудио-полоса под именем активной стороны, по РЕАЛЬНОЙ амплитуде ---
    env, rate = _envelope(audio) if audio else (None, 50)
    tt = 0.0 if t is None else t
    amp = _amp_at(env, rate, tt) if env is not None else 0.55
    ax0, ax1 = halves[active]
    _wave(d, (ax0 + ax1) // 2, 278, 560, col[active], amp, tt, reveal)

    # --- реплика активной стороны, по центру своей половины -----------------
    if text:
        from quote_variants import _wrap_keep
        f = sc._f(sc.F_HEAD, 50)
        max_w = MID_X - 2 * 90
        lines = _wrap_keep(text, f, max_w, accent)
        cx = (ax0 + ax1) // 2
        y = 400
        for i, ln in enumerate(lines):
            at = 0.16 + i * 0.13
            av, dxv = sc._rv(t, at)
            if av <= 0:
                y += 70
                continue
            full_w = f.getlength(ln)
            x0ln = cx - full_w / 2
            if accent and accent in ln:
                k = ln.index(accent)
                pre, post = ln[:k], ln[k + len(accent):]
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
            y += 70
        d = ImageDraw.Draw(im)

    d = ImageDraw.Draw(im)
    sc.footer(d, source, watermark)
    if out:
        im.save(out, quality=94)
        return out
    return im
