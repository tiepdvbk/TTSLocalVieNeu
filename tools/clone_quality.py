"""Signal-level helpers for cloning quality. Pure numpy: importable by the GUI
and by the CPU/GPU worker processes (``tools`` is on the worker's sys.path).

None of this recognises words. The checks are acoustic heuristics:
  * pick reference segments that start/end in natural pauses,
  * score generated takes (speaker similarity, speaking rate, tail, clipping),
  * repair abrupt / very quiet sentence endings with bounded gain.
"""
from __future__ import annotations
import math
import re
import numpy as np

RATE = 48000
FRAME_S = .02


def frame_rms(wav, sr, frame_s=FRAME_S):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    frame = max(1, round(frame_s * sr))
    n = len(wav) // frame
    if n == 0:
        return np.zeros(0), frame
    blocks = wav[:n * frame].reshape(n, frame).astype(np.float64)
    return np.sqrt(np.mean(blocks ** 2, axis=1)), frame


def activity(rms):
    if not len(rms):
        return np.zeros(0, dtype=bool), .003
    threshold = max(.003, float(np.percentile(rms, 90)) * .08)
    return rms > threshold, threshold


def _runs(mask):
    """[(start, end)) runs of True."""
    if not len(mask):
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


# ── Reference selection ────────────────────────────────────────────────────

def segment_score(wav, sr):
    """Higher is better. Rewards continuous, steady, unclipped speech."""
    rms, _ = frame_rms(wav, sr)
    if len(rms) < 10:
        return -9., {}
    active, _ = activity(rms)
    ratio = float(np.mean(active))
    silences = [b - a for a, b in _runs(~active)]
    inner = silences[1:-1] if len(silences) > 2 else []
    longest = max(inner, default=0) * FRAME_S
    clip = float(np.mean(np.abs(np.asarray(wav)) >= .995))
    db = 20 * np.log10(np.maximum(rms[active], 1e-6)) if active.any() else np.array([-90.])
    spread = float(np.std(db))
    level = float(np.median(db))
    seconds = len(wav) / sr
    score = (ratio + .15 * min(seconds, 8.) / 8. - .8 * max(0., longest - .5)
             - 4 * clip - .03 * max(0., spread - 7) - .02 * max(0., -34 - level))
    return score, dict(active_ratio=round(ratio, 3), longest_pause=round(longest, 2),
                       clipping=round(clip, 5), level_db=round(level, 1), seconds=round(seconds, 2))


def find_segments(wav, sr, count=3, min_s=5., max_s=8., pause_s=.16, lead_s=.08, tail_s=.22):
    """Return up to ``count`` non-overlapping (start_s, end_s, score, info) that
    begin and end inside pauses, best first. Falls back to sliding windows
    snapped to the quietest frame when the recording has no usable pauses."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    total = len(wav) / sr
    if total < 3:
        raise ValueError('Audio mẫu cần ít nhất 3 giây.')
    min_s = min(min_s, max(3., total - .01))
    rms, frame = frame_rms(wav, sr)
    active, _ = activity(rms)
    gap = max(1, round(pause_s / FRAME_S))
    starts, ends = [0.], [total]
    for a, b in _runs(~active):
        if b - a >= gap or a == 0 or b == len(active):
            ends.append(min(total, a * FRAME_S + tail_s))        # speech ended at frame a
            starts.append(max(0., b * FRAME_S - lead_s))         # speech resumes at frame b
    starts, ends = sorted(set(starts)), sorted(set(ends))
    candidates = []
    for s in starts:
        for e in ends:
            if min_s <= e - s <= max_s:
                candidates.append((s, e))
    if not candidates:
        quiet = rms / (np.max(rms) + 1e-9)
        step = .5
        for s in np.arange(0, max(total - min_s, 0) + 1e-6, step):
            lo, hi = s + min_s, min(total, s + max_s)
            a, b = int(lo / FRAME_S), max(int(lo / FRAME_S) + 1, int(hi / FRAME_S))
            e = (a + int(np.argmin(quiet[a:b]))) * FRAME_S if b <= len(quiet) and b > a else hi
            candidates.append((float(s), float(min(total, e + .02))))
    scored = []
    for s, e in candidates:
        seg = wav[int(s * sr):int(e * sr)]
        score, info = segment_score(seg, sr)
        scored.append((score, s, e, info))
    scored.sort(key=lambda x: -x[0])
    chosen = []
    for score, s, e, info in scored:
        if all(e <= cs or s >= ce for cs, ce, _, _ in chosen):
            chosen.append((round(s, 3), round(e, 3), round(score, 4), info))
        if len(chosen) >= count:
            break
    return chosen


def average_embedding(embeddings, weights=None):
    """L2-normalised weighted mean, rescaled to the mean input norm."""
    embs = [np.asarray(e, dtype=np.float64).reshape(-1) for e in embeddings]
    if len(embs) == 1:
        return embs[0].astype(np.float32)
    weights = np.ones(len(embs)) if weights is None else np.asarray(weights, dtype=np.float64)
    norms = np.array([np.linalg.norm(e) + 1e-9 for e in embs])
    mean = sum(w * e / n for w, e, n in zip(weights, embs, norms)) / weights.sum()
    return (mean / (np.linalg.norm(mean) + 1e-9) * norms.mean()).astype(np.float32)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── Take scoring ────────────────────────────────────────────────────────────

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def syllables(text):
    """Vietnamese is written one syllable per word; digits read longer."""
    count = 0
    for word in _WORD.findall(re.sub(r'\[[^\]]*\]', ' ', text or '')):
        count += max(1, math.ceil(len(word) / 1.5)) if word.isdigit() else 1
    return count


def take_metrics(wav, sr, text):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    rms, frame = frame_rms(wav, sr, .01)
    m = dict(seconds=len(wav) / sr, syllables=syllables(text), abrupt=False, end_ratio=1.,
             longest_pause=0., clipping=float(np.mean(np.abs(wav) >= .995)) if len(wav) else 0.,
             rate=None, speech_seconds=0.)
    if len(rms) < 5:
        m['empty'] = True
        return m
    active, _ = activity(rms)
    if not active.any():
        m['empty'] = True
        return m
    idx = np.flatnonzero(active)
    first, last = idx[0], idx[-1]
    speech = float(np.median(rms[active]))
    m['speech_seconds'] = (last - first + 1) * .01
    m['rate'] = m['syllables'] / max(m['speech_seconds'], .2)
    m['abrupt'] = bool(np.mean(rms[-2:]) > .35 * speech)
    tail = rms[max(first, last - 40):last + 1]              # final ~0.4 s of speech
    loud = tail[tail >= np.percentile(tail, 40)] if len(tail) > 3 else tail
    m['end_ratio'] = float(np.mean(loud) / (speech + 1e-9)) if len(loud) else 1.
    inner = [(b - a) for a, b in _runs(~active[first:last + 1])]
    m['longest_pause'] = max(inner, default=0) * .01
    return m


def take_score(m, similarity=None, expected_rate=None):
    """-> (score, flags). Flags mark takes worth regenerating."""
    flags = []
    if m.get('empty'):
        return -9., ['empty']
    score = similarity if similarity is not None else .5
    if m['rate'] and m['syllables'] >= 5:
        expected = expected_rate or 4.8
        ratio = m['rate'] / expected
        deviation = abs(math.log(ratio))
        score -= 1.2 * max(0., deviation - .22)
        if ratio > 1.75:
            flags.append('too_fast')          # likely skipped words
        elif ratio < .55:
            flags.append('too_slow')          # likely looped / babbled
    if m['longest_pause'] > 1.2:
        score -= .3
        flags.append('long_pause')
    if m['abrupt']:
        score -= .2
        flags.append('abrupt_end')
    if m['end_ratio'] < .22 and m['syllables'] >= 3:
        score -= .12
        flags.append('quiet_end')
    if m['clipping'] > .001:
        score -= .3
        flags.append('clipping')
    if similarity is not None and similarity < .45:
        flags.append('voice_drift')
    return score, flags


SEVERE = {'empty', 'too_fast', 'too_slow', 'long_pause', 'voice_drift', 'clipping'}


def expected_rate(metrics):
    rates = [m['rate'] for m in metrics if m.get('rate') and m['syllables'] >= 5]
    return float(np.median(rates)) if len(rates) >= 3 else None


# ── Tail repair ─────────────────────────────────────────────────────────────

def finish_chunk(wav, sr=RATE, lift=True, max_gain_db=6.):
    """Soft edges for every chunk; lift a very quiet final word (bounded)."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1).copy()
    if len(wav) < sr // 10:
        return wav
    fade_in = max(1, round(.003 * sr))
    wav[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    rms, frame = frame_rms(wav, sr, .01)
    active, _ = activity(rms)
    if active.any():
        idx = np.flatnonzero(active)
        first, last = idx[0], idx[-1]
        speech = float(np.median(rms[active]))
        if lift and last - first > 60:
            start = max(first, last - 45)
            loud = rms[start:last + 1]
            loud = loud[loud >= np.percentile(loud, 40)]
            level = float(np.mean(loud)) if len(loud) else speech
            if level < .32 * speech:
                gain = min(10 ** (max_gain_db / 20), .55 * speech / max(level, 1e-6))
                ramp_from = max(0, (start - 20) * frame)
                ramp_to = start * frame
                curve = np.ones(len(wav), dtype=np.float32)
                curve[ramp_from:ramp_to] = np.linspace(1, gain, ramp_to - ramp_from, dtype=np.float32)
                curve[ramp_to:] = gain
                wav *= curve
        if np.mean(rms[-2:]) > .35 * speech:
            fade = min(len(wav), round(.05 * sr))
            wav[-fade:] *= (.5 + .5 * np.cos(np.linspace(0, np.pi, fade))).astype(np.float32)
            wav = np.concatenate([wav, np.zeros(round(.12 * sr), dtype=np.float32)])
    fade_out = max(1, round(.008 * sr))
    wav[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.
    if peak > .98:
        wav *= .98 / peak
    return wav


# ── Candidate selection ────────────────────────────────────────────────────

def select_takes(texts, generate, mode='auto', count=3, embed=None, reference=None,
                 tail_fix=True, sr=RATE, log=None):
    """generate(list[str]) -> list[wav]. Returns best (finished) wav per text.

    mode 'off'  : one take, edge repair only.
    mode 'auto' : regenerate only takes with severe flags, up to ``count`` takes.
    mode 'best' : always ``count`` takes, keep the highest score.
    """
    texts = list(texts)
    count = max(1, int(count))
    wavs = [np.asarray(w, dtype=np.float32).reshape(-1) for w in generate(texts)]
    if mode == 'off' or count == 1:
        return [finish_chunk(w, sr, tail_fix) for w in wavs]

    def similarity(w):
        if embed is None or reference is None or len(w) < sr:
            return None
        try:
            return cosine(embed(w), reference)
        except Exception:
            return None

    metrics = [take_metrics(w, sr, t) for w, t in zip(wavs, texts)]
    rate = expected_rate(metrics)
    best = []
    for w, m in zip(wavs, metrics):
        score, flags = take_score(m, similarity(w), rate)
        best.append(dict(wav=w, score=score, flags=flags, takes=1))
    for _ in range(count - 1):
        todo = [i for i, b in enumerate(best)
                if mode == 'best' or (SEVERE & set(b['flags']))]
        if not todo:
            break
        again = generate([texts[i] for i in todo])
        for i, w in zip(todo, again):
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            score, flags = take_score(take_metrics(w, sr, texts[i]), similarity(w), rate)
            best[i]['takes'] += 1
            if score > best[i]['score']:
                best[i].update(wav=w, score=score, flags=flags)
    if log:
        redone = sum(b['takes'] > 1 for b in best)
        if redone:
            left = sum(bool(SEVERE & set(b['flags'])) for b in best)
            log(f'Chọn bản tốt nhất: sinh lại {redone}/{len(best)} đoạn; còn {left} đoạn nên nghe lại.')
    return [finish_chunk(b['wav'], sr, tail_fix) for b in best]
