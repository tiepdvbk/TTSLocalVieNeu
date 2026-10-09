"""Clone take selection, tail repair, reference search and reading helpers."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf
from tools import clone_quality as cq
from tools.runtime_support import synthesize_texts, speakable, synthesis_settings
from app.core import (DEFAULTS, Store, Runner, read_text, split_text, apply_pronunciations,
                      load_pronunciations, export_audio, soft_limit, valid_chunk)
from tests.test_core import FakeEngine

SR = 48000


def speech(seconds, syllables_per_s=5., amp=.2, seed=0):
    """Syllable-like bursts: 120 ms tone + 80 ms gap per syllable."""
    rng = np.random.default_rng(seed)
    n = round(seconds * SR)
    t = np.arange(n) / SR
    env = ((t * syllables_per_s) % 1) < .6
    return (amp * env * np.sin(2 * np.pi * (180 + 20 * rng.random()) * t)).astype('float32')


def test_find_segments_snaps_to_pauses():
    parts = [np.zeros(SR // 2)]
    for i in range(5):
        parts += [speech(3.2, seed=i), np.zeros(int(SR * .4))]
    wav = np.concatenate(parts).astype('float32')
    segments = cq.find_segments(wav, SR, count=3)
    assert segments and len(segments) <= 3
    for start, end, score, info in segments:
        assert 5 <= end - start <= 8
        # both edges fall inside silence
        for edge in (start, end):
            i = min(len(wav) - 1, int(edge * SR))
            assert np.max(np.abs(wav[max(0, i - 240):i + 240])) < .05
    spans = sorted((s, e) for s, e, _, _ in segments)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_find_segments_without_pauses_and_too_short():
    segs = cq.find_segments(speech(20), SR, count=2)
    assert segs and all(3 <= e - s <= 8 for s, e, _, _ in segs)
    with pytest.raises(ValueError):
        cq.find_segments(speech(2), SR)


def test_average_embedding_keeps_scale_and_direction():
    a, b = np.ones(4), np.array([1, 1, 1, 1.2])
    m = cq.average_embedding([a * 3, b * 3])
    assert np.linalg.norm(m) == pytest.approx(np.mean([np.linalg.norm(a * 3), np.linalg.norm(b * 3)]), rel=1e-5)
    assert cq.cosine(m, a) > .99


def test_take_metrics_and_flags():
    text = ' '.join(['một'] * 20)
    good = np.concatenate([speech(4), np.zeros(SR // 4)])
    m = cq.take_metrics(good, SR, text)
    assert 4 <= m['rate'] <= 6 and not m['abrupt']
    assert cq.take_score(m)[1] == []
    fast = cq.take_metrics(np.concatenate([speech(1.2), np.zeros(SR // 4)]), SR, text)
    assert 'too_fast' in cq.take_score(fast)[1]
    slow = cq.take_metrics(np.concatenate([speech(9), np.zeros(SR // 4)]), SR, text)
    assert 'too_slow' in cq.take_score(slow)[1]
    cut = cq.take_metrics(np.full(SR * 4, .2, dtype='float32') * np.sin(np.arange(SR * 4) / 8), SR, text)
    assert 'abrupt_end' in cq.take_score(cut)[1]
    assert 'voice_drift' in cq.take_score(m, similarity=.2)[1]


def test_finish_chunk_fades_abrupt_end_and_lifts_quiet_word():
    wav = (.3 * np.sin(np.arange(SR * 2) * 2 * np.pi * 200 / SR)).astype('float32')
    out = cq.finish_chunk(wav)
    assert len(out) > len(wav) and abs(out[-1]) < 1e-3 and abs(out[0]) < 1e-3
    quiet = wav.copy()
    quiet[-int(SR * .45):] *= .1
    quiet = np.concatenate([quiet, np.zeros(SR // 5, dtype='float32')])
    lifted = cq.finish_chunk(quiet)
    end = slice(len(wav) - int(SR * .3), len(wav))
    ratio = np.sqrt(np.mean(lifted[end] ** 2)) / np.sqrt(np.mean(quiet[end] ** 2))
    assert 1.5 < ratio <= 2.01            # bounded +6 dB
    assert np.max(np.abs(lifted)) <= .98
    assert np.allclose(cq.finish_chunk(quiet, lift=False)[SR // 100:SR], quiet[SR // 100:SR], atol=1e-3)


def test_select_takes_regenerates_only_flagged_and_keeps_best():
    text = ' '.join(['chữ'] * 20)
    takes = {'a': [np.concatenate([speech(1.0), np.zeros(SR // 4)]), np.concatenate([speech(4), np.zeros(SR // 4)])]}
    calls = []
    def generate(texts):
        calls.append(list(texts))
        return [takes['a'].pop(0) if t == text and takes['a'] else np.concatenate([speech(4), np.zeros(SR // 4)])
                for t in texts]
    other = ' '.join(['câu'] * 20)
    out = cq.select_takes([text, other, other, other], generate, mode='auto', count=3)
    assert len(out) == 4 and calls[1] == [text] and len(calls) == 2
    m = cq.take_metrics(out[0], SR, text)
    assert 4 <= m['rate'] <= 6
    calls.clear()
    cq.select_takes([other], generate, mode='best', count=3)
    assert len(calls) == 3
    calls.clear()
    cq.select_takes([other], generate, mode='off', count=3)
    assert len(calls) == 1


def test_similarity_drives_choice():
    text = ' '.join(['giọng'] * 20)
    order = iter([1, 2, 3])
    def generate(texts):
        k = next(order)
        w = np.concatenate([speech(4, seed=k), np.zeros(SR // 4)])
        w[0] = k / 1000                     # tag the take
        return [w]
    embed = lambda w: np.array([1., 0.]) if abs(w[0] - .002) < 1e-4 else np.array([0., 1.])
    out = cq.select_takes([text], generate, mode='best', count=3, embed=embed,
                          reference=np.array([1., 0.]), tail_fix=False)
    assert abs(out[0][0]) < .01            # faded-in first sample of take 2 kept
    assert cq.cosine(embed(np.array([.002])), [1, 0]) == pytest.approx(1)


def test_speakable_and_legacy_passthrough():
    assert speakable('Anh bước vào nhà') == 'Anh bước vào nhà,'
    assert speakable('Anh bước vào nhà', 'newline') == 'Anh bước vào nhà'
    assert speakable('Xong rồi.') == 'Xong rồi.'
    assert speakable('“Đi thôi”') == '“Đi thôi”'
    seen = []
    gen = lambda texts: seen.extend(texts) or [np.zeros(10)] * len(texts)
    assert len(synthesize_texts(None, ['a b'], {}, gen)) == 1 and seen == ['a b']
    from app.core import spoken_text
    settings = dict(take_select='off', _plan=[dict(boundary='chunk'), dict(boundary='newline')])
    assert spoken_text(dict(idx=0, text='Một đoạn bị cắt'), settings) == 'Một đoạn bị cắt,'
    assert spoken_text(dict(idx=1, text='Hết đoạn'), settings) == 'Hết đoạn'
    assert spoken_text(dict(idx=0, text='Bản cũ'), {}) == 'Bản cũ'


def test_mojibake_rejected_but_cp1258_ok(tmp_path):
    p = tmp_path / 'a.txt'
    p.write_bytes('Ngày xưa có nàng công chúa.'.encode('cp1258'))
    assert read_text(p).startswith('Ngày')
    p.write_bytes('Ngày xưa có một nàng công chúa.'.encode('utf-8').decode('cp1252').encode('utf-8'))
    with pytest.raises(ValueError, match='mã hóa'):
        read_text(p)
    p.write_text('NGÃ BA ĐỒNG LỘC', encoding='utf-8')
    assert read_text(p) == 'NGÃ BA ĐỒNG LỘC'


def test_abbreviations_are_not_sentence_ends():
    text = 'Ông làm việc tại TP. Hồ Chí Minh cùng TS. Nguyễn Văn An và nhiều người khác nữa. ' * 3
    pieces = split_text(text, 60)
    assert not any(p.endswith(('TP.', 'TS.')) for p in pieces)
    assert ''.join(''.join(p.split()) for p in pieces) == ''.join(text.split())


def test_pronunciation_dictionary(tmp_path):
    p = tmp_path / 'dict.txt'
    p.write_text('# ghi chú\nTP.HCM = thành phố Hồ Chí Minh\nAI = ây ai\nOK=ô kê\nsai dòng\n', encoding='utf-8')
    pairs = load_pronunciations(p)
    assert pairs[0][0] == 'TP.HCM' and len(pairs) == 3
    out = apply_pronunciations('Về TP.HCM, ok chưa? MAIL AI.', pairs)
    assert out == 'Về thành phố Hồ Chí Minh, ô kê chưa? MAIL ây ai.'
    store = Store(tmp_path)
    jid = store.add('Đi TP.HCM nhé.', 'x', DEFAULTS | dict(output_dir=str(tmp_path / 'o'), _pronunciations=pairs))
    assert 'thành phố Hồ Chí Minh' in store.chunks(jid)[0]['text']


def test_cleanup_marks_cache_and_reexport_explains(tmp_path):
    store = Store(tmp_path)
    s = DEFAULTS | dict(output_dir=str(tmp_path / 'out'), cleanup=True, take_select='off')
    jid = store.add('Một câu ngắn để thử.', 'a', s)
    Runner(store, FakeEngine()).run([jid])
    assert store.job(jid)['status'] == 'DONE' and not store.cache_available(jid)
    assert store.jobs()[0]['completed'] == store.jobs()[0]['total']
    with pytest.raises(ValueError, match='Xóa cache khi hoàn tất'):
        export_audio(store, jid, dict(volume=80))
    with pytest.raises(ValueError, match='dọn'):
        store.reset_chunks(jid, [0])


def test_reset_chunk_regenerates_and_progress(tmp_path):
    store = Store(tmp_path)
    s = DEFAULTS | dict(output_dir=str(tmp_path / 'out'), chunk_size=40)
    jid = store.add('Câu thứ nhất khá dài. Câu thứ hai cũng dài. Câu thứ ba.', 'a', s)
    engine = FakeEngine()
    Runner(store, engine).run([jid])
    calls = engine.calls
    store.reset_chunks(jid, [1])
    assert store.job(jid)['status'] == 'PAUSED'
    p = store.progress([jid])[jid]
    assert p['left_chars'] == len(store.chunks(jid)[1]['text']) and p['done_chars'] > 0
    Runner(store, engine).run([jid])
    assert engine.calls == calls + 1 and store.job(jid)['status'] == 'DONE'
    assert store.speed_hint() is not None
    assert all(valid_chunk(c, deep=True) for c in store.chunks(jid))


def test_jobs_omits_text_and_caches_done_totals(tmp_path):
    store = Store(tmp_path)
    s = DEFAULTS | dict(output_dir=str(tmp_path / 'out'))
    jid = store.add('Xin chào.', 'a', s)
    assert 'text' not in store.jobs()[0] and store.jobs()[0]['total'] == 1
    Runner(store, FakeEngine()).run([jid])
    first = store.jobs()[0]
    assert first['completed'] == 1 and first['duration'] > 0


def test_soft_limit_and_fast_render(tmp_path):
    wav = np.concatenate([np.full(4800, .5), np.full(4800, 1.4), np.full(4800, .5)]).astype('float32')
    out = soft_limit(wav)
    assert np.max(np.abs(out)) <= .98 and out[1000] == pytest.approx(.5, abs=.01)
    from app.core import render_audio
    raw, target = tmp_path / 'r.wav', tmp_path / 't.wav'
    sf.write(raw, np.full(4800, .4, dtype='float32'), SR)
    render_audio(raw, target, dict(speed=1., volume=50, pitch=0))
    assert sf.read(target)[0][100] == pytest.approx(.2, abs=.01)


def test_vietnamese_order_and_voice_groups(tmp_path, monkeypatch):
    from app import voice_library as v
    names = ['Đức Trí', 'Ngọc Linh', 'Adam', 'Ân', 'Ba', 'Ăn', 'Hải Đăng', 'Dũng', 'Âu', 'Bình', 'Điệp', 'Em']
    order = sorted(['Hà', 'Há', 'Hạ', 'Ha', 'Hả', 'Hã'], key=v.vi_sort_key)
    assert order == ['Ha', 'Hà', 'Hả', 'Hã', 'Há', 'Hạ']
    assert sorted(['Điệp', 'Dũng', 'Em', 'Đức', 'Dung'], key=v.vi_sort_key) == ['Dung', 'Dũng', 'Điệp', 'Đức', 'Em']
    assert sorted(['Bình', 'Ân', 'Ăn', 'Adam', 'Âu'], key=v.vi_sort_key) == ['Adam', 'Ăn', 'Ân', 'Âu', 'Bình']
    monkeypatch.setattr(v, 'LIBRARY_DIR', tmp_path / 'voices')
    v.LIBRARY_DIR.mkdir()
    keys = {name: 'clone:' + c * 32 for name, c in (('Zed', 'a'), ('Ánh', 'b'), ('Bố', 'c'))}
    for name, key in keys.items():
        v.voice_file(key).write_text(json.dumps(dict(id=key, name=name)), encoding='utf-8')
    items = [('Thiện Minh', ''), (keys['Zed'], ''), ('Adam', ''), (keys['Ánh'], ''), ('Đức Trí', ''), (keys['Bố'], ''), ('Mỹ Duyên', '')]
    result = [k for k, _ in v.ordered_voices(items)]
    assert result == [keys['Ánh'], keys['Bố'], keys['Zed'], 'Adam', 'Đức Trí', 'Mỹ Duyên', 'Thiện Minh']


def test_expressive_preset_snapshot():
    s = synthesis_settings(DEFAULTS | dict(voice='clone:' + 'a' * 32, take_select='off'))
    assert s['take_select'] == 'off'      # explicit choice respected
    assert synthesis_settings(DEFAULTS | dict(voice='Thiện Minh'))['temperature'] == DEFAULTS['temperature']
