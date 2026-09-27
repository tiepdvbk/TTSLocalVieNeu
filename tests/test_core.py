import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from app.core import Store, Runner, DEFAULTS, split_text, read_text, digest, export_audio

class FakeEngine:
    def __init__(self, fail=0):
        self.calls = 0
        self.fail = fail
    def synthesize(self, text, settings):
        self.calls += 1
        if self.calls <= self.fail:
            raise RuntimeError('temporary failure')
        return (0.1 * np.sin(2 * np.pi * 440 * np.arange(12000)/48000)).astype('float32')

@pytest.fixture
def context(tmp_path):
    s = DEFAULTS | {'output_dir': str(tmp_path/'out'), 'chunk_size': 100}
    return Store(tmp_path), s

def test_chunk_integrity_and_boundaries():
    text = 'Xin chào Việt Nam. Đây là một câu kiểm tra khá dài để thử việc chia đoạn. ' * 80
    pieces = split_text(text, 100)
    assert all(0 < len(p) <= 100 for p in pieces)
    assert ''.join(''.join(p.split()) for p in pieces) == ''.join(text.split())
    assert all(p[-1] == '.' for p in pieces)
    assert len(split_text('x'*501, 100)) == 6
    assert split_text(' \n\t') == []

def test_encodings(tmp_path):
    for encoding in ['utf-8-sig', 'utf-16', 'cp1258']:
        p = tmp_path / 'in.txt'
        p.write_bytes('Xin chào'.encode(encoding))
        assert read_text(p) == 'Xin chào'

def test_unique_names_and_snapshots(context):
    store, s = context
    a = store.add('Xin chào.', 'test.txt', s)
    b = store.add('Nội dung khác.', 'test.txt', s)
    assert store.job(a)['output'] != store.job(b)['output']
    s['voice'] = 'changed'
    assert json.loads(store.job(a)['settings'])['voice'] != 'changed'

def test_resume_and_corrupt_cache(context):
    store, s = context
    jid = store.add('Câu thứ nhất.\nCâu thứ hai.\nCâu thứ ba.', 'resume', s)
    engine = FakeEngine()
    runner = Runner(store, engine)
    def event(kind, value):
        if kind == 'refresh' and any(c['status'] == 'DONE' for c in store.chunks(jid)):
            runner.pause.set()
    runner.event = event
    runner.run([jid])
    assert store.job(jid)['status'] == 'PAUSED'
    assert engine.calls == 1
    first = store.chunks(jid)[0]
    Runner(store, engine).run([jid])
    assert engine.calls == 3
    assert store.job(jid)['status'] == 'DONE'
    assert digest(first['path']) == first['hash']
    Path(first['path']).write_bytes(b'corrupt')
    Runner(store, engine).run([jid])
    assert engine.calls == 4
    assert store.job(jid)['status'] == 'DONE'

def test_source_changed(context, tmp_path):
    store, s = context
    src = tmp_path/'source.txt'
    src.write_text('Xin chào.', encoding='utf-8')
    jid = store.add(read_text(src), src.name, s, src)
    src.write_text('Đã sửa.', encoding='utf-8')
    engine = FakeEngine()
    Runner(store, engine).run([jid])
    assert store.job(jid)['status'] == 'FAILED'
    assert engine.calls == 0

def test_retry_and_recovery(context):
    store, s = context
    jid = store.add('Xin chào.', 'retry', s)
    engine = FakeEngine(fail=1)
    Runner(store, engine).run([jid])
    assert store.job(jid)['status'] == 'DONE'
    assert engine.calls == 2
    store.status(jid, 'RUNNING')
    store.recover()
    assert store.job(jid)['status'] == 'PAUSED'

@pytest.mark.parametrize('fmt', ['wav', 'mp3', 'flac', 'm4a'])
def test_output_and_postprocess(context, fmt):
    store, s = context
    s['format'] = fmt
    jid = store.add('Xin chào.', 'format', s)
    Runner(store, FakeEngine()).run([jid])
    assert store.job(jid)['status'] == 'DONE', store.job(jid)['error']
    out = Path(store.job(jid)['output'])
    assert out.stat().st_size > 100
    original = digest(out)
    second = export_audio(store, jid, s | {'speed': 1.5, 'volume': 120})
    assert second != out and second.is_file()
    assert digest(out) == original
    if fmt in ('wav', 'flac'):
        assert sf.info(second).duration < sf.info(out).duration

def test_fail_continues_next_file(context):
    store, s = context
    s['retries'] = 0
    first = store.add('Lỗi.', 'bad', s)
    second = store.add('Đọc tiếp.', 'good', s)
    Runner(store, FakeEngine(fail=1)).run([first, second])
    assert store.job(first)['status'] == 'FAILED'
    assert store.job(second)['status'] == 'DONE'
