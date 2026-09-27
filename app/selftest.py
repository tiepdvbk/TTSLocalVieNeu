import json
import time
import socket
from pathlib import Path
from .core import ROOT, Store, Runner, Engine, DEFAULTS, voices, export_audio

def run():
    # Prove inference works without network, including packaged model assets.
    original_connect = socket.socket.connect
    def no_network(*args, **kwargs):
        raise RuntimeError('Network disabled for offline test')
    socket.socket.connect = no_network
    test_root = ROOT / 'logs' / time.strftime('selftest-%Y%m%d-%H%M%S')
    store = Store(test_root)
    engine = Engine()
    _, default = voices()
    settings = DEFAULTS | {'voice': default, 'output_dir': str(test_root / 'audio'), 'retries': 0, 'accel_mode':'cpu'}
    text = 'Xin chào bạn. Đây là phần mềm chuyển văn bản thành giọng nói tiếng Việt, chạy hoàn toàn trên máy tính.\nBạn có thể đọc nhiều tập tin, tạm dừng và tiếp tục công việc khi cần.'
    jid = store.add(text, 'Kiem_tra_offline', settings)
    messages = []
    def event(kind, value):
        if kind == 'log':
            messages.append(value)
    start = time.perf_counter()
    Runner(store, engine, event).run([jid])
    job = store.job(jid)
    result = {'status': job['status'], 'error': job['error'], 'wall_seconds': time.perf_counter()-start,
              'output': job['output'], 'chunks': store.chunks(jid), 'messages': messages,
              'offline_socket_blocked': True}
    if job['status'] == 'DONE':
        result['mp3'] = str(export_audio(store, jid, settings | {'format': 'mp3', 'speed': 1.1}))
        result['providers'] = engine.tts.engine.sess_dec.get_providers()
        result['voices'] = engine.tts.list_preset_voices()
    (ROOT / 'logs/selftest-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    engine.close()
    socket.socket.connect = original_connect
    if job['status'] != 'DONE':
        raise RuntimeError(job['error'])
