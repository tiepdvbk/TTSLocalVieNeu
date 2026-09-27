from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager

ROOT = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('TTS_STUDIO_HOME', ROOT))
for name in ('data', 'logs', 'cache/jobs', 'output'):
    (ROOT / name).mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(ROOT / 'models/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
DEFAULTS = dict(voice='', threads=4, chunk_size=220, speed=1.0, volume=100,
                gap=0.25, format='wav', output_dir=str(ROOT / 'output'), same_folder=False,
                recursive=True, preserve_tree=True, cleanup=False, temperature=0.8,
                top_k=25, top_p=0.95, repetition_penalty=1.2, retries=2, theme='dark',
                accel_mode='auto', gpu_batch=2, voice_profile=True, auto_preview=True,
                pitch=0., normalize_volume=False, trim_silence=False, use_ref_codes=True,
                pause_custom=False, pause_comma=.18, pause_sentence=.4, pause_newline=.7,
                export_srt=False, srt_mode='continuous', srt_max_speed=2.)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def load_config():
    try:
        return DEFAULTS | json.loads((ROOT / 'data/config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return DEFAULTS.copy()

def read_text(path):
    raw = Path(path).read_bytes()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        return raw.decode('utf-16')
    for enc in ('utf-8-sig', 'cp1258'):
        try:
            value = raw.decode(enc)
            if '\x00' in value:
                raise ValueError('File chứa ký tự NUL; hãy lưu lại dưới dạng UTF-8.')
            return value
        except UnicodeDecodeError:
            pass
    raise ValueError('Không đọc được mã hóa file. Hãy lưu TXT dưới dạng UTF-8.')

def split_text(text, limit=220):
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    text = ''.join(c for c in text if c in '\n\t' or unicodedata.category(c) != 'Cc')
    result = []
    for para in re.split(r'\n+', text):
        para = re.sub(r'\s+', ' ', para).strip()
        while para:
            if len(para) <= limit:
                result.append(para)
                break
            window = para[:limit + 1]
            cut = 0
            for pattern in (r'[.!?…]["”’]?\s+', r'[;:]\s+', r',\s+', r'\s+'):
                points = [m.end() for m in re.finditer(pattern, window) if m.end() <= limit]
                if points:
                    cut = points[-1]
                    if cut >= limit // 3:
                        break
            cut = cut or limit
            result.append(para[:cut].strip())
            para = para[cut:].strip()
    return [s for s in result if s]

def voices():
    import vieneu
    path = Path(vieneu.__file__).parent / 'assets/voices_v3_turbo.json'
    obj = json.loads(path.read_text(encoding='utf-8'))
    from .voice_profiles import NARRATORS
    items = [(key, value.get('description', '')) for key, value in obj['presets'].items()]
    items.sort(key=lambda item: NARRATORS.index(item[0]) if item[0] in NARRATORS else len(NARRATORS))
    from .voice_library import entries
    items.extend((key, 'Giọng cá nhân • '+name) for key,name in entries())
    return items, 'Thiện Minh' if 'Thiện Minh' in obj['presets'] else obj.get('default_voice', '')

class Store:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        (self.root / 'data').mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'data/app.db'
        with self.connect() as db:
            db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, name TEXT, source TEXT, source_hash TEXT, text TEXT,
              output TEXT, settings TEXT, fingerprint TEXT, status TEXT DEFAULT 'PENDING',
              error TEXT DEFAULT '', created REAL, updated REAL);
            CREATE TABLE IF NOT EXISTS chunks (
              job_id TEXT, idx INTEGER, text TEXT, status TEXT DEFAULT 'PENDING',
              path TEXT DEFAULT '', hash TEXT DEFAULT '', duration REAL DEFAULT 0,
              elapsed REAL DEFAULT 0, attempts INTEGER DEFAULT 0,
              PRIMARY KEY(job_id,idx), FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def recover(self):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='PAUSED' WHERE status='RUNNING'")
            db.execute("UPDATE chunks SET status='PENDING' WHERE status='RUNNING'")

    def jobs(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('''SELECT j.*, count(c.idx) total,
                sum(CASE WHEN c.status='DONE' THEN 1 ELSE 0 END) completed,
                coalesce(sum(c.duration),0) duration, coalesce(sum(c.elapsed),0) elapsed
                FROM jobs j LEFT JOIN chunks c ON j.id=c.job_id GROUP BY j.id ORDER BY j.created''')]

    def job(self, jid):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (jid,)).fetchone()
            if not row:
                raise ValueError('Tác vụ không tồn tại')
            return dict(row)

    def chunks(self, jid):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM chunks WHERE job_id=? ORDER BY idx', (jid,))]

    def status(self, jid, status, error=''):
        with self.connect() as db:
            db.execute('UPDATE jobs SET status=?,error=?,updated=? WHERE id=?', (status, error, time.time(), jid))

    def add(self, text, name, settings, source=None, relative=None, is_srt=False):
        from .speech_plan import make_plan
        settings=DEFAULTS | settings.copy()
        settings.pop('voice_data',None)
        if settings['voice'].startswith('clone:'):
            from .voice_library import load_voice
            settings['voice_data']=load_voice(settings['voice'])
        plan,cues=make_plan(text,settings, is_srt or bool(source and Path(source).suffix.lower()=='.srt'))
        pieces=[p['text'] for p in plan]
        settings['_plan']=[{k:v for k,v in p.items() if k!='text'} for p in plan]
        settings['_srt_cues']=cues
        if not pieces:
            raise ValueError('Nội dung rỗng.')
        jid = uuid.uuid4().hex
        source = str(Path(source).resolve()) if source else ''
        shash = digest(source) if source else hashlib.sha256(text.encode()).hexdigest()
        directory = Path(source).parent if source and settings['same_folder'] else Path(settings['output_dir'])
        if relative and settings['preserve_tree'] and not settings['same_folder']:
            directory /= Path(relative).parent
        directory.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', Path(name).stem).strip(' .') or 'Bai_doc'
        if re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', stem):
            stem = '_' + stem
        stem = stem[:140]
        with self.connect() as db:
            reserved = {r[0].casefold() for r in db.execute('SELECT output FROM jobs')}
            out = directory / (stem + '.' + settings['format'])
            n = 1
            while out.exists() or str(out).casefold() in reserved:
                out = directory / f'{stem}_{n:03d}.{settings["format"]}'
                n += 1
            fp = hashlib.sha256(json.dumps([shash, settings, 'v3turbo-fp32'], sort_keys=True).encode()).hexdigest()
            now = time.time()
            db.execute('INSERT INTO jobs(id,name,source,source_hash,text,output,settings,fingerprint,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (jid, name, source, shash, text, str(out), json.dumps(settings), fp, now, now))
            db.executemany('INSERT INTO chunks(job_id,idx,text) VALUES(?,?,?)', [(jid, i, s) for i, s in enumerate(pieces)])
        return jid

    def remove(self, jid):
        if self.job(jid)['status'] == 'RUNNING':
            raise ValueError('Hãy dừng tác vụ trước khi xóa.')
        with self.connect() as db:
            db.execute('DELETE FROM jobs WHERE id=?', (jid,))
        target = (self.root / 'cache/jobs' / jid).resolve()
        if target.parent == (self.root / 'cache/jobs').resolve() and target.is_dir():
            shutil.rmtree(target)

class Engine:
    def __init__(self):
        self.tts = None
        self.threads = None
        self.manager = None

    def load(self, threads):
        if self.tts is not None and threads == self.threads:
            return
        self.close()
        from vieneu.v3turbo import V3TurboVieNeuTTS
        self.tts = V3TurboVieNeuTTS(device='cpu', backend='onnx', precision='fp32', threads=threads)
        self.threads = threads

    def synthesize(self, text, settings):
        self.load(settings['threads'])
        from tools.runtime_support import voice_argument
        if settings['voice'].startswith('clone:') and not settings.get('voice_data'):
            from .voice_library import load_voice
            settings=settings | dict(voice_data=load_voice(settings['voice']))
        return self.tts.infer(text, voice=voice_argument(settings),use_ref_codes=settings.get('use_ref_codes',True),
            max_chars=settings['chunk_size'], temperature=settings['temperature'],
            top_k=settings['top_k'], top_p=settings['top_p'],
            repetition_penalty=settings['repetition_penalty'], apply_watermark=False, batch_size=1)

    def close(self):
        if self.manager is not None:
            self.manager.close()
            self.manager = None
        if self.tts:
            self.tts.close()
        self.tts = None

def valid_chunk(chunk):
    import soundfile as sf
    try:
        info = sf.info(chunk['path'])
        return info.frames > 0 and info.samplerate == 48000 and info.channels == 1 and digest(chunk['path']) == chunk['hash']
    except Exception:
        return False

def render_audio(raw, target, settings):
    """Identical tempo, volume and peak protection for previews and exports."""
    s = settings
    ffmpeg = ROOT / 'tools/ffmpeg.exe'
    pitch=2**(s.get('pitch',0)/12)
    tempo=s['speed']/pitch
    filters=[]
    if pitch!=1: filters.extend([f'asetrate={48000*pitch}', 'aresample=48000'])
    while tempo>2: filters.append('atempo=2'); tempo/=2
    while tempo<.5: filters.append('atempo=0.5'); tempo/=.5
    filters.append(f'atempo={tempo}')
    if s.get('normalize_volume'): filters.append('loudnorm=I=-20:TP=-1:LRA=11')
    filters.extend([f'volume={s["volume"]/100}', 'alimiter=limit=0.98:level=false'])
    args = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(raw),
            '-af', ','.join(filters), '-ar', '48000']
    codecs = {'wav': ['-c:a', 'pcm_s16le'], 'mp3': ['-c:a', 'libmp3lame', '-b:a', '192k'],
              'flac': ['-c:a', 'flac'], 'm4a': ['-c:a', 'aac', '-b:a', '192k']}
    result = subprocess.run(args + codecs[Path(target).suffix[1:]] + [str(target)], capture_output=True,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
    if not Path(target).exists() or Path(target).stat().st_size < 100:
        raise RuntimeError('FFmpeg không tạo được audio.')


def export_audio(store, jid, settings=None):
    import numpy as np
    import soundfile as sf
    job = store.job(jid)
    original_settings=json.loads(job['settings'])
    s = original_settings | (settings or {})
    # Segmentation and source cue timing always belong to the saved job.
    for key in ('_plan','_srt_cues'):
        if key in original_settings: s[key]=original_settings[key]
    chunks = store.chunks(jid)
    if not chunks or any(c['status'] != 'DONE' or not valid_chunk(c) for c in chunks):
        raise ValueError('Cache thiếu hoặc hỏng; hãy chọn Tiếp tục / Thử lại để tạo lại đoạn thiếu.')
    cache = store.root / 'cache/jobs' / jid
    cache.mkdir(parents=True, exist_ok=True)
    from .audio_export import build_audio
    from .speech_plan import format_srt
    processed,captions=build_audio(chunks,s,cache,render_audio)
    target = Path(job['output'])
    if settings is not None:
        target = target.with_name(target.stem + '_export.' + s['format'])
    original = target
    n = 1
    while target.exists() or (s.get('export_srt') and target.with_suffix('.srt').exists()):
        target = original.with_name(f'{original.stem}_{n:03d}{original.suffix}')
        n += 1
    tmp = cache / ('final.tmp' + target.suffix)
    if target.suffix=='.wav':
        tmp=processed
    else:
        codecs={'mp3':['libmp3lame','-b:a','192k'],'flac':['flac'],'m4a':['aac','-b:a','192k']}
        result=subprocess.run([str(ROOT/'tools/ffmpeg.exe'),'-hide_banner','-loglevel','error','-nostdin','-y',
            '-i',str(processed),'-c:a',*codecs[target.suffix[1:]],str(tmp)],capture_output=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode: raise RuntimeError(result.stderr.decode('utf-8',errors='replace'))
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write on the destination volume, then atomically publish. Windows rename
    # fails if another application creates the target in the meantime.
    staged = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
    subtitle=target.with_suffix('.srt')
    subtitle_created=False
    try:
        with open(staged, 'xb') as dest, open(tmp, 'rb') as src:
            shutil.copyfileobj(src, dest)
            dest.flush()
            os.fsync(dest.fileno())
        if s.get('export_srt'):
            # Exclusive create: never replace an input SRT or an existing sidecar.
            with open(subtitle,'x',encoding='utf-8-sig') as handle:
                subtitle_created=True
                handle.write(format_srt(captions)); handle.flush(); os.fsync(handle.fileno())
        os.rename(staged, target)
    except FileExistsError:
        if subtitle_created: subtitle.unlink(missing_ok=True)
        raise RuntimeError('Tên output vừa được chương trình khác tạo; thử xuất lại.')
    except Exception:
        if subtitle_created: subtitle.unlink(missing_ok=True)
        raise
    finally:
        staged.unlink(missing_ok=True)
    tmp.unlink(missing_ok=True)
    for name in ('merged.wav','processed.wav','cue.wav','cue-fit.wav'):
        (cache/name).unlink(missing_ok=True)
    if settings is None:
        with store.connect() as db:
            db.execute('UPDATE jobs SET output=? WHERE id=?', (str(target), jid))
    return target

class Runner:
    def __init__(self, store, engine=None, event=None):
        self.store = store
        self.engine = engine or Engine()
        self.event = event or (lambda *args: None)
        self.pause = threading.Event()
        self.stop = threading.Event()
        self.execution = {}

    def run(self, ids):
        for jid in ids:
            if self.pause.is_set() or self.stop.is_set():
                break
            try:
                self.run_job(jid)
            except Exception as e:
                logging.exception('Job %s failed', jid)
                self.store.status(jid, 'FAILED', str(e))
                self.event('log', f'Lỗi: {e}')
            self.event('refresh', jid)

    def run_job(self, jid):
        import numpy as np
        import soundfile as sf
        job = self.store.job(jid)
        s = json.loads(job['settings'])
        s.update({k:v for k,v in self.execution.items() if k in ('accel_mode','threads','gpu_batch')})
        if job['source'] and (not Path(job['source']).is_file() or digest(job['source']) != job['source_hash']):
            raise ValueError('File nguồn đã thay đổi hoặc bị xóa. Thêm file lại để tạo tác vụ mới.')
        self.store.status(jid, 'RUNNING')
        self.event('log', 'Đang xử lý: ' + job['name'])
        cache = self.store.root / 'cache/jobs' / jid
        cache.mkdir(parents=True, exist_ok=True)
        atomic_json(cache / 'manifest.json', job)
        chunks = self.store.chunks(jid)
        if s.get('accel_mode','cpu') != 'cpu' and isinstance(self.engine,Engine):
            from .acceleration import accelerated_job
            accelerated_job(self,jid,s,chunks,cache)
            if self.pause.is_set() or self.stop.is_set():
                self.store.status(jid,'STOPPED' if self.stop.is_set() else 'PAUSED')
                return
            self.event('log','Đang ghép và xuất audio…')
            output=export_audio(self.store,jid)
            self.store.status(jid,'DONE')
            self.event('log','Hoàn tất: '+str(output))
            if s['cleanup']:
                for wav in cache.glob('*.wav'):
                    wav.unlink()
            return
        for chunk in chunks:
            if self.pause.is_set() or self.stop.is_set():
                self.store.status(jid, 'STOPPED' if self.stop.is_set() else 'PAUSED')
                return
            if chunk['status'] == 'DONE' and valid_chunk(chunk):
                continue
            self.event('log', f'Đoạn {chunk["idx"]+1}/{len(chunks)} • {job["name"]}')
            self.event('refresh', jid)
            output = cache / f'{chunk["idx"]:06d}.wav'
            for attempt in range(s['retries'] + 1):
                try:
                    with self.store.connect() as db:
                        db.execute("UPDATE chunks SET status='RUNNING',attempts=attempts+1 WHERE job_id=? AND idx=?", (jid, chunk['idx']))
                    started = time.perf_counter()
                    wav = np.asarray(self.engine.synthesize(chunk['text'], s), dtype=np.float32).reshape(-1)
                    if wav.size < 4800 or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-6:
                        raise RuntimeError('Model trả về audio rỗng, quá ngắn hoặc không hợp lệ.')
                    tmp = output.with_suffix('.wav.tmp')
                    sf.write(str(tmp), wav, 48000, format='WAV', subtype='PCM_16')
                    with open(tmp, 'r+b') as handle:
                        os.fsync(handle.fileno())
                    os.replace(tmp, output)
                    elapsed = time.perf_counter() - started
                    with self.store.connect() as db:
                        db.execute("UPDATE chunks SET status='DONE',path=?,hash=?,duration=?,elapsed=? WHERE job_id=? AND idx=?",
                            (str(output), digest(output), len(wav)/48000, elapsed, jid, chunk['idx']))
                    self.event('log', f'Đã lưu đoạn {chunk["idx"]+1} • {len(wav)/48000:.1f}s audio / {elapsed:.1f}s xử lý')
                    self.event('refresh', jid)
                    break
                except Exception:
                    if attempt == s['retries']:
                        with self.store.connect() as db:
                            db.execute("UPDATE chunks SET status='FAILED' WHERE job_id=? AND idx=?", (jid, chunk['idx']))
                        raise
                    self.event('log', f'Thử lại đoạn {chunk["idx"]+1} ({attempt+1}/{s["retries"]})')
        if self.pause.is_set() or self.stop.is_set():
            self.store.status(jid, 'STOPPED' if self.stop.is_set() else 'PAUSED')
            return
        self.event('log', 'Đang ghép và xuất audio…')
        output = export_audio(self.store, jid)
        self.store.status(jid, 'DONE')
        self.event('log', 'Hoàn tất: ' + str(output))
        if s['cleanup']:
            for wav in cache.glob('*.wav'):
                wav.unlink()
