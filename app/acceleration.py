from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import psutil
from .core import ROOT, DEFAULTS, digest, atomic_json
_SPAWN_LOCK=threading.Lock()
BENCHMARK_VERSION = 'v4-vieneu-3.8.3'


def schedule_items(items, window=512):
    """Length buckets within bounded windows; stable IDs preserve export order."""
    items = list(items)
    return [item for start in range(0, len(items), window)
            for item in sorted(items[start:start + window], key=lambda x: len(x[1]))]

def runtime_python(device: str) -> Path:
    """Prefer the relocatable runtimes shipped beside the EXE."""
    portable = ROOT / ('runtime_gpu/python.exe' if device == 'gpu' else 'runtime_cpu/python.exe')
    if portable.is_file():
        return portable
    return ROOT / ('.gpu/Scripts/python.exe' if device == 'gpu' else '.venv/Scripts/python.exe')

LABELS = {'cpu':'CPU • một tiến trình', 'cpu2':'CPU • hai tiến trình',
          'gpu':'GPU • xử lý theo nhóm', 'hybrid':'GPU + CPU', 'auto':'Tự động theo kết quả đo'}
from .voice_profiles import STORY_TEXT, profile_settings
SAMPLE_TEXT = STORY_TEXT
BENCH_TEXTS = [
 'Buổi sáng, những tia nắng đầu tiên chiếu qua khung cửa. Cô gái mở trang sách đang đọc dở, lắng nghe tiếng chim hót ngoài vườn và mỉm cười chào một ngày mới.',
 'Trên con đường nhỏ dẫn về làng, hàng cây xanh rì rào trong gió. Mọi người vẫn giữ thói quen chào hỏi nhau mỗi khi gặp mặt, dù ai cũng bận rộn với công việc riêng.',
 'Chúng ta có thể hoàn thành những việc khó khăn bằng cách chia chúng thành từng bước nhỏ. Điều quan trọng là kiên nhẫn, tập trung và dành thời gian nghỉ ngơi hợp lý.',
 'Khi màn đêm buông xuống, thành phố bắt đầu lên đèn. Dòng người trở về nhà sau một ngày làm việc, mang theo những câu chuyện giản dị để chia sẻ cùng gia đình.'
]
# Eight independent chunks let large GPU batches and a CPU helper both do useful
# work during calibration, rather than measuring mostly startup/short-call latency.
BENCH_TEXTS += [
 'Cơn mưa vừa tạnh, để lại trên những chiếc lá xanh các giọt nước trong veo. Từ căn bếp nhỏ, mùi cơm mới tỏa ra thơm dịu, gợi nhớ những bữa cơm gia đình ấm áp.',
 'Anh đặt chiếc ba lô xuống bên hiên nhà rồi nhìn quanh khu vườn quen thuộc. Sau nhiều năm xa cách, mọi thứ dường như vẫn còn đó, chỉ có những hàng cây đã cao hơn.',
 'Để đọc một cuốn sách thật hiệu quả, bạn không cần phải vội vàng. Hãy dành thời gian suy nghĩ về điều tác giả muốn nói, ghi lại những ý tưởng khiến bạn thấy thú vị.',
 'Tiếng sóng biển vỗ nhẹ vào bờ cát trong buổi chiều yên tĩnh. Những đứa trẻ chạy theo cánh diều đầy màu sắc, còn người lớn ngồi trò chuyện dưới bóng cây mát rượi.'
]

def hardware_info():
    info = dict(cpu=platform.processor(), logical=psutil.cpu_count(), physical=psutil.cpu_count(logical=False),
                ram_gb=round(psutil.virtual_memory().total/2**30,1), gpu='', vram_mb=0,
                gpu_runtime=runtime_python('gpu').is_file())
    command = shutil.which('nvidia-smi')
    if command:
        try:
            result = subprocess.run([command,'--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode == 0:
                name, memory, driver = [x.strip() for x in result.stdout.splitlines()[0].split(',')]
                info.update(gpu=name,vram_mb=int(memory),driver=driver)
        except Exception:
            pass
    info['signature'] = f'{info["cpu"]}|{info["logical"]}|{info["gpu"]}|{info["vram_mb"]}'
    return info

def recommendation(info=None):
    info = info or hardware_info()
    try:
        data = json.loads((ROOT/'data/benchmark.json').read_text(encoding='utf-8'))
        if data.get('version') == BENCHMARK_VERSION and data['hardware']['signature'] == info['signature']:
            mode, batch = data['recommended']['mode'], data['recommended']['batch']
            if mode in ('gpu','hybrid') and not (info['gpu'] and info['gpu_runtime']):
                return 'cpu2', 2
            return mode, batch
    except (OSError, ValueError, KeyError):
        pass
    if info.get('gpu') and info.get('gpu_runtime'):
        return 'gpu', 8
    return ('cpu2' if info['physical'] and info['physical'] >= 4 and psutil.virtual_memory().available > 3*2**30 else 'cpu'), 2

class ProcessEngine:
    def __init__(self, device, threads=2):
        self.device = device
        python = runtime_python(device)
        if not python.is_file():
            raise RuntimeError(f'Thiếu runtime: {python}')
        self.logfile = open(ROOT/'logs'/f'worker-{device}-{time.time_ns()}.log','w',encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONIOENCODING']='utf-8'
        env['TTS_PARENT_PID']=str(os.getpid())
        for key in ('PYTHONHOME','PYTHONPATH','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH'):
            env.pop(key,None)
        if getattr(sys,'frozen',False):
            env['PATH']=os.pathsep.join(p for p in env.get('PATH','').split(os.pathsep) if '_internal' not in p)
        # A frozen parent sets a DLL search directory inherited by Windows children.
        # Worker Python/torch must load their own runtime DLLs, not the GUI's copies.
        with _SPAWN_LOCK:
            frozen=getattr(sys,'frozen',False)
            if frozen:
                import ctypes
                ctypes.windll.kernel32.SetDllDirectoryW(None)
            try:
                self.process = subprocess.Popen([str(python),'-u',str(ROOT/'tools/engine_worker.py')],
                    cwd=ROOT, stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.logfile,
                    text=True,encoding='utf-8',bufsize=1,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            finally:
                if frozen:
                    ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)
        self.responses = queue.Queue()
        self.reader = threading.Thread(target=self._read,daemon=True)
        self.reader.start()
        try:
            self.info = self.ask(dict(op='init',device=device,threads=threads),timeout=180)['info']
        except Exception:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    continue
        except (OSError,ValueError):
            pass
        finally:
            self.responses.put({'ok':False,'error':'Tiến trình model đã đóng. Xem logs/worker-*.'})

    def ask(self, req, timeout=300):
        if self.process.poll() is not None:
            raise RuntimeError('Tiến trình model không còn chạy.')
        self.process.stdin.write(json.dumps(req,ensure_ascii=True)+'\n')
        self.process.stdin.flush()
        try:
            result = self.responses.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError('Model quá thời gian chờ 5 phút; tiến trình đã được dừng.')
        if not result.get('ok'):
            raise RuntimeError(result.get('error','Lỗi model'))
        return result

    def generate(self, texts, paths, settings):
        return self.ask(dict(op='generate',texts=texts,paths=list(map(str,paths)),settings=settings,seed=4321))

    def close(self):
        proc = getattr(self,'process',None)
        if proc and proc.poll() is None:
            try:
                proc.stdin.write('{"op":"close"}\n')
                proc.stdin.flush()
                proc.wait(timeout=3)
            except Exception:
                # Windows venv launchers may have a Python child: stop our whole tree.
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,
                               timeout=8,creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for stream in (getattr(proc,'stdin',None),getattr(proc,'stdout',None),getattr(self,'logfile',None)):
            if stream:
                stream.close()

class Manager:
    def __init__(self):
        self.lanes = []
        self.key = None
        self.mode = None
        self.batch = 2

    def configure(self, settings, event=lambda *a:None, strict=False):
        mode = settings.get('accel_mode','cpu')
        batch = int(settings.get('gpu_batch',2))
        threads=settings['threads']
        if mode == 'auto':
            mode,batch = recommendation()
            try:
                threads=json.loads((ROOT/'data/benchmark.json').read_text(encoding='utf-8'))['recommended'].get('threads',4)
            except (OSError,ValueError,KeyError):
                threads=4
        if mode in ('cpu2','hybrid') and psutil.virtual_memory().available < 2*2**30 and not self.lanes:
            if strict:
                raise RuntimeError('RAM trống thấp hơn 2 GB; bỏ qua cấu hình nhiều tiến trình.')
            event('log','RAM trống thấp; dùng một tiến trình để tránh tràn bộ nhớ.')
            mode = 'cpu'
        key = (mode,batch,threads)
        if key == self.key and all(lane.process.poll() is None for lane in self.lanes):
            return mode,batch
        self.close()
        plan = [('cpu',threads)] if mode == 'cpu' else [('cpu',threads),('cpu',threads)] if mode == 'cpu2' else [('gpu',2)] if mode == 'gpu' else [('gpu',2),('cpu',threads)]
        event('log','Đang nạp: '+LABELS[mode])
        try:
            # Load sequentially to limit peak memory during checkpoint loading.
            self.lanes = [ProcessEngine(*plan[0])]
            for spec in plan[1:]:
                self.lanes.append(ProcessEngine(*spec))
        except Exception as exc:
            self.close()
            if strict:
                raise
            event('log',f'Không chạy được {LABELS[mode]}: {exc}. Chuyển về CPU.')
            mode='cpu'
            self.lanes=[ProcessEngine('cpu',min(settings['threads'],4))]
        self.key = (mode,batch,threads)
        self.mode,self.batch = mode,batch
        return mode,batch

    def close(self):
        for lane in self.lanes:
            lane.close()
        self.lanes=[]
        self.key=None

    def stream(self, items, settings, halted=lambda:False, event=lambda *a:None, strict=False):
        """items: (stable ID,text,path). Yield each completed group for immediate DB checkpoint."""
        self.configure(settings,event,strict=strict)
        pending=list(items)
        if self.mode in ('gpu', 'hybrid') and settings.get('length_bucketing', True):
            pending = schedule_items(pending)
        with ThreadPoolExecutor(max_workers=len(self.lanes)) as pool:
            active={}
            def submit(lane):
                if not pending or halted():
                    return
                count=self.batch if lane.device=='gpu' else 1
                group=pending[:count]
                del pending[:count]
                event('log',f'{lane.device.upper()}: đang tạo nhóm {len(group)} đoạn; lưu tiến trình khi nhóm hoàn tất.')
                active[pool.submit(lane.generate,[x[1] for x in group],[x[2] for x in group],settings)]=(lane,group,time.perf_counter())
            for lane in self.lanes:
                submit(lane)
            while active:
                done,_=wait(active,timeout=0.25,return_when=FIRST_COMPLETED)
                for future in done:
                    lane,group,started=active.pop(future)
                    try:
                        result=future.result()
                    except Exception as exc:
                        if strict:
                            raise
                        event('log',f'{lane.device.upper()} gặp lỗi: {exc}. Thử lại nhóm trên CPU.')
                        lane.close()
                        index=self.lanes.index(lane)
                        lane=ProcessEngine('cpu',2)
                        self.lanes[index]=lane
                        self.key=None
                        self.mode='cpu' if all(x.device=='cpu' for x in self.lanes) else 'hybrid'
                        result=lane.generate([x[1] for x in group],[x[2] for x in group],settings)
                    elapsed = time.perf_counter()-started
                    effective = result.get('batch_limit')
                    if lane.device == 'gpu' and effective:
                        self.batch = min(self.batch, effective)
                    # The next inference can overlap durable checkpointing in the
                    # caller. On pause, drain and checkpoint every submitted group.
                    submit(lane)
                    yield group,result,elapsed

def accelerated_job(runner,jid,settings,chunks,cache):
    from .core import valid_chunk
    if not hasattr(runner.engine,'manager') or runner.engine.manager is None:
        runner.engine.manager=Manager()
    # Sequential engine is not needed while process workers are active.
    if runner.engine.tts is not None:
        runner.engine.tts.close()
        runner.engine.tts=None
    manager=runner.engine.manager
    pending=[(c['idx'],c['text'],cache/f'{c["idx"]:06d}.wav') for c in chunks
             if not (c['status']=='DONE' and valid_chunk(c))]
    if not pending:
        return
    with runner.store.connect() as db:
        db.executemany("UPDATE chunks SET status='PENDING' WHERE job_id=? AND idx=?",[(jid,x[0]) for x in pending])
    for group,result,elapsed in manager.stream(pending,settings,lambda:runner.pause.is_set() or runner.stop.is_set(),runner.event):
        updates = []
        for item,duration in zip(group,result['durations']):
            idx,_,path=item
            updates.append((str(path),digest(path),duration,elapsed/len(group),jid,idx))
        # All WAVs have been fsynced and published by the worker. Commit the
        # whole group atomically, retaining SQLite FULL durability.
        with runner.store.connect() as db:
            db.executemany("UPDATE chunks SET status='DONE',path=?,hash=?,duration=?,elapsed=?,attempts=attempts+1 WHERE job_id=? AND idx=?", updates)
            completed = db.execute("SELECT count(*) FROM chunks WHERE job_id=? AND status='DONE'", (jid,)).fetchone()[0]
        runner.event('log',f'Đã lưu {completed}/{len(chunks)} đoạn • nhóm {len(group)} • {sum(result["durations"]):.1f}s audio / {elapsed:.1f}s xử lý • {manager.mode}')
        runner.event('refresh',jid)

def preview_path(voice):
    import re
    import hashlib
    name=re.sub(r'[<>:"/\\|?*]','_',voice)
    revision=(ROOT/'vendor-revision.txt').read_text().strip() if (ROOT/'vendor-revision.txt').exists() else 'v3'
    key=hashlib.sha256(json.dumps([revision, SAMPLE_TEXT, profile_settings(voice)], sort_keys=True).encode()).hexdigest()[:12]
    return ROOT/'voice_samples'/f'{name}_{key}.wav'

def valid_preview(voice):
    path=preview_path(voice)
    try:
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        return meta['hash']==digest(path) and meta['text']==SAMPLE_TEXT and meta['profile']==profile_settings(voice)
    except (OSError, ValueError, KeyError):
        return False

def create_preview(engine,voice,force=False):
    import numpy as np
    import soundfile as sf
    from .core import split_text, render_audio
    path=preview_path(voice)
    metadata=path.with_suffix('.json')
    if not force and valid_preview(voice):
        return path
    # CPU preview preserves the current voice quality and has low single-call latency.
    settings=DEFAULTS | profile_settings(voice) | dict(voice=voice,threads=4,accel_mode='cpu')
    from tools.runtime_support import synthesis_settings
    settings=synthesis_settings(settings)
    parts=[]
    for text in split_text(SAMPLE_TEXT, settings['chunk_size']):
        if parts:
            parts.append(np.zeros(round(48000*settings['gap']), dtype=np.float32))
        parts.append(np.asarray(engine.synthesize(text,settings),dtype=np.float32))
    wav=np.concatenate(parts)
    if len(wav)<4800 or not np.isfinite(wav).all():
        raise RuntimeError('Mẫu nghe thử không hợp lệ.')
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=path.with_suffix('.raw.wav')
    tmp=path.with_suffix('.tmp.wav')
    try:
        sf.write(str(raw),wav,48000,format='WAV',subtype='PCM_16')
        render_audio(raw,tmp,settings)
        os.replace(tmp,path)
        atomic_json(metadata,dict(voice=voice,text=SAMPLE_TEXT,profile=profile_settings(voice),hash=digest(path),duration=sf.info(path).duration))
    finally:
        raw.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return path

def create_tuned_preview(engine,settings):
    import uuid
    from .core import Store, Runner
    folder=ROOT/'cache/previews'/uuid.uuid4().hex
    store=Store(folder)
    settings=settings.copy()
    text=settings.pop('_preview_text',None) or SAMPLE_TEXT
    settings=settings | dict(accel_mode='cpu', output_dir=str(folder),same_folder=False,
                             cleanup=True,export_srt=False)
    jid=store.add(text,'Mau_tuy_chinh',settings)
    Runner(store,engine).run([jid])
    job=store.job(jid)
    if job['status']!='DONE': raise RuntimeError(job['error'] or 'Không tạo được mẫu.')
    return Path(job['output'])


def benchmark(event=lambda *a:None,halted=lambda:False):
    import soundfile as sf
    info=hardware_info()
    root=ROOT/'logs'/time.strftime('benchmark-%Y%m%d-%H%M%S')
    root.mkdir(parents=True,exist_ok=True)
    # Include varied lengths and enough batches to measure graph reuse rather
    # than one uniform batch. Do not rely on eight repeated, equal-size texts.
    texts = BENCH_TEXTS + [t.split('.')[0] + '.' for t in BENCH_TEXTS] + [
        'Vâng.', 'Tôi hiểu rồi.', 'Anh dừng lại, lắng nghe tiếng gió bên ngoài.',
        'Cảm ơn bạn. Hẹn gặp lại vào ngày mai.'
    ]
    texts = texts * 4
    plans=[('cpu',1,2),('cpu',1,4),('cpu',1,6),('cpu2',1,2),('cpu2',1,4)]
    if info['gpu'] and info['gpu_runtime']:
        plans += [('gpu',b,4) for b in (2,4,8,16,24,32)] + [('hybrid',8,2)]
    results=[]
    for mode,batch,threads in plans:
        if halted():
            break
        manager=Manager()
        settings=DEFAULTS | dict(accel_mode=mode,gpu_batch=batch,voice='Ngọc Huyền',threads=threads)
        event('log',f'Đo {LABELS[mode]}, batch {batch}…')
        row=dict(mode=mode,batch=batch,threads=threads)
        try:
            begin=time.perf_counter()
            manager.configure(settings,event,strict=True)
            row['load_seconds']=time.perf_counter()-begin
            folder=root/f'{mode}-{batch}-{threads}'
            items=[(i,text,folder/f'{i}.wav') for i,text in enumerate(texts)]
            begin=time.perf_counter()
            audio=0
            for group,result,elapsed in manager.stream(items,settings,halted,event,strict=True):
                audio+=sum(result['durations'])
                row['gpu_peak_mb']=max(row.get('gpu_peak_mb',0),result.get('gpu_peak_mb',0))
            if halted():
                break
            row.update(ok=True,seconds=time.perf_counter()-begin,audio_seconds=audio)
            row['realtime']=audio/row['seconds']
            event('log',f'{LABELS[mode]} batch {batch}: {row["seconds"]:.1f}s, {row["realtime"]:.2f}× thời gian thực')
        except Exception as exc:
            row.update(ok=False,error=str(exc))
            event('log','Bỏ qua cấu hình lỗi: '+str(exc))
        finally:
            manager.close()
        results.append(row)
        atomic_json(root/'results.json',dict(hardware=info,results=results))
    good=[r for r in results if r.get('ok')]
    if not good:
        raise RuntimeError('Chưa đo được cấu hình thành công.')
    best=min(good,key=lambda x:x['seconds'])
    report=dict(version=BENCHMARK_VERSION,hardware=info,results=results,recommended=best,created=time.time(),sample_chars=sum(map(len,texts)))
    if not halted():
        atomic_json(ROOT/'data/benchmark.json',report)
    return report
