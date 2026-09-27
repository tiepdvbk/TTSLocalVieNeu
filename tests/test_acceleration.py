import json
import numpy as np
import soundfile as sf
from app import acceleration as a
from app.core import Store, Engine, Runner, DEFAULTS, digest
from tests.test_core import FakeEngine

def test_preview_is_local_and_cached(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    engine=FakeEngine()
    path=a.create_preview(engine,'Giọng thử')
    assert path.parent==tmp_path/'voice_samples'
    assert sf.info(path).samplerate==48000
    assert engine.calls==2
    assert a.create_preview(engine,'Giọng thử')==path
    assert engine.calls==2
    path.write_bytes(b'broken')
    a.create_preview(engine,'Giọng thử')
    assert engine.calls==4
    a.create_preview(engine,'Giọng thử',force=True)
    assert engine.calls==6

def test_auto_requires_matching_hardware(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    (tmp_path/'data').mkdir()
    report={'version':a.BENCHMARK_VERSION,'hardware':{'signature':'current'},'recommended':{'mode':'gpu','batch':4}}
    (tmp_path/'data/benchmark.json').write_text(json.dumps(report))
    hardware={'signature':'current','gpu':'GTX','gpu_runtime':True,'physical':6}
    assert a.recommendation(hardware)==('gpu',4)
    assert a.recommendation(hardware|{'gpu_runtime':False})==('cpu2',2)
    assert a.recommendation(hardware|{'signature':'different'}) == ('gpu',8)

class FakeManager:
    mode='cpu2'
    def __init__(self):
        self.calls=[]
    def stream(self,items,settings,halted,event):
        # Deliberately finish later chunks before earlier ones.
        for item in reversed(items):
            if halted():
                return
            self.calls.append(item[0])
            sf.write(str(item[2]),np.ones(12000,dtype=np.float32)*0.1,48000)
            yield [item],{'durations':[0.25]},0.1
    def close(self):
        pass


def test_length_scheduler_preserves_ids_and_bounds_lookahead():
    items = [(i, 'x' * n, str(i)) for i, n in enumerate([100, 2, 80, 5, 9, 1])]
    ordered = a.schedule_items(items, window=4)
    assert [x[0] for x in ordered] == [1, 3, 2, 0, 5, 4]
    assert sorted(ordered) == sorted(items)


def test_pause_drains_prefetched_group(monkeypatch):
    class Stub:
        device = 'gpu'
        def generate(self, texts, paths, settings):
            return {'durations': [1] * len(texts)}
    manager = a.Manager()
    manager.lanes = [Stub()]
    manager.mode = 'gpu'
    manager.batch = 2
    monkeypatch.setattr(manager, 'configure', lambda *a, **k: None)
    paused = False
    completed = []
    for group, result, elapsed in manager.stream([(i, str(i), str(i)) for i in range(7)],
                                                DEFAULTS, halted=lambda: paused):
        completed.extend(x[0] for x in group)
        paused = True
    assert completed == [0, 1, 2, 3]

def test_parallel_checkpoint_pause_resume_and_merge_order(tmp_path):
    store=Store(tmp_path)
    s=DEFAULTS|{'output_dir':str(tmp_path/'out'),'accel_mode':'cpu2'}
    jid=store.add('Câu một.\nCâu hai.\nCâu ba.','parallel',s)
    engine=Engine()
    engine.manager=FakeManager()
    runner=Runner(store,engine)
    def event(kind,value):
        if kind=='refresh' and any(c['status']=='DONE' for c in store.chunks(jid)):
            runner.pause.set()
    runner.event=event
    runner.run([jid])
    assert store.job(jid)['status']=='PAUSED'
    assert engine.manager.calls==[2]
    saved=digest(store.chunks(jid)[2]['path'])
    Runner(store,engine).run([jid])
    assert store.job(jid)['status']=='DONE'
    assert engine.manager.calls==[2,1,0]
    assert digest(store.chunks(jid)[2]['path'])==saved
    assert [c['idx'] for c in store.chunks(jid)]==[0,1,2]

def test_execution_overrides_do_not_change_voice(tmp_path):
    store=Store(tmp_path)
    jid=store.add('Xin chào.','execution',DEFAULTS|{'output_dir':str(tmp_path/'out'),'voice':'original'})
    class Recorder(FakeEngine):
        def synthesize(self,text,settings):
            self.settings=settings.copy()
            return super().synthesize(text,settings)
    engine=Recorder()
    runner=Runner(store,engine)
    runner.execution={'voice':'wrong','threads':2,'accel_mode':'cpu','gpu_batch':2}
    runner.run([jid])
    assert engine.settings['voice']=='original'
    assert engine.settings['threads']==2

def test_gpu_failure_falls_back_and_strict_benchmark_reports_failure(monkeypatch,tmp_path):
    import pytest
    class Stub:
        def __init__(self,device,threads=2):
            if device=='gpu':
                raise RuntimeError('GPU unavailable')
            self.device=device
            self.process=type('P',(),{'poll':lambda self:None})()
        def close(self):
            pass
    monkeypatch.setattr(a,'ProcessEngine',Stub)
    manager=a.Manager()
    events=[]
    mode,batch=manager.configure(DEFAULTS|{'accel_mode':'gpu'},lambda *args:events.append(args))
    assert mode=='cpu'
    assert events and manager.lanes[0].device=='cpu'
    manager.close()
    with pytest.raises(RuntimeError,match='GPU unavailable'):
        manager.configure(DEFAULTS|{'accel_mode':'gpu'},strict=True)

def test_cpu_threads_reach_each_worker(monkeypatch):
    class Stub:
        def __init__(self,device,threads):
            self.device,self.threads=device,threads
            self.process=type('P',(),{'poll':lambda self:None})()
        def close(self): pass
    monkeypatch.setattr(a,'ProcessEngine',Stub)
    for mode,expected in [('cpu',[('cpu',3)]),('cpu2',[('cpu',3),('cpu',3)]),('hybrid',[('gpu',2),('cpu',3)]),('gpu',[('gpu',2)])]:
        manager=a.Manager()
        manager.configure(DEFAULTS|{'accel_mode':mode,'threads':3,'gpu_batch':32},strict=True)
        assert [(x.device,x.threads) for x in manager.lanes]==expected
        assert manager.batch==32
        manager.close()
