"""Offline release check using a synthetic sample; isolated from user jobs/voices."""
import json
import time
import socket
import numpy as np
import soundfile as sf
from .core import ROOT, Engine, Store, Runner, DEFAULTS
from . import voice_library
from .acceleration import ProcessEngine
from .speech_plan import parse_srt


def run():
    folder=ROOT/'logs'/time.strftime('features-%Y%m%d-%H%M%S')
    folder.mkdir(parents=True)
    original=voice_library.LIBRARY_DIR
    connect=socket.socket.connect
    def offline(*a,**k): raise RuntimeError('Feature test forbids network')
    socket.socket.connect=offline
    voice_library.LIBRARY_DIR=folder/'voices'
    engine=Engine()
    gpu=None
    try:
        source=next(p for p in (ROOT/'voice_samples').glob('*.wav') if sf.info(p).duration>=8)
        key=voice_library.create_voice(engine,'Synthetic release test',source,seconds=6,denoise=True)
        settings=DEFAULTS | dict(voice=key,threads=4,accel_mode='cpu',output_dir=str(folder/'audio'),
            same_folder=False,export_srt=True,srt_mode='timeline',pitch=1.,normalize_volume=True)
        store=Store(folder)
        text='1\n00:00:01,000 --> 00:00:12,000\nBuổi sáng, cô mở cửa đón ánh nắng.\n\n2\n00:00:13,000 --> 00:00:25,000\nNgoài vườn, tiếng chim vang lên thật dịu dàng.\n'
        jid=store.add(text,'clone-srt',settings,is_srt=True)
        Runner(store,engine).run([jid])
        job=store.job(jid)
        assert job['status']=='DONE',job['error']
        from pathlib import Path
        output=Path(job['output'])
        assert abs(sf.info(output).duration-25)<.001
        assert parse_srt(output.with_suffix('.srt').read_text(encoding='utf-8-sig'))==parse_srt(text)
        payload=json.loads(job['settings'])
        gpu=ProcessEngine('gpu',4)
        target=folder/'gpu-clone.wav'
        gpu.generate(['Câu chuyện bắt đầu dưới ánh trăng.'],[target],payload)
        wav,sr=sf.read(target)
        assert sr==48000 and len(wav)>4800 and np.isfinite(wav).all() and np.max(np.abs(wav))>.001
        result=dict(success=True,cpu_timeline_seconds=sf.info(output).duration,gpu_clone_seconds=len(wav)/sr,
                    denoise=True,voice_shape=list(np.asarray(payload['voice_data']['codes']).shape),folder=str(folder))
        (ROOT/'logs/features-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    finally:
        if gpu: gpu.close()
        engine.close()
        voice_library.LIBRARY_DIR=original
        socket.socket.connect=connect
