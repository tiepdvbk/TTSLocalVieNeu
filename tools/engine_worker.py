"""Private, persistent TTS worker. One JSON request/response per line."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['HF_HOME'] = str(ROOT / 'models/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json
import time
import traceback
from contextlib import redirect_stdout
protocol = sys.stdout
sys.stdout = sys.stderr  # keep library messages out of the protocol
import numpy as np
import soundfile as sf
from runtime_support import gpu_synth, instrument_gpu, voice_argument, synthesis_settings
if os.environ.get('TTS_PARENT_PID'):
    import threading
    import psutil
    parent_pid=int(os.environ['TTS_PARENT_PID'])
    def watch_parent():
        while True:
            time.sleep(2)
            if not psutil.pid_exists(parent_pid):
                os._exit(0)
    threading.Thread(target=watch_parent,daemon=True).start()

def reply(data):
    protocol.write(json.dumps(data, ensure_ascii=True) + '\n')
    protocol.flush()

def synth(tts, texts, s, device):
    s=synthesis_settings(s)
    args = dict(voice=voice_argument(s), use_ref_codes=s.get('use_ref_codes',True), max_chars=s['chunk_size'],
                temperature=s['temperature'], top_k=s['top_k'], top_p=s['top_p'],
                repetition_penalty=s['repetition_penalty'], apply_watermark=False)
    if device == 'gpu':
        import torch
        return gpu_synth(tts, texts, args, torch)
    return [tts.infer(text, batch_size=1, **args) for text in texts]

tts = None
for line in sys.stdin:
    try:
        req = json.loads(line)
        if req['op'] == 'close':
            break
        if req['op'] == 'init':
            from vieneu.v3turbo import V3TurboVieNeuTTS
            device = req['device']
            info = {'device':device, 'pid':os.getpid()}
            if device == 'gpu':
                import torch
                torch.set_num_threads(2)
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA không khả dụng trong runtime đã cài.')
                test = torch.ones((32,32),device='cuda')
                (test @ test).sum().item()
                props = torch.cuda.get_device_properties(0)
                # Leave room for Windows/display and avoid WDDM shared-memory
                # spill. OOM is handled by the adaptive batch retry below.
                torch.cuda.set_per_process_memory_fraction(0.85)
                info.update(name=props.name, vram_mb=props.total_memory//2**20,
                            capability=list(torch.cuda.get_device_capability(0)), torch=torch.__version__,
                            memory_budget_mb=round(props.total_memory * 0.85 / 2**20))
                # Pascal has no BF16 support. Keep the original FP32 fidelity.
                tts = V3TurboVieNeuTTS(device='cuda', backend='pytorch', dtype='float32')
                metrics = instrument_gpu(tts, torch)
                from importlib.metadata import version
                info.update(vieneu=version('vieneu'), fused=tts._get_batch_engine().use_fused)
            else:
                tts = V3TurboVieNeuTTS(device='cpu', backend='onnx', precision='fp32', threads=req.get('threads',2))
            reply({'ok':True, 'info':info})
        elif req['op'] == 'generate':
            start = time.perf_counter()
            if device == 'gpu':
                metrics.clear()
                torch.cuda.reset_peak_memory_stats()
            if 'seed' in req:
                np.random.seed(req['seed'])
                if device == 'gpu':
                    torch.manual_seed(req['seed'])
            wavs = synth(tts, req['texts'], req['settings'], device)
            synth_seconds = time.perf_counter() - start
            if len(wavs) != len(req['paths']):
                raise RuntimeError('Model trả về sai số lượng đoạn.')
            durations = []
            for wav, name in zip(wavs, req['paths']):
                wav = np.asarray(wav,dtype=np.float32).reshape(-1)
                if len(wav) < 4800 or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-6:
                    raise RuntimeError('Model trả về đoạn audio không hợp lệ.')
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.wav.tmp')
                sf.write(str(tmp),wav,48000,format='WAV',subtype='PCM_16')
                with open(tmp,'r+b') as f:
                    os.fsync(f.fileno())
                os.replace(tmp,path)
                durations.append(len(wav)/48000)
            info = {'ok':True, 'durations':durations, 'seconds':time.perf_counter()-start}
            info.update(synth_seconds=synth_seconds, write_seconds=info['seconds']-synth_seconds)
            if device == 'gpu':
                info['gpu_peak_mb'] = torch.cuda.max_memory_allocated()/2**20
                info['gpu_reserved_mb'] = torch.cuda.memory_reserved()/2**20
                info['timings'] = metrics.copy()
                info['batch_limit'] = getattr(tts, '_studio_batch_limit', None)
                info['fused'] = tts._get_batch_engine().use_fused
            reply(info)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        reply({'ok':False, 'error':f'{type(exc).__name__}: {exc}'})
