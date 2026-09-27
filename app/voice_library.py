"""Local, immutable voice enrolments. Never edit the SDK's bundled voices."""
import json
import re
import subprocess
import uuid
from pathlib import Path
import numpy as np
import soundfile as sf
from .core import ROOT, atomic_json
LIBRARY_DIR = ROOT / 'data/voices'


def voice_file(key):
    if not re.fullmatch(r'clone:[0-9a-f]{32}', key):
        raise ValueError('Mã giọng cá nhân không hợp lệ.')
    return LIBRARY_DIR / (key[6:] + '.json')


def entries():
    result=[]
    for path in sorted(LIBRARY_DIR.glob('*.json')):
        try:
            obj=json.loads(path.read_text(encoding='utf-8'))
            if voice_file(obj['id']).name == path.name:
                result.append((obj['id'], obj['name']))
        except (OSError, ValueError, KeyError):
            continue
    return result


def display_name(key):
    return dict(entries()).get(key, key)


def load_voice(key):
    try:
        obj=json.loads(voice_file(key).read_text(encoding='utf-8'))
        data=obj['voice']
        emb=np.asarray(data['speaker_emb'], dtype=np.float32)
        codes=np.asarray(data['codes'], dtype=np.int64)
        if emb.ndim!=1 or not 1<=emb.size<=4096 or not np.isfinite(emb).all():
            raise ValueError('Embedding không hợp lệ.')
        if codes.ndim!=2 or codes.shape[1]!=16 or not 1<=len(codes)<=1500 or codes.min()<0 or codes.max()>1023:
            raise ValueError('Mã âm thanh không hợp lệ.')
        return data
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('Không đọc được giọng cá nhân; hãy tạo lại giọng từ mẫu thu.') from exc


def create_voice(engine, name, source, start=0., seconds=8., denoise=False):
    name=name.strip()
    if not name or len(name)>80:
        raise ValueError('Tên giọng cần từ 1 đến 80 ký tự.')
    if any(n.casefold()==name.casefold() for _,n in entries()):
        raise ValueError('Tên giọng đã tồn tại. Hãy dùng tên khác.')
    if not 3<=seconds<=8 or start<0:
        raise ValueError('Chọn đoạn mẫu dài 3–8 giây, vị trí bắt đầu không âm.')
    key='clone:'+uuid.uuid4().hex
    target=voice_file(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    sample=target.with_suffix('.wav')
    try:
        result=subprocess.run([str(ROOT/'tools/ffmpeg.exe'),'-hide_banner','-loglevel','error','-nostdin','-y',
            '-ss',str(start),'-i',str(source),'-t',str(seconds),'-vn','-ac','1','-ar','48000',str(sample)],
            capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:
            raise ValueError(result.stderr.decode('utf-8',errors='replace'))
        wav,sr=sf.read(sample)
        if len(wav)<3*sr or not np.isfinite(wav).all() or np.max(np.abs(wav))<.001:
            raise ValueError('Mẫu cần ít nhất 3 giây, có tiếng nói rõ; đoạn đang chọn quá ngắn hoặc im lặng.')
        engine.load(4)
        if denoise and getattr(engine.tts.engine,'denoiser',None) is None:
            raise ValueError('Thiếu model lọc nhiễu. Chạy BUILD.bat -SetupOnly để tải, hoặc bỏ chọn lọc nhiễu.')
        emb,codes=engine.tts.encode_reference(sample,denoise=denoise)
        data=dict(speaker_emb=np.asarray(emb,dtype=float).reshape(-1).tolist(), codes=np.asarray(codes,dtype=int).tolist())
        atomic_json(target,dict(id=key,name=name,voice=data,seconds=len(wav)/sr,denoise=denoise))
        load_voice(key)
        return key
    except Exception:
        target.unlink(missing_ok=True)
        sample.unlink(missing_ok=True)
        raise
