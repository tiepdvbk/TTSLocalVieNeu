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


def sample_report(wav, sr):
    """Signal heuristics only: does not identify words, music or speakers."""
    wav=np.asarray(wav,dtype=np.float32).reshape(-1)
    if len(wav)<sr or not np.isfinite(wav).all():
        raise ValueError('Mẫu quá ngắn hoặc có dữ liệu không hợp lệ.')
    frame=max(1,round(.02*sr))
    blocks=wav[:len(wav)//frame*frame].reshape(-1,frame)
    rms=np.sqrt(np.mean(blocks.astype(np.float64)**2,axis=1))
    threshold=max(.003, float(np.percentile(rms,90))*.08)
    active=rms>threshold
    level=20*np.log10(max(float(np.sqrt(np.mean(wav.astype(np.float64)**2))),1e-9))
    clipping=float(np.mean(np.abs(wav)>=.995))
    warnings=[]
    if level < -32: warnings.append('Âm lượng mẫu nhỏ; nên thu gần micro hơn.')
    if clipping>.002: warnings.append('Mẫu có dấu hiệu vỡ đỉnh; giảm mức thu, chuẩn hóa không sửa được tiếng đã vỡ.')
    if float(np.mean(active))<.55: warnings.append('Mẫu có nhiều đoạn rất nhỏ hoặc im lặng; chọn phần nói liền mạch hơn.')
    return dict(seconds=len(wav)/sr,level_db=round(level,1),clipping_percent=round(100*clipping,2),
                active_seconds=round(float(np.sum(active))*frame/sr,2),warnings=warnings)


def prepare_sample(wav,sr):
    report=sample_report(wav,sr)
    if report['active_seconds']<1.5:
        raise ValueError('Mẫu có quá ít tín hiệu rõ. Chọn 3–8 giây nói liền mạch, ít im lặng.')
    wav=np.asarray(wav,dtype=np.float32)-np.mean(wav)
    frame=round(.02*sr)
    blocks=wav[:len(wav)//frame*frame].reshape(-1,frame)
    rms=np.sqrt(np.mean(blocks.astype(np.float64)**2,axis=1))
    active=np.flatnonzero(rms>max(.003,float(np.percentile(rms,90))*.08))
    if len(active):
        start=max(0,active[0]*frame-round(.08*sr))
        end=min(len(wav),(active[-1]+1)*frame+round(.08*sr))
        if end-start>=3*sr: wav=wav[start:end]
    # Constant gain preserves prosody/dynamics; no gate, compressor or pitch shift.
    speech_rms=float(np.sqrt(np.mean(rms[active]**2))) if len(active) else .1
    gain=min(4.,.1/max(speech_rms,1e-6),.95/max(float(np.max(np.abs(wav))),1e-6))
    return (wav*gain).astype(np.float32),report


def crop_sample(source,target,start,seconds):
    if not 3<=seconds<=8 or start<0:
        raise ValueError('Chọn đoạn mẫu dài 3–8 giây, vị trí bắt đầu không âm.')
    result=subprocess.run([str(ROOT/'tools/ffmpeg.exe'),'-hide_banner','-loglevel','error','-nostdin','-y',
        '-ss',str(start),'-i',str(source),'-t',str(seconds),'-vn','-ac','1','-ar','48000',str(target)],
        capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode: raise ValueError(result.stderr.decode('utf-8',errors='replace'))
    wav,sr=sf.read(target,dtype='float32')
    if len(wav)<3*sr or not np.isfinite(wav).all() or np.max(np.abs(wav))<.001:
        raise ValueError('Mẫu cần ít nhất 3 giây, có tiếng nói rõ; đoạn đang chọn quá ngắn hoặc im lặng.')
    return wav,sr


def inspect_sample(source,start=0.,seconds=8.):
    path=ROOT/'cache/clone-previews'/f'{uuid.uuid4().hex}.wav'
    path.parent.mkdir(parents=True,exist_ok=True)
    wav,sr=crop_sample(source,path,start,seconds)
    return dict(path=str(path),report=sample_report(wav,sr))


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


def create_voice(engine, name, source, start=0., seconds=8., denoise=True, prepare=True):
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
    prepared=target.with_suffix('.prepared.wav')
    try:
        wav,sr=crop_sample(source,sample,start,seconds)
        report=sample_report(wav,sr)
        if prepare:
            wav,report=prepare_sample(wav,sr)
            sf.write(prepared,wav,sr,subtype='PCM_16')
        engine.load(4)
        if denoise and getattr(engine.tts.engine,'denoiser',None) is None:
            raise ValueError('Thiếu model lọc nhiễu. Chạy BUILD.bat -SetupOnly để tải, hoặc bỏ chọn lọc nhiễu.')
        emb,codes=engine.tts.encode_reference(prepared if prepare else sample,denoise=denoise)
        data=dict(speaker_emb=np.asarray(emb,dtype=float).reshape(-1).tolist(), codes=np.asarray(codes,dtype=int).tolist())
        atomic_json(target,dict(id=key,name=name,voice=data,seconds=len(wav)/sr,denoise=denoise,
            prepared=prepare,quality=report,preparation_version=2))
        load_voice(key)
        return key
    except Exception:
        target.unlink(missing_ok=True)
        sample.unlink(missing_ok=True)
        prepared.unlink(missing_ok=True)
        raise
