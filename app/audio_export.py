"""Stream cached chunks into narration or a fixed SRT timeline; bounded RAM."""
import numpy as np
import soundfile as sf
from .speech_plan import pause_after

RATE=48000


def silence(dest, frames):
    block=np.zeros(65536,dtype=np.float32)
    while frames>0:
        size=min(frames,len(block)); dest.write(block[:size]); frames-=size


def copy_chunk(dest, path, trim=False):
    if trim:
        wav,sr=sf.read(path,dtype='float32')
        active=np.flatnonzero(np.abs(wav)>.005)
        if len(active): wav=wav[max(0,active[0]-240):min(len(wav),active[-1]+481)]
        dest.write(wav)
    else:
        with sf.SoundFile(path) as src:
            for block in src.blocks(blocksize=65536,dtype='float32'): dest.write(block)


def build_audio(chunks, settings, cache, render):
    plan=settings.get('_plan') or [dict(boundary='chunk') for _ in chunks]
    cues=settings.get('_srt_cues',[])
    if len(plan)!=len(chunks): raise ValueError('Kế hoạch đoạn không khớp cache.')
    raw=cache/'merged.wav'; processed=cache/'processed.wav'
    captions=[]
    def merge(dest, indexes):
        spans=[]
        for n,i in enumerate(indexes):
            start=dest.tell()/RATE
            copy_chunk(dest,chunks[i]['path'],settings.get('trim_silence',False))
            spans.append(dict(start=start,end=dest.tell()/RATE,text=chunks[i]['text']))
            if n<len(indexes)-1:
                silence(dest,round(RATE*pause_after(plan[i]['boundary'],settings)))
        return spans
    if cues and settings.get('srt_mode')=='timeline':
        groups={}
        for i,p in enumerate(plan): groups.setdefault(p.get('cue'),[]).append(i)
        if any(b['start']<a['end'] for a,b in zip(cues,cues[1:])):
            raise ValueError('Mốc SRT bị chồng lấn; dùng chế độ đọc nối tiếp.')
        with sf.SoundFile(processed,'w',samplerate=RATE,channels=1,subtype='PCM_16') as dest:
            for ci,cue in enumerate(cues):
                indexes=groups.get(ci,[])
                with sf.SoundFile(raw,'w',samplerate=RATE,channels=1,subtype='PCM_16') as part: merge(part,indexes)
                fitted=cache/'cue.wav'
                render(raw,fitted,settings)
                budget=round(cue['end']*RATE)-round(cue['start']*RATE)
                actual=sf.info(fitted).frames
                if actual>budget:
                    factor=actual/budget*1.01
                    if factor>settings.get('srt_max_speed',2.):
                        raise ValueError(f'SRT mục {ci+1} quá dài cho khung giờ (cần tăng {factor:.2f}×). Tăng giới hạn co giọng hoặc chọn đọc nối tiếp. Không cắt lời.')
                    compressed=cache/'cue-fit.wav'
                    render(fitted,compressed,dict(speed=factor,volume=100))
                    fitted=compressed
                if sf.info(fitted).frames>budget:
                    raise ValueError(f'Không thể ghép SRT mục {ci+1} mà không cắt lời. Chọn đọc nối tiếp.')
                silence(dest, max(0,round(cue['start']*RATE)-dest.tell()))
                copy_chunk(dest,fitted)
                silence(dest,max(0,round(cue['end']*RATE)-dest.tell()))
                captions.append(cue)
    else:
        with sf.SoundFile(raw,'w',samplerate=RATE,channels=1,subtype='PCM_16') as dest:
            captions=merge(dest,list(range(len(chunks))))
            raw_duration=dest.tell()/RATE
        render(raw,processed,settings)
        ratio=sf.info(processed).duration/raw_duration
        captions=[dict(c,start=c['start']*ratio,end=c['end']*ratio) for c in captions]
    return processed,captions
