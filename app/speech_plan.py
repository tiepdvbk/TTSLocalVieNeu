"""Subtitle parsing and punctuation-aware chunk plans, independent of inference."""
import html
import re

STAMP=r'(\d{2,}):([0-5]\d):([0-5]\d)[,.](\d{3})'
TIMING=re.compile(r'^'+STAMP+r'\s*-->\s*'+STAMP+r'(?:\s+.*)?$')


def parse_srt(text):
    cues=[]
    for block in re.split(r'\n\s*\n', text.lstrip('\ufeff').replace('\r\n','\n').replace('\r','\n').strip()):
        lines=block.splitlines()
        if not lines:
            continue
        if lines[0].strip().isdigit():
            lines=lines[1:]
        match=TIMING.fullmatch(lines[0].strip()) if lines else None
        if not match:
            raise ValueError(f'SRT sai dòng thời gian ở mục {len(cues)+1}.')
        nums=list(map(int, match.groups()))
        def ms(n): return ((n[0]*60+n[1])*60+n[2])*1000+n[3]
        start,end=ms(nums[:4]),ms(nums[4:])
        body=html.unescape(re.sub(r'<[^>]*>|\{\\[^}]*\}', '', '\n'.join(lines[1:]))).strip()
        if end<=start or not body:
            raise ValueError(f'SRT mục {len(cues)+1}: thời gian hoặc nội dung không hợp lệ.')
        cues.append(dict(start=start/1000,end=end/1000,text=body))
    if not cues:
        raise ValueError('SRT không có nội dung.')
    return cues


def format_srt(cues):
    def stamp(seconds):
        ms=max(0,round(seconds*1000)); sec,ms=divmod(ms,1000); minute,sec=divmod(sec,60); hour,minute=divmod(minute,60)
        return f'{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}'
    return '\n\n'.join(f'{i}\n{stamp(c["start"])} --> {stamp(c["end"])}\n{c["text"]}' for i,c in enumerate(cues,1))+'\n'


def plan_text(text, settings):
    from .core import split_text, apply_pronunciations
    text=apply_pronunciations(text, settings.get('_pronunciations'))
    result=[]
    for paragraph in re.split(r'\n+', text.replace('\r\n','\n').replace('\r','\n')):
        if not paragraph.strip(): continue
        if settings.get('pause_custom'):
            units=[]; start=0
            for m in re.finditer(r'[,;:.!?…]["”’»)]*(?=\s|$)', paragraph):
                units.append((paragraph[start:m.end()], 'comma' if m.group()[0] in ',;:' else 'sentence'))
                start=m.end()
            if paragraph[start:].strip(): units.append((paragraph[start:], 'chunk'))
        else:
            units=[(paragraph,'chunk')]
        for unit,kind in units:
            parts=split_text(unit,settings['chunk_size'])
            for i,piece in enumerate(parts):
                result.append(dict(text=piece, boundary=kind if i==len(parts)-1 else 'chunk'))
        if result: result[-1]['boundary']='newline'
    return result


def make_plan(text, settings, is_srt=False):
    cues=parse_srt(text) if is_srt else []
    if cues and settings.get('srt_mode')=='timeline':
        if any(b['start']<a['end'] for a,b in zip(cues,cues[1:])):
            raise ValueError('SRT có mốc chồng lấn hoặc đảo thứ tự. Chọn Đọc nối tiếp hoặc sửa mốc thời gian.')
    plan=[]
    for i,cue in enumerate(cues):
        plan.extend(dict(p,cue=i) for p in plan_text(cue['text'],settings))
    if not cues: plan=plan_text(text,settings)
    return plan,cues


def pause_after(boundary, settings):
    if settings.get('pause_custom'):
        return settings.get('pause_'+boundary, settings.get('gap',.25))
    return settings.get('gap',.25)


def reading_preview(text, settings, is_srt=False, limit=400):
    """What the model will actually read: app chunk plan, pronunciation
    dictionary, then the SDK's own number/symbol normalisation."""
    from .core import load_pronunciations
    from tools.runtime_support import synthesis_settings, speakable
    s=synthesis_settings(settings)
    s.setdefault('_pronunciations', load_pronunciations())
    plan,_=make_plan(text,s,is_srt)
    try:
        from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps as sdk_chunks
    except Exception:
        sdk_chunks=None
    rows=[]
    for i,piece in enumerate(plan[:limit]):
        spoken=speakable(piece['text'],piece['boundary']) if 'take_select' in s else piece['text']
        normalized=' | '.join(sdk_chunks(spoken,max_chars=s['chunk_size'])[0]) if sdk_chunks else spoken
        rows.append(dict(index=i+1,text=piece['text'],normalized=normalized,boundary=piece['boundary']))
    return rows,len(plan)
