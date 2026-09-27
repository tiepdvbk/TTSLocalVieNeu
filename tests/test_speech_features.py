import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from app.core import Store, Runner, DEFAULTS
from app.speech_plan import parse_srt, format_srt, make_plan
from app import voice_library
from tools.runtime_support import voice_argument
from tests.test_core import FakeEngine

SRT='1\n00:00:01,000 --> 00:00:02,000\n<i>Xin chào.</i>\n\n2\n00:00:03,000 --> 00:00:04,000\nCâu thứ hai.\n'

def test_srt_parse_roundtrip_and_reject():
    cues=parse_srt('\ufeff'+SRT.replace('\n','\r\n'))
    assert cues[0]['text']=='Xin chào.'
    assert parse_srt(format_srt(cues))==cues
    for bad in ('', '1\nnot a time\nHello',SRT.replace('00:00:02,000','00:00:00,500')):
        with pytest.raises(ValueError): parse_srt(bad)
    overlap=SRT.replace('00:00:03,000','00:00:01,500')
    with pytest.raises(ValueError): make_plan(overlap,DEFAULTS|dict(srt_mode='timeline'),True)
    assert len(make_plan(overlap,DEFAULTS,True)[1])==2

def test_punctuation_decimal_newline():
    plan,_=make_plan('Giá 3.5 đồng, rất tốt. Đi thôi!\nNgày mai.',DEFAULTS|dict(pause_custom=True))
    assert [p['boundary'] for p in plan]==['comma','sentence','newline','newline']
    assert plan[0]['text']=='Giá 3.5 đồng,'

@pytest.mark.parametrize('mode',['continuous','timeline'])
def test_srt_export_preserves_source_and_timing(tmp_path,mode):
    src=tmp_path/'story.srt'; src.write_text(SRT,encoding='utf-8')
    store=Store(tmp_path)
    settings=DEFAULTS|dict(output_dir=str(tmp_path),same_folder=True,export_srt=True,srt_mode=mode,pitch=2,speed=1.2)
    jid=store.add(SRT,src.name,settings,src)
    Runner(store,FakeEngine()).run([jid])
    job=store.job(jid); assert job['status']=='DONE',job['error']
    output=Path(job['output']); sidecar=output.with_suffix('.srt')
    assert sidecar!=src and src.read_text(encoding='utf-8')==SRT
    cues=parse_srt(sidecar.read_text(encoding='utf-8-sig'))
    if mode=='timeline':
        assert cues==parse_srt(SRT)
        assert sf.info(output).duration==4
        wav,_=sf.read(output); assert np.max(np.abs(wav[:48000]))==0
    else:
        assert cues[0]['start']==0
        assert abs(cues[-1]['end']-sf.info(output).duration)<.002

def test_timeline_refuses_to_cut_words(tmp_path):
    store=Store(tmp_path)
    settings=DEFAULTS|dict(output_dir=str(tmp_path/'out'),srt_mode='timeline',srt_max_speed=1.)
    jid=store.add('1\n00:00:00,000 --> 00:00:00,100\nXin chào.', 'short',settings,is_srt=True)
    Runner(store,FakeEngine()).run([jid])
    assert store.job(jid)['status']=='FAILED'
    assert 'Không cắt lời' in store.job(jid)['error']

def test_timeline_compresses_within_limit(tmp_path):
    class Longer(FakeEngine):
        def synthesize(self,text,settings):
            return np.tile(super().synthesize(text,settings),8)
    store=Store(tmp_path)
    settings=DEFAULTS|dict(output_dir=str(tmp_path/'out'),srt_mode='timeline',srt_max_speed=2.)
    jid=store.add('1\n00:00:00,000 --> 00:00:01,500\nXin chào.','fit',settings,is_srt=True)
    Runner(store,Longer()).run([jid])
    job=store.job(jid)
    assert job['status']=='DONE',job['error']
    assert sf.info(job['output']).duration==1.5

def test_clone_snapshot_and_worker_arrays(tmp_path,monkeypatch):
    monkeypatch.setattr(voice_library,'LIBRARY_DIR',tmp_path/'voices')
    key='clone:'+'a'*32; path=voice_library.voice_file(key); path.parent.mkdir()
    data=dict(speaker_emb=[.1]*192,codes=[[1]*16]*12)
    path.write_text(json.dumps(dict(id=key,name='Test',voice=data)),encoding='utf-8')
    store=Store(tmp_path)
    jid=store.add('Xin chào.','test',DEFAULTS|dict(voice=key))
    path.unlink()
    settings=json.loads(store.job(jid)['settings'])
    assert settings['voice_data']==data
    resolved=voice_argument(settings)
    assert resolved['speaker_emb'].dtype==np.float32
    assert resolved['codes'].shape==(12,16)
    with pytest.raises(ValueError): voice_library.voice_file('clone:../../bad')
