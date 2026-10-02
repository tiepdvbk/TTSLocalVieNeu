import json
from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf
from app import voice_library as v
from app.core import DEFAULTS,Store
from tools.runtime_support import synthesis_settings


def tone(seconds=4,amplitude=.02):
    return (amplitude*np.sin(np.arange(round(seconds*48000))*2*np.pi*220/48000)).astype('float32')


def test_preparation_preserves_internal_pause_and_bounds():
    wav=np.concatenate([np.zeros(24000),tone(2),np.zeros(24000),tone(2),np.zeros(24000)])
    prepared,report=v.prepare_sample(wav,48000)
    assert 4.5<=len(prepared)/48000<5.5
    assert np.max(np.abs(prepared))<=.951
    assert report['active_seconds']==pytest.approx(4,abs=.04)
    assert np.max(np.abs(prepared))>np.max(np.abs(wav))
    assert np.count_nonzero(np.abs(prepared)<1e-8)>=23000


def test_signal_feedback_and_silence_rejection():
    assert v.sample_report(tone(amplitude=.002),48000)['warnings']
    assert v.sample_report(np.clip(tone(amplitude=2),-1,1),48000)['clipping_percent']>0
    with pytest.raises(ValueError,match='quá ít'):
        v.prepare_sample(np.zeros(4*48000),48000)


def test_profile_applies_only_to_new_clone_settings():
    user=DEFAULTS|dict(voice='clone:'+'a'*32,temperature=1.,chunk_size=300)
    stable=synthesis_settings(user)
    assert stable['temperature']==.7 and stable['chunk_size']==180
    assert synthesis_settings(user|dict(clone_quality='natural'))['temperature']==.8
    assert synthesis_settings(user|dict(clone_quality='manual'))['temperature']==1.
    assert synthesis_settings(user|dict(voice='Thiện Minh'))['temperature']==1.
    legacy={k:value for k,value in user.items() if k!='clone_quality'}
    assert synthesis_settings(legacy)==legacy
    assert user['temperature']==1.


def test_enrollment_keeps_raw_and_prepared_samples(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'LIBRARY_DIR',tmp_path/'voices')
    source=tmp_path/'input.wav';sf.write(source,tone(4),48000)
    before=source.read_bytes()
    def encode(path,denoise):
        assert path.name.endswith('.prepared.wav') and denoise
        return np.ones(192),np.ones((50,16),dtype=int)
    engine=SimpleNamespace(load=lambda _:None,tts=SimpleNamespace(engine=SimpleNamespace(denoiser=True),encode_reference=encode))
    key=v.create_voice(engine,'Test voice',source,seconds=4)
    meta=json.loads(v.voice_file(key).read_text(encoding='utf-8'))
    assert meta['prepared'] and meta['denoise'] and meta['quality']['seconds']==4
    assert source.read_bytes()==before
    assert v.voice_file(key).with_suffix('.wav').exists()
    assert v.voice_file(key).with_suffix('.prepared.wav').exists()
    store=Store(tmp_path)
    jid=store.add('Một câu kiểm tra.', 'sample',DEFAULTS|dict(voice=key,temperature=1.))
    assert json.loads(store.job(jid)['settings'])['temperature']==.7


def test_delete_only_selected_private_voice_and_preserve_job_snapshot(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'LIBRARY_DIR',tmp_path/'voices')
    folder=v.LIBRARY_DIR;folder.mkdir()
    keys=['clone:'+'a'*32,'clone:'+'b'*32]
    data=dict(speaker_emb=[.1]*192,codes=[[1]*16]*10)
    for key in keys:
        target=v.voice_file(key)
        target.write_text(json.dumps(dict(id=key,name=key,voice=data)),encoding='utf-8')
        for audio in (target.with_suffix('.wav'),target.with_suffix('.prepared.wav')): audio.write_bytes(b'voice')
    store=Store(tmp_path)
    jid=store.add('Một câu.','story',DEFAULTS|dict(voice=keys[0]))
    assert v.delete_voice(keys[0])==keys[0]
    assert not v.voice_file(keys[0]).exists()
    assert not v.voice_file(keys[0]).with_suffix('.wav').exists()
    assert not v.voice_file(keys[0]).with_suffix('.prepared.wav').exists()
    assert v.voice_file(keys[1]).exists()
    assert json.loads(store.job(jid)['settings'])['voice_data']==data
    with pytest.raises(ValueError): v.delete_voice('Thiện Minh')


@pytest.mark.parametrize('is_srt',[False,True])
@pytest.mark.parametrize('custom',[False,True])
def test_short_clone_preserves_content_and_cue_mapping(is_srt,custom):
    from app.speech_plan import make_plan
    text='Đêm ấy, nàng đứng bên cửa sổ nhìn những giọt mưa rơi xuống khu vườn vắng lặng. Chàng vẫn chưa trở về, dù lời hẹn đã qua từ rất lâu. '
    text=text*5
    source=f'1\n00:00:00,000 --> 00:01:00,000\n{text}\n\n2\n00:01:00,000 --> 00:02:00,000\n{text}' if is_srt else text
    original=DEFAULTS|dict(voice='clone:'+'a'*32,clone_quality='short80',chunk_size=300,temperature=.85,pause_custom=custom)
    settings=synthesis_settings(original)
    assert settings['chunk_size']==79 and settings['temperature']==.85
    assert original['chunk_size']==300
    plan,cues=make_plan(source,settings,is_srt)
    assert all(0<len(p['text'])<80 for p in plan)
    expected=' '.join((text*(2 if is_srt else 1)).split())
    assert ' '.join(p['text'] for p in plan)==expected
    if is_srt:
        assert {p['cue'] for p in plan}=={0,1}
        assert cues[-1]['end']==120
    assert synthesis_settings(original|dict(voice='Thiện Minh'))['chunk_size']==300
    assert synthesis_settings(original|dict(chunk_size=50))['chunk_size']==50


def test_short_clone_hard_limit_for_unbroken_text():
    from app.core import split_text
    text='x'*250
    chunks=split_text(text,synthesis_settings(DEFAULTS|dict(voice='clone:a',clone_quality='short80'))['chunk_size'])
    assert ''.join(chunks)==text and max(map(len,chunks))==79
