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
