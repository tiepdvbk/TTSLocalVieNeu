import json
import numpy as np
import soundfile as sf
from app import acceleration as a
from app.voice_profiles import profile_settings, PROFILES, canonical_voice
from app.core import voices
from tests.test_core import FakeEngine


def test_profiles_cover_presets_and_aliases():
    assert set(PROFILES) == {name for name, _ in voices()[0]}
    assert canonical_voice('Minh Quân') == 'Hải Đăng'
    assert canonical_voice('Anh Khôi') == 'Thiện Minh'
    assert profile_settings('Anh Khôi') == profile_settings('Thiện Minh')


def test_story_preview_applies_profile_and_invalidates_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'ROOT', tmp_path)
    class Tone(FakeEngine):
        def synthesize(self, text, settings):
            self.calls += 1
            return np.sin(np.arange(48000) * (440 * 2 * np.pi / 48000)).astype(np.float32) * .2
    engine = Tone()
    path = a.create_preview(engine, 'Thiện Minh')
    info = sf.info(path)
    expected = (2 + profile_settings('Thiện Minh')['gap']) / profile_settings('Thiện Minh')['speed']
    assert abs(info.duration - expected) < .1
    assert a.valid_preview('Thiện Minh')
    assert np.max(np.abs(sf.read(path)[0])) <= .98
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert meta['profile'] == profile_settings('Thiện Minh')
    monkeypatch.setattr(a, 'SAMPLE_TEXT', a.SAMPLE_TEXT + ' Một câu mới.')
    assert a.preview_path('Thiện Minh') != path
    assert not a.valid_preview('Thiện Minh')
