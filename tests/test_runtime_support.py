from types import SimpleNamespace
import pytest
from tools.runtime_support import SingleGraphCache, gpu_synth


def test_graph_evicted_before_allocating_a_new_shape():
    released = []
    cache = SingleGraphCache(lambda: released.append(len(cache)))
    graph = object()
    cache['first'] = graph
    assert cache.get('first') is graph
    assert not released
    assert cache.get('second') is None
    assert released == [0]


def test_oom_reduces_and_remembers_batch_preserving_order():
    class OOM(Exception):
        pass
    calls = []
    engine = SimpleNamespace(_fused={}, _graphs={})
    class TTS:
        def _get_batch_engine(self):
            return engine
        def infer_batch(self, texts, batch_size, **args):
            calls.append(batch_size)
            if batch_size > 2:
                raise OOM()
            return texts
    torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda:None))
    tts = TTS()
    texts = list(range(7))
    assert gpu_synth(tts, texts, {}, torch) == texts
    assert tts._studio_batch_limit <= 2
    calls.clear()
    assert gpu_synth(tts, texts, {}, torch) == texts
    assert max(calls) <= 2
    with pytest.raises(OOM):
        gpu_synth(SimpleNamespace(infer_batch=lambda *a,**k: (_ for _ in ()).throw(OOM())), [1], {}, torch)
