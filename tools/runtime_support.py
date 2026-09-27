"""Small-memory adapters around the unmodified, pinned VieNeu SDK."""
import gc
import time


def voice_argument(settings):
    data=settings.get('voice_data')
    if data:
        import numpy as np
        return dict(speaker_emb=np.asarray(data['speaker_emb'],dtype=np.float32),
                    codes=np.asarray(data['codes'],dtype=np.int64))
    if settings.get('voice','').startswith('clone:'):
        raise ValueError('Tác vụ thiếu dữ liệu giọng clone.')
    return settings.get('voice') or None


class SingleGraphCache(dict):
    """Retain only one CUDA graph shape on 4 GB cards, evict BEFORE allocation."""
    def __init__(self, release):
        super().__init__()
        self.release = release

    def get(self, key, default=None):
        if key not in self and self:
            self.clear()
            gc.collect()
            self.release()
        return super().get(key, default)


def instrument_gpu(tts, torch):
    batch = tts._get_batch_engine()
    if torch.cuda.get_device_properties(0).total_memory <= 6 * 2**30:
        batch._fused = SingleGraphCache(torch.cuda.empty_cache)
    metrics = {}
    for owner, name, label in ((batch, '_generate_codes_batch', 'codes_seconds'),
                               (tts.engine, '_decode_codes', 'decode_seconds')):
        original = getattr(owner, name)
        def timed(*args, _fn=original, _label=label, **kwargs):
            # Stage boundaries only, never synchronize inside each generated frame.
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                return _fn(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                metrics[_label] = metrics.get(_label, 0.0) + time.perf_counter() - start
        setattr(owner, name, timed)
    return metrics


def gpu_synth(tts, texts, args, torch):
    """Remember a successful reduced batch; never repeatedly retry an OOM size."""
    limit = min(len(texts), getattr(tts, '_studio_batch_limit', len(texts)))
    if len(texts) > limit:
        return [wav for i in range(0, len(texts), limit)
                for wav in gpu_synth(tts, texts[i:i + limit], args, torch)]
    try:
        return tts.infer_batch(texts, batch_size=limit, **args)
    except torch.cuda.OutOfMemoryError:
        # Release the exception traceback (and its tensors) before retrying.
        if limit == 1:
            raise
    batch = tts._get_batch_engine()
    batch._fused.clear()
    batch._graphs.clear()
    gc.collect()
    torch.cuda.empty_cache()
    tts._studio_batch_limit = max(1, limit // 2)
    print(f'GPU thiếu VRAM: giảm nhóm xuống {tts._studio_batch_limit} và ghi nhớ cho các nhóm sau.', flush=True)
    return gpu_synth(tts, texts, args, torch)
