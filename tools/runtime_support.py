"""Small-memory adapters around the unmodified, pinned VieNeu SDK."""
import gc
import time


try:
    from .clone_quality import select_takes
except ImportError:          # worker process: tools/ itself is on sys.path
    from clone_quality import select_takes

# temperature, top_k, top_p, repetition penalty, max chunk chars
CLONE_PRESETS = {
    'stable': (.7, 25, .9, 1.2, 180),
    'natural': (.8, 25, .95, 1.2, 220),
    # Expressive sampling made safe by take selection: several candidates are
    # scored (speaker similarity, speaking rate, tail) and the best one is kept.
    'expressive': (.85, 30, .95, 1.15, 160),
}


def synthesis_settings(settings):
    """Opt-in presets are snapshotted in new jobs; legacy jobs stay unchanged."""
    s=settings.copy()
    if s.get('voice','').startswith('clone:'):
        mode=s.get('clone_quality','manual')
        if mode == 'short80':
            s['chunk_size']=min(s['chunk_size'],79)
        if mode in CLONE_PRESETS:
            temperature,top_k,top_p,penalty,chars=CLONE_PRESETS[mode]
            s.update(temperature=temperature,top_k=top_k,top_p=top_p,repetition_penalty=penalty)
            s['chunk_size']=min(s['chunk_size'],chars)
            if mode == 'expressive' and s.get('take_select','auto')=='auto':
                s['take_select']='best'
    return s


def reference_embedding(tts, settings):
    """Target speaker vector used to score takes (clone or preset)."""
    import numpy as np
    data=settings.get('voice_data')
    if data and data.get('reference_emb') is not None:
        return np.asarray(data['reference_emb'],dtype=np.float32)
    try:
        return tts._resolve_ref(voice_argument(settings),None,False,False)[0]
    except Exception:
        return None


def speaker_embedder(tts):
    engine=getattr(tts,'engine',None)
    try:
        encoder=engine._ensure_speaker_encoder() if hasattr(engine,'_ensure_speaker_encoder') else getattr(engine,'speaker_encoder',None)
    except Exception:
        encoder=None
    return (lambda wav: encoder.embed(wav,48000)) if encoder is not None else None


_END = tuple('.,;:!?…"”’)»]')


def speakable(text, boundary='chunk'):
    """A chunk cut mid-sentence (plan boundary 'chunk') gets a comma so the
    model keeps a continuing intonation instead of the SDK's added full stop,
    which made the cut sound like a clipped sentence end."""
    stripped=text.rstrip()
    if boundary!='chunk' or not stripped or stripped.endswith(_END):
        return stripped
    return stripped+','


def synthesize_texts(tts, texts, settings, generate, log=None):
    """Run ``generate`` (texts -> wavs) through take selection + tail repair.
    Jobs created before this feature have no ``take_select`` key: unchanged."""
    if 'take_select' not in settings:
        return generate(texts)
    mode=settings.get('take_select','auto')
    embed=speaker_embedder(tts) if mode!='off' else None
    reference=reference_embedding(tts,settings) if embed else None
    return select_takes(texts,generate,mode,settings.get('take_count',3),embed,reference,
                        settings.get('tail_fix',True),log=log)


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
