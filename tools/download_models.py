"""Download the exact tested CPU and GPU model revisions, resumable via HF cache."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / 'models/huggingface')
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# Explicitly use the supported copy/move cache mode on Windows. Parallel
# capability probes can otherwise race and attempt a privileged symlink.
if os.name == 'nt':
    os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
PATTERNS = {
    'pnnbao-ump/VieNeu-TTS-v3-Turbo': ['onnx_update/*', 'update/*', 'config.json', 'speaker_encoder.onnx', 'voices_v3_turbo.json', 'LICENSE*', 'README.md'],
    'OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano': ['*.json', '*.py', '*.safetensors', 'LICENSE*', 'README.md'],
    'OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX': ['moss_audio_tokenizer_decode_full.onnx', 'moss_audio_tokenizer_decode_shared.data', 'moss_audio_tokenizer_decode_step.onnx', 'codec_browser_onnx_meta.json', 'moss_audio_tokenizer_encode.onnx', 'moss_audio_tokenizer_encode.data', 'LICENSE*', 'README.md'],
}

def main():
    from huggingface_hub import snapshot_download
    revisions = json.loads((ROOT / 'model-revisions.json').read_text(encoding='utf-8-sig'))
    for i, (repo, patterns) in enumerate(PATTERNS.items(), 1):
        print(f'Model {i}/{len(PATTERNS)}: {repo}', flush=True)
        snapshot_download(repo, revision=revisions[repo], allow_patterns=patterns, max_workers=3)
        # SDK requests main in offline mode. Publish its ref only after all files finish.
        ref = ROOT / 'models/huggingface/hub' / ('models--' + repo.replace('/', '--')) / 'refs/main'
        ref.parent.mkdir(parents=True, exist_ok=True)
        tmp = ref.with_suffix('.tmp')
        tmp.write_text(revisions[repo], encoding='ascii')
        os.replace(tmp, ref)
    print('MODELS READY: CPU and GPU; offline inference enabled in the app.', flush=True)

if __name__ == '__main__':
    main()
