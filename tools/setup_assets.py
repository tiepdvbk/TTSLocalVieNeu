"""Fetch pinned upstream source and a wheel-distributed FFmpeg; no Git required."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    revision = (ROOT / 'vendor-revision.txt').read_text().strip()
    target = ROOT / 'vendor/VieNeu-TTS'
    if not (target / 'pyproject.toml').exists():
        archive = ROOT / '.tools/vieneu-source.zip'
        urllib.request.urlretrieve(f'https://codeload.github.com/pnnbao97/VieNeu-TTS/zip/{revision}', archive)
        expected = json.loads((ROOT / 'setup-assets.json').read_text())['sdk_zip_sha256']
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise RuntimeError('VieNeu source checksum mismatch.')
        with tempfile.TemporaryDirectory(dir=ROOT / '.tools') as tmp:
            base = Path(tmp).resolve()
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    if not (base / name).resolve().is_relative_to(base):
                        raise RuntimeError('Unsafe archive member.')
                z.extractall(base)
            target.parent.mkdir(exist_ok=True)
            shutil.move(str(base / f'VieNeu-TTS-{revision}'), target)
        archive.unlink()
    import imageio_ffmpeg
    ffmpeg = ROOT / 'tools/ffmpeg.exe'
    if not ffmpeg.exists():
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
    # Retain the SDK's Apache license in built distributions.
    license_dir = ROOT / 'licenses/vieneu-3.8.3'
    license_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target / 'LICENSE', license_dir / 'LICENSE')
    print('Pinned SDK and FFmpeg ready.', flush=True)


if __name__ == '__main__':
    main()
