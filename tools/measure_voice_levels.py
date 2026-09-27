"""Measure cached story samples; report suggested gain, never alter user settings."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import ROOT, voices
from app.acceleration import preview_path, valid_preview


def main():
    rows=[]
    for voice,_ in voices()[0]:
        if not valid_preview(voice):
            raise RuntimeError(f'Create the current story sample first: {voice}')
        path=preview_path(voice)
        result=subprocess.run([str(ROOT/'tools/ffmpeg.exe'), '-hide_banner', '-nostdin', '-i', str(path),
            '-af', 'loudnorm=I=-20:TP=-1:LRA=11:print_format=json', '-f', 'null', '-'],
            capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError(result.stderr)
        level=json.JSONDecoder().raw_decode(result.stderr[result.stderr.rfind('{'):])[0]
        lufs,peak=float(level['input_i']),float(level['input_tp'])
        gain=min(125,max(80,round(100*10**((-20-lufs)/20))))
        rows.append(dict(voice=voice,lufs=lufs,peak_db=peak,relative_volume=gain,path=str(path)))
        print(f'{voice}: {lufs:.2f} LUFS, peak {peak:.2f} dBTP, suggested relative gain {gain}%', flush=True)
    (ROOT/'logs/voice-levels.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
