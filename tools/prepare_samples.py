"""Pre-cache every narration sample; reusable by BUILD.bat and maintainers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import Engine, voices
from app.acceleration import create_preview, valid_preview


def main():
    items = voices()[0]
    engine = Engine()
    try:
        for i, (voice, _) in enumerate(items, 1):
            print(f'[{i}/{len(items)}] {voice}: ' + ('cached' if valid_preview(voice) else 'generating story...'), flush=True)
            create_preview(engine, voice)
        print('ALL STORY SAMPLES READY.', flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
