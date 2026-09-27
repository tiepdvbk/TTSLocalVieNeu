"""Repeatable offline performance sample; never changes user jobs or settings."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.acceleration import Manager
from app.core import DEFAULTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--order', choices=('original', 'length'), default='original')
    args = parser.parse_args()
    if args.count < 1 or args.batch < 1:
        parser.error('--count and --batch must be positive')
    with sqlite3.connect((ROOT / 'data/app.db').as_uri() + '?mode=ro', uri=True) as db:
        job = db.execute('SELECT job_id FROM chunks GROUP BY job_id ORDER BY count(*) DESC LIMIT 1').fetchone()
        if job is None:
            parser.error('No saved job with text chunks is available')
        rows = db.execute('SELECT idx,text FROM chunks WHERE job_id=? ORDER BY idx', job).fetchall()
        settings = DEFAULTS | json.loads(db.execute('SELECT settings FROM jobs WHERE id=?', job).fetchone()[0])
    # Evenly spaced samples cover the whole document; the IDs are saved for reproduction.
    sample_count = min(args.count, len(rows))
    rows = [rows[i * len(rows) // sample_count] for i in range(sample_count)]
    settings.update(accel_mode='gpu', gpu_batch=args.batch, length_bucketing=args.order == 'length')
    folder = ROOT / 'logs' / ('performance-' + args.label)
    folder.mkdir(parents=True, exist_ok=True)
    manager = Manager()
    report = dict(label=args.label, settings=settings, sample_ids=[r[0] for r in rows], groups=[])
    try:
        begin = time.perf_counter()
        manager.configure(settings, strict=True)
        report['load_seconds'] = time.perf_counter() - begin
        report['worker'] = manager.lanes[0].info
        # Small warm-up kept separate from the measured sample.
        manager.lanes[0].generate([rows[0][1]], [folder / 'warmup.wav'], settings)
        start = time.perf_counter()
        for group, result, elapsed in manager.stream(
                [(i, text, folder / f'{i:06d}.wav') for i, text in rows], settings, strict=True):
            report['groups'].append(dict(ids=[x[0] for x in group], wall_seconds=elapsed, **result))
            print(json.dumps(report['groups'][-1]), flush=True)
        report['seconds'] = time.perf_counter() - start
        report['audio_seconds'] = sum(sum(g['durations']) for g in report['groups'])
        report['realtime'] = report['audio_seconds'] / report['seconds']
    finally:
        manager.close()
        (folder / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('groups', 'settings')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
