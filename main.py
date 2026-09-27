import sys
import logging
import time
from logging.handlers import RotatingFileHandler
from app.core import ROOT

def main():
    handler = RotatingFileHandler(ROOT / 'logs' / (time.strftime('%Y-%m-%d') + '.log'),
                                  maxBytes=5_000_000, backupCount=4, encoding='utf-8')
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    if '--features-test' in sys.argv:
        from app.featuretest import run
        run()
        return
    if '--self-test' in sys.argv:
        from app.selftest import run
        run()
        return
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import QLockFile, QTimer
    from app import ui
    e2e = '--ui-e2e' in sys.argv
    if e2e:
        from app.core import Store
        test_root = ROOT / 'logs' / time.strftime('ui-e2e-%Y%m%d-%H%M%S')
        ui.Store = lambda: Store(test_root)
    app = QApplication(sys.argv)
    app.setApplicationName('TTS Studio')
    lock = QLockFile(str(ROOT / 'data/studio.lock'))
    lock.setStaleLockTime(0)
    if not lock.tryLock(100):
        QMessageBox.information(None, 'TTS Studio', 'Ứng dụng đang mở. Hãy sử dụng cửa sổ hiện tại.')
        return
    window = ui.Window()
    if any(flag in sys.argv for flag in ('--ui-smoke','--ui-e2e','--preview-test','--theme-test')):
        # Test runs must never alter the user's voice, output folder or settings.
        window.save=lambda:None
    window.show()
    if '--theme-test' in sys.argv:
        from PySide6.QtTest import QTest
        def capture_themes():
            for theme in ('dark','light'):
                window.theme.setCurrentIndex(window.theme.findData(theme))
                box=QMessageBox(QMessageBox.Information,'Kết quả đo tốc độ','GPU • xử lý theo nhóm 32\nĐã chọn: GPU',QMessageBox.Ok,window)
                box.show()
                QTest.qWait(150)
                box.grab().save(str(ROOT/'logs'/f'exe-dialog-{theme}.png'))
                box.close()
                window.voice.showPopup()
                QTest.qWait(150)
                window.voice.view().window().grab().save(str(ROOT/'logs'/f'exe-popup-{theme}.png'))
                window.voice.hidePopup()
            window.close()
        QTimer.singleShot(500,capture_themes)
    if '--ui-smoke' in sys.argv:
        QTimer.singleShot(1600, lambda: window.grab().save(str(ROOT / 'logs/ui-smoke.png')))
        QTimer.singleShot(2200, window.close)
    if '--preview-test' in sys.argv:
        import json
        window.tabs.setCurrentIndex(4)
        window.audio_output.setVolume(0)
        window.sample_table.selectRow(0)
        window.preview_selected()
        start=time.monotonic()
        preview_timer=QTimer(window)
        def check_preview():
            if window.player.position()>200 or time.monotonic()-start>8:
                preview_timer.stop()
                result=dict(error=window.player.error().value,duration=window.player.duration(),
                            position=window.player.position(),source=window.player.source().toLocalFile(),
                            samples=window.sample_table.rowCount(),voice=window.voice.currentData(),
                            settings={key:window.settings()[key] for key in ('speed','gap','volume')})
                (ROOT/'logs/preview-test.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                window.grab().save(str(ROOT/'logs/preview-test.png'))
                window.close()
                app.exit(0 if result['error']==0 and result['duration']>0 and result['position']>0 else 2)
        preview_timer.timeout.connect(check_preview)
        preview_timer.start(100)
    if e2e:
        import json
        import socket
        import os
        def no_network(*args, **kwargs):
            raise RuntimeError('Network blocked during GUI test')
        socket.socket.connect = no_network
        test_mode=os.environ.get('TTS_TEST_MODE','auto')
        window.accel_mode.setCurrentIndex(window.accel_mode.findData(test_mode))
        if test_mode=='hybrid':
            window.settings_widgets['gpu_batch'].setValue(4)
        window.outdir.setText(str(test_root / 'audio'))
        from app.acceleration import BENCH_TEXTS
        window.editor.setPlainText('\n'.join(BENCH_TEXTS * int(os.environ.get('TTS_TEST_REPEAT','1'))))
        window.name.setText('Kiem_tra_EXE_UI')
        window.add_text()
        window.start()
        poll = QTimer(window)
        def check_finished():
            if window.worker and not window.worker.isRunning():
                poll.stop()
                window.refresh()
                jobs = window.store.jobs()
                success = all(j['status'] == 'DONE' for j in jobs)
                window.tabs.setCurrentIndex(2 if success else 3)
                window.grab().save(str(ROOT / 'logs/ui-e2e.png'))
                (ROOT / 'logs/ui-e2e-result.json').write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding='utf-8')
                manager=window.engine.manager
                diagnostics=dict(requested_mode=test_mode,success=success,
                                 task_percent=window.file_progress.value(),queue_percent=window.total_progress.value(),
                                 mode=manager.mode if manager else 'cpu',
                                 batch=manager.batch if manager else 1,
                                 workers=[lane.info for lane in manager.lanes] if manager else [])
                (ROOT/'logs'/f'ui-e2e-{test_mode}-diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
                window.outdir.setText(str(ROOT / 'output'))
                window.close()
                app.exit(0 if success else 2)
        poll.timeout.connect(check_finished)
        poll.start(300)
    sys.exit(app.exec())

if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception('Fatal startup error')
        sys.exit(1)
