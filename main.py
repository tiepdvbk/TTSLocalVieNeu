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
    from app.testmodes import FLAGS
    if any(flag in sys.argv for flag in FLAGS):
        # Test runs must never alter the user's voice, output folder or settings.
        window.save=lambda:None
    window.show()
    from app import testmodes
    testmodes.install(window, app, test_root if e2e else None)
    sys.exit(app.exec())

if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception('Fatal startup error')
        sys.exit(1)
