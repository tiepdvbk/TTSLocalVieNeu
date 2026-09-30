import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from app import ui
from app.core import Store, DEFAULTS
from tests.test_core import FakeEngine


def test_delete_saved_clone_from_ui(tmp_path,monkeypatch):
    import json
    from app import voice_library as library
    app=QApplication.instance() or QApplication([])
    key='clone:'+'c'*32
    monkeypatch.setattr(library,'LIBRARY_DIR',tmp_path/'voices')
    library.LIBRARY_DIR.mkdir()
    profile=library.voice_file(key)
    profile.write_text(json.dumps(dict(id=key,name='My voice',voice=dict(speaker_emb=[.1]*192,codes=[[1]*16]*10))),encoding='utf-8')
    profile.with_suffix('.wav').write_bytes(b'audio')
    monkeypatch.setattr(ui,'Store',lambda:Store(tmp_path))
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    monkeypatch.setattr(ui,'load_config',lambda:DEFAULTS|dict(voice=key,auto_preview=False))
    monkeypatch.setattr(ui.QMessageBox,'question',lambda *a,**k:ui.QMessageBox.Yes)
    w=ui.Window()
    try:
        assert w.clone_saved.findData(key)>=0
        w.delete_clone()
        assert not profile.exists() and not profile.with_suffix('.wav').exists()
        assert w.clone_saved.findData(key)<0 and w.voice.findData(key)<0
        assert w.voice.currentData() and not w.voice.currentData().startswith('clone:')
    finally:w.close()


def test_clone_modes_and_emotion_controls(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    key='clone:'+'a'*32
    monkeypatch.setattr(ui,'Store',lambda:Store(tmp_path))
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    monkeypatch.setattr(ui,'voices',lambda:([('Thiện Minh','Preset'),(key,'Clone')],'Thiện Minh'))
    monkeypatch.setattr(ui,'load_config',lambda:DEFAULTS|dict(voice=key,auto_preview=False,voice_profile=False))
    w=ui.Window()
    try:
        assert w.clone_quality.currentData()=='stable'
        assert not w.settings_widgets['temperature'].isEnabled()
        w.clone_quality.setCurrentIndex(w.clone_quality.findData('manual'))
        assert w.settings_widgets['temperature'].isEnabled()
        w.settings_widgets['temperature'].setValue(.85)
        w.voice.setCurrentIndex(w.voice.findData('Thiện Minh'))
        assert not w.clone_quality.isEnabled()
        assert w.settings_widgets['temperature'].isEnabled()
        w.insert_emotion(2)
        assert '[thở dài]' in w.editor.toPlainText()
        assert w.clone_prepare.isChecked() and w.clone_denoise.isChecked()
        import json
        metadata=tmp_path/'voice.json'
        metadata.write_text(json.dumps(dict(name='Clone',seconds=6.,quality=dict(seconds=8.))),encoding='utf-8')
        monkeypatch.setattr(ui,'voice_file',lambda _:metadata)
        w.voice.setCurrentIndex(w.voice.findData(key))
        w.reuse_clone()
        assert w.clone_length.value()==8.  # reuse the original crop, not its trimmed duration
    finally: w.close()


def test_srt_and_voice_controls(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui,'Store',lambda:Store(tmp_path))
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    monkeypatch.setattr(ui,'load_config',lambda:DEFAULTS|dict(output_dir=str(tmp_path/'out'),auto_preview=False))
    w=ui.Window()
    try:
        w.set_rhythm(2)
        assert w.settings()['pause_custom'] and w.settings()['pause_newline']==.7
        assert w.settings_widgets['pause_comma'].isEnabled()
        w.set_rhythm(1)
        assert not w.settings_widgets['pause_comma'].isEnabled()
        w.settings_widgets['export_srt'].setChecked(True)
        w.srt_mode.setCurrentIndex(w.srt_mode.findData('timeline'))
        w.editor_srt.setChecked(True)
        w.editor.setPlainText('1\n00:00:01,000 --> 00:00:02,000\nXin chào.')
        w.name.setText('Subtitle')
        w.add_text()
        import json
        settings=json.loads(w.store.jobs()[0]['settings'])
        assert settings['_srt_cues'][0]['start']==1
        assert settings['srt_mode']=='timeline' and settings['export_srt']
        assert w.tabs.tabText(5)=='Clone giọng'
    finally:
        w.close()

def test_add_run_history_and_persistence(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Store', lambda: Store(tmp_path))
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    monkeypatch.setattr(ui, 'load_config', lambda: DEFAULTS | {'output_dir': str(tmp_path/'out')})
    window = ui.Window()
    fake = FakeEngine()
    fake.close = lambda: None
    window.engine = fake
    window.editor.setPlainText('Xin chào Việt Nam.\nĐây là bài kiểm tra giao diện.')
    window.name.setText('Kiểm tra UI')
    window.add_text()
    assert window.queue.rowCount() == 1
    window.start()
    for _ in range(250):
        app.processEvents()
        if not window.worker.isRunning():
            break
        QTest.qWait(20)
    assert not window.worker.isRunning()
    window.refresh()
    assert window.queue.rowCount() == 0
    assert window.history.rowCount() == 1
    assert window.total_progress.value() == 100
    assert fake.calls == 2
    window.close()
    restored = ui.Window()
    assert restored.history.rowCount() == 1
    restored.close()
from pathlib import Path
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QMessageBox
from app.core import Runner


def test_voice_profiles_latest_preview_and_manual_settings(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Store', lambda: Store(tmp_path))
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    monkeypatch.setattr(ui, 'load_config', lambda: DEFAULTS | {'voice':'Minh Quân','speed':1.2,'output_dir':str(tmp_path/'out')})
    w = ui.Window()
    calls=[]
    monkeypatch.setattr(w, 'make_preview', lambda voice,force=False: calls.append(voice))
    try:
        assert w.voice.currentData() == 'Hải Đăng'
        assert w.settings()['speed'] == 1.2  # opening the app preserves saved custom values
        w.voice.setCurrentIndex(w.voice.findData('Thiền Tâm Đức'))
        assert w.settings()['speed'] == .95
        assert w.settings()['gap'] == .4
        w.voice.setCurrentIndex(w.voice.findData('Thái Sơn'))
        QTest.qWait(450)
        app.processEvents()
        assert calls == ['Thái Sơn']
        w.settings_widgets['voice_profile'].setChecked(False)
        w.settings_widgets['auto_preview'].setChecked(False)
        w.settings_widgets['speed'].setValue(1.15)
        w.voice.setCurrentIndex(w.voice.findData('Ngọc Linh'))
        QTest.qWait(450)
        assert w.settings()['speed'] == 1.15
        assert calls == ['Thái Sơn']
        w.settings_widgets['auto_preview'].setChecked(True)
        w.voice.setCurrentIndex(w.voice.findData('Thiện Minh'))
        w.stop_preview()
        QTest.qWait(450)
        assert calls == ['Thái Sơn']
    finally:
        w.close()


def test_controls_progress_history_and_theme(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, 'Store', lambda: Store(tmp_path))
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    monkeypatch.setattr(ui, 'load_config', lambda: DEFAULTS | {'output_dir': str(tmp_path/'out')})
    w = ui.Window()
    try:
        for mode in ('cpu','cpu2','hybrid','gpu','auto'):
            w.accel_mode.setCurrentIndex(w.accel_mode.findData(mode))
            assert w.preset.isEnabled() == (mode in ('cpu','cpu2','hybrid'))
            assert w.settings_widgets['gpu_batch'].isEnabled() == (mode in ('gpu','hybrid'))
        w.accel_mode.setCurrentIndex(w.accel_mode.findData('hybrid'))
        w.settings_widgets['threads'].setValue(3)
        w.settings_widgets['gpu_batch'].setValue(32)
        assert w.settings()['threads'] == 3
        w.settings_widgets['threads'].setValue(128)
        assert w.settings()['threads'] == 128
        assert w.preset.currentData() == 0
        w.settings_widgets['threads'].setValue(3)
        assert w.settings()['gpu_batch'] == 32
        for key, value in dict(speed=1.25,volume=135,gap=.65,chunk_size=300,temperature=.7,top_k=40,top_p=.9,repetition_penalty=1.4,retries=3).items():
            w.settings_widgets[key].setValue(value)
            assert w.settings()[key] == value
        for fmt in ('wav','mp3','flac','m4a'):
            w.format.setCurrentText(fmt)
            assert w.settings()['format'] == fmt
        for theme in ('dark','light'):
            w.theme.setCurrentIndex(w.theme.findData(theme))
            box = QMessageBox(w)
            assert box.palette().color(QPalette.Window).lightness() != box.palette().color(QPalette.WindowText).lightness()
            assert w.voice.view().palette().color(QPalette.Base).lightness() != w.voice.view().palette().color(QPalette.Text).lightness()
        assert 'VieNeu' not in w.windowTitle()
        s = DEFAULTS | {'output_dir': str(tmp_path/'out'), 'accel_mode':'cpu'}
        old = w.store.add('Old sample.', 'old', s)
        Runner(w.store, FakeEngine()).run([old])
        output = Path(w.store.job(old)['output'])
        w.refresh()
        assert w.total_progress.value() == w.file_progress.value() == 0
        jid = w.store.add('New sample.', 'new', s)
        w.refresh()
        assert w.total_progress.value() == 0
        w.run_ids = [jid]
        with w.store.connect() as db:
            db.execute("UPDATE chunks SET status='DONE' WHERE job_id=?",(jid,))
            db.execute("UPDATE jobs SET status='RUNNING' WHERE id=?",(jid,))
        w.refresh()
        assert 0 < w.total_progress.value() < 100
        assert w.file_progress.value() < 100
        with w.store.connect() as db:
            db.execute("UPDATE jobs SET status='DONE' WHERE id=?",(jid,))
        w.refresh()
        assert w.total_progress.value() == w.file_progress.value() == 100
        for row in range(w.history.rowCount()):
            if w.history.item(row,0).data(ui.Qt.UserRole)==old:
                w.history.selectRow(row)
        w.remove_history()
        assert output.exists()
        assert old not in [j['id'] for j in w.store.jobs()]
        new = w.store.add('Another sample.', 'another', s)
        w.refresh()
        assert w.total_progress.value() == w.file_progress.value() == 0
    finally:
        w.close()
