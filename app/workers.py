"""Background QThreads used by the main window."""
from __future__ import annotations
import logging
import threading
from PySide6.QtCore import QThread, Signal
from .core import Runner, export_audio
from .acceleration import hardware_info, create_preview, benchmark, create_tuned_preview
from .voice_library import create_voice, inspect_sample


class Worker(QThread):
    event = Signal(str, str)
    def __init__(self, store, engine, ids, export=None, execution=None):
        super().__init__()
        self.runner = Runner(store, engine, lambda a, b: self.event.emit(a, b))
        self.ids = ids
        self.export = export
        self.runner.execution = execution or {}

    def run(self):
        try:
            if self.export:
                path = export_audio(self.runner.store, self.ids[0], self.export)
                self.event.emit('log', 'Đã xuất bản mới: ' + str(path))
                self.event.emit('export', str(path))
            else:
                self.runner.run(self.ids)
        except Exception as exc:
            logging.exception('Worker failed')
            self.event.emit('log', 'Lỗi: ' + str(exc))

class TaskWorker(QThread):
    event = Signal(str,str)
    result = Signal(object)
    def __init__(self,task,engine=None,voice_names=None,force=False,request=None):
        super().__init__()
        self.task,self.engine,self.names,self.force=task,engine,voice_names or [],force
        self.cancel=threading.Event()
        self.request=request or {}
    def run(self):
        try:
            if self.task=='hardware':
                self.result.emit(hardware_info())
            elif self.task=='benchmark':
                self.engine.close()
                result=benchmark(lambda a,b:self.event.emit(a,b),self.cancel.is_set)
                self.result.emit(result)
            elif self.task=='clone':
                self.event.emit('log','Đang tạo giọng cá nhân từ đoạn mẫu…')
                self.result.emit(create_voice(self.engine,**self.request))
            elif self.task=='inspect_clone':
                self.result.emit(inspect_sample(**self.request))
            elif self.task=='reading_preview':
                from .speech_plan import reading_preview
                self.result.emit(reading_preview(**self.request))
            elif self.task=='tuned':
                self.event.emit('tuned_preview',str(create_tuned_preview(self.engine,self.request)))
            else:
                for i,voice in enumerate(self.names):
                    if self.cancel.is_set():
                        break
                    self.event.emit('log',f'Mẫu giọng {i+1}/{len(self.names)}: {voice}')
                    path=create_preview(self.engine,voice,self.force)
                    self.event.emit('samples',str(path))
                    if len(self.names)==1:
                        self.event.emit('preview',str(path))
        except Exception as exc:
            logging.exception('Auxiliary task failed')
            self.event.emit('log','Lỗi: '+str(exc))
            self.event.emit('error',str(exc))
