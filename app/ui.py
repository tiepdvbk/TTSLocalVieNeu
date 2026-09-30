from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import time
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QColor, QPalette
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QPlainTextEdit, QFileDialog, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QFormLayout, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QTabWidget, QMessageBox, QScrollArea, QAbstractItemView,
    QSplitter, QSizePolicy)
from .core import ROOT, Store, Runner, Engine, load_config, atomic_json, read_text, voices, export_audio
from .acceleration import LABELS, hardware_info, recommendation, preview_path, create_preview, valid_preview, benchmark, create_tuned_preview
from .voice_profiles import canonical_voice, profile, profile_settings, NARRATORS
from .voice_library import display_name, create_voice, inspect_sample, voice_file
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import threading

STATUS = {'PENDING': 'Đang chờ', 'RUNNING': 'Đang đọc', 'PAUSED': 'Tạm dừng',
          'STOPPED': 'Đã dừng', 'DONE': 'Hoàn tất', 'FAILED': 'Có lỗi'}

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

def button(text, callback, primary=False):
    b = QPushButton(text)
    if primary:
        b.setObjectName('primary')
    b.clicked.connect(callback)
    return b

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.store = Store()
        self.store.recover()
        self.config = load_config()
        self.engine = Engine()
        self.worker = None
        self.aux = None
        self.pending_preview = None
        self.requested_preview = None
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(350)
        self.preview_timer.timeout.connect(self.preview_pending)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.9)
        self.player.setAudioOutput(self.audio_output)
        self.player.errorOccurred.connect(lambda error,message:self.append_log('Phát audio: '+message) if message else None)
        self.closing = False
        self.started = None
        self.run_ids = []
        self.rows = []
        self.setWindowTitle('TTS Studio • CPU / GPU')
        self.resize(1260, 860)
        self.setMinimumSize(960, 680)
        self.setAcceptDrops(True)
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)
        layout.setContentsMargins(24, 20, 24, 16)
        header = QHBoxLayout()
        title = QLabel('<b>TTS Studio</b>')
        title.setObjectName('title')
        header.addWidget(title)
        header.addStretch()
        chip = QLabel('●  OFFLINE  /  CPU + GPU')
        chip.setObjectName('chip')
        header.addWidget(chip)
        layout.addLayout(header)
        import psutil
        self.hardware = QLabel(f'{psutil.cpu_count(logical=False)} nhân / {psutil.cpu_count()} luồng  •  '
                              f'RAM {psutil.virtual_memory().total / 2**30:.1f} GB  •  FP32  •  Audio 48 kHz')
        self.hardware.setObjectName('muted')
        layout.addWidget(self.hardware)
        self.gpu_label = QLabel('Đang kiểm tra GPU…')
        self.gpu_label.setObjectName('muted')
        layout.addWidget(self.gpu_label)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)
        left = QWidget()
        l = QVBoxLayout(left)
        l.setContentsMargins(0, 12, 12, 0)
        self.tabs = QTabWidget()
        l.addWidget(self.tabs, 1)
        input_page = QWidget()
        inp = QVBoxLayout(input_page)
        self.name = QLineEdit()
        self.name.setPlaceholderText('Tên bài đọc, ví dụ: Chương 01')
        inp.addWidget(self.name)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText('Nhập hoặc dán nội dung tiếng Việt tại đây…\n\nBạn cũng có thể kéo file TXT / SRT hoặc thư mục vào cửa sổ.')
        inp.addWidget(self.editor)
        self.editor_srt=QCheckBox('Nội dung dán ở trên là SRT (có mốc thời gian)')
        inp.addWidget(self.editor_srt)
        self.emotion_cue=QComboBox()
        self.emotion_cue.addItems(['Chèn biểu cảm thử nghiệm…','[cười]','[thở dài]','[hắng giọng]'])
        self.emotion_cue.setToolTip('Chèn tại con trỏ. Thẻ thử nghiệm của model; ngữ điệu chủ yếu theo giọng mẫu.')
        self.emotion_cue.activated.connect(self.insert_emotion)
        inp.addWidget(self.emotion_cue)
        row = QHBoxLayout()
        self.count = QLabel('0 ký tự')
        self.editor.textChanged.connect(lambda: self.count.setText(f'{len(self.editor.toPlainText()):,} ký tự'))
        row.addWidget(self.count)
        row.addStretch()
        row.addWidget(button('Thêm bài vào hàng đợi', self.add_text, True))
        inp.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(button('+ Chọn TXT / SRT', self.pick_files))
        row.addWidget(button('+ Chọn thư mục', self.pick_folder))
        inp.addLayout(row)
        self.tabs.addTab(input_page, 'Soạn nội dung')
        self.queue_page = QWidget()
        q = QVBoxLayout(self.queue_page)
        self.queue = self.make_table()
        q.addWidget(self.queue)
        actions = QHBoxLayout()
        self.retry_btn = button('Tiếp tục / Thử lại', self.retry)
        self.remove_btn = button('Xóa tác vụ', self.remove)
        actions.addWidget(self.retry_btn)
        actions.addWidget(self.remove_btn)
        actions.addWidget(button('Xem chi tiết', self.details))
        q.addLayout(actions)
        self.tabs.addTab(self.queue_page, 'Hàng đợi')
        history = QWidget()
        h = QVBoxLayout(history)
        self.history = self.make_table()
        h.addWidget(self.history)
        row = QHBoxLayout()
        row.addWidget(button('Nghe audio', self.play))
        row.addWidget(button('Mở thư mục', self.open_selected_folder))
        self.export_btn = button('Xuất lại tốc độ / âm lượng', self.reexport)
        row.addWidget(self.export_btn)
        h.addLayout(row)
        self.delete_history_btn = button('Xóa mục đã chọn', self.remove_history)
        h.addWidget(self.delete_history_btn)
        h.addWidget(QLabel('Xóa lịch sử và cache; giữ nguyên file audio đã xuất.'))
        self.tabs.addTab(history, 'Lịch sử')
        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2500)
        log_layout.addWidget(self.log)
        log_layout.addWidget(button('Mở thư mục nhật ký', lambda: self.open_path(ROOT / 'logs')))
        self.tabs.addTab(log_page, 'Nhật ký')
        sample_page = QWidget()
        spl = QVBoxLayout(sample_page)
        intro = QLabel('Mẫu truyện ngắn dùng thiết lập kể chuyện gợi ý của từng giọng. Tạo lần đầu rồi lưu trên máy; những lần sau nghe ngay. 10 giọng kể / đọc truyện được xếp đầu.')
        intro.setWordWrap(True)
        spl.addWidget(intro)
        self.sample_table = QTableWidget(0,3)
        self.sample_table.setHorizontalHeaderLabels(['Giọng','Mô tả','Mẫu local'])
        self.sample_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch)
        self.sample_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents)
        self.sample_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeToContents)
        self.sample_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sample_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sample_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sample_table.verticalHeader().hide()
        self.sample_table.doubleClicked.connect(lambda _:self.preview_selected())
        spl.addWidget(self.sample_table)
        row=QHBoxLayout()
        row.addWidget(button('▶ Nghe giọng đã chọn',self.preview_selected,True))
        row.addWidget(button('■ Dừng phát',self.stop_preview))
        self.all_samples_btn=button('Tạo / bổ sung tất cả mẫu',self.all_previews)
        row.addWidget(self.all_samples_btn)
        spl.addLayout(row)
        row=QHBoxLayout()
        self.regen_sample_btn=button('Tạo lại mẫu đã chọn',lambda:self.preview_selected(True))
        row.addWidget(self.regen_sample_btn)
        row.addWidget(button('Mở thư mục mẫu',self.open_samples))
        spl.addLayout(row)
        self.tabs.addTab(sample_page,'Nghe thử giọng')
        clone_page=QWidget()
        clone_form=QFormLayout(clone_page)
        clone_note=QLabel('Tạo giọng cá nhân offline từ một người nói rõ, ít nhạc nền. Mẫu 3–8 giây thường phù hợp; có thể chọn tối đa 8 giây. Không cần gõ lại lời mẫu. Chất giọng và ngữ điệu phụ thuộc đoạn thu.')
        clone_note.setWordWrap(True)
        clone_form.addRow(clone_note)
        self.clone_name=QLineEdit(); self.clone_name.setPlaceholderText('Tên giọng của bạn')
        self.clone_file=QLineEdit(); self.clone_file.setReadOnly(True)
        clone_form.addRow('Tên giọng',self.clone_name)
        clone_form.addRow('File mẫu',self.clone_file)
        clone_form.addRow(button('Chọn audio mẫu…',self.pick_clone))
        self.clone_start=QDoubleSpinBox(); self.clone_start.setRange(0,86400); self.clone_start.setSuffix(' s')
        self.clone_length=QDoubleSpinBox(); self.clone_length.setRange(3,8); self.clone_length.setValue(8); self.clone_length.setSuffix(' s')
        clone_form.addRow('Bắt đầu tại',self.clone_start)
        clone_form.addRow('Lấy đoạn dài',self.clone_length)
        self.clone_denoise=QCheckBox('Lọc nhiễu mẫu bằng model (chậm hơn)')
        self.clone_denoise.setChecked(True)
        clone_form.addRow(self.clone_denoise)
        self.clone_prepare=QCheckBox('Chuẩn hóa mức âm và giảm im lặng ở hai đầu')
        self.clone_prepare.setChecked(True)
        clone_form.addRow(self.clone_prepare)
        clone_form.addRow(button('Nghe và kiểm tra đoạn mẫu',self.inspect_clone))
        clone_form.addRow(button('Lấy mẫu từ giọng cá nhân đang chọn',self.reuse_clone))
        self.clone_report=QLabel('Chọn đoạn nói liền mạch, rõ dấu tiếng Việt và có ngữ điệu bạn muốn. Với bản thu đã sạch, thử cả bật / tắt lọc nhiễu để so sánh.')
        self.clone_report.setWordWrap(True)
        clone_form.addRow(self.clone_report)
        self.clone_btn=button('Tạo và lưu giọng cá nhân',self.start_clone,True)
        clone_form.addRow(self.clone_btn)
        clone_form.addRow(QLabel('Giọng được lưu riêng trong data/voices và xuất hiện trong danh sách Giọng.'))
        self.tabs.addTab(clone_page,'Clone giọng')
        self.progress_label = QLabel('Sẵn sàng • Thêm nội dung rồi nhấn Bắt đầu')
        l.addWidget(self.progress_label)
        self.file_progress = QProgressBar()
        self.file_progress.setFormat('Tác vụ hiện tại: %p%')
        l.addWidget(self.file_progress)
        self.total_progress = QProgressBar()
        self.total_progress.setFormat('Toàn bộ hàng đợi: %p%')
        l.addWidget(self.total_progress)
        self.stats = QLabel('Tiến trình được lưu sau mỗi đoạn.')
        self.stats.setObjectName('muted')
        l.addWidget(self.stats)
        controls = QHBoxLayout()
        self.start_btn = button('▶  Bắt đầu', self.start, True)
        self.pause_btn = button('Tạm dừng', self.pause)
        self.stop_btn = button('Dừng', self.stop)
        controls.addWidget(self.start_btn, 2)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.stop_btn)
        l.addLayout(controls)
        splitter.addWidget(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(355)
        panel = QWidget()
        form_layout = QVBoxLayout(panel)
        form_layout.setContentsMargins(16, 16, 12, 16)
        self.settings_widgets = {}
        group = QGroupBox('GIỌNG ĐỌC')
        form = QFormLayout(group)
        self.voice = QComboBox()
        items, default = voices()
        for key, description in items:
            self.voice.addItem(('★ ' if key in NARRATORS else 'Cá nhân • ' if key.startswith('clone:') else '') + display_name(key), key)
            self.voice.setItemData(self.voice.count()-1, description, Qt.ToolTipRole)
        self.voice.setCurrentIndex(max(0, self.voice.findData(canonical_voice(self.config['voice'] or default))))
        form.addRow('Giọng', self.voice)
        preview_buttons=QHBoxLayout()
        preview_buttons.addWidget(button('▶ Nghe thử',self.preview_current))
        preview_buttons.addWidget(button('■ Dừng',self.stop_preview))
        form.addRow(preview_buttons)
        tuned_button=button('Nghe với chỉnh hiện tại',self.preview_tuned)
        tuned_button.setToolTip('Bôi đen một câu trong Soạn nội dung để nghe câu đó. Không bôi đen: nghe mẫu truyện mặc định.')
        form.addRow(tuned_button)
        self.clone_quality=QComboBox()
        for label,key in [('Rõ và ổn định','stable'),('Tự nhiên theo mẫu','natural'),('Tự chỉnh nâng cao','manual')]:
            self.clone_quality.addItem(label,key)
        self.clone_quality.setCurrentIndex(max(0,self.clone_quality.findData(self.config.get('clone_quality','stable'))))
        form.addRow('Chế độ clone',self.clone_quality)
        self.clone_quality.currentIndexChanged.connect(self.sync_clone_controls)
        self.clone_quality_note=QLabel()
        self.clone_quality_note.setWordWrap(True)
        form.addRow(self.clone_quality_note)
        info = QLabel('★ Giọng kể / đọc truyện theo mô tả VieNeu. Thiết lập là gợi ý ban đầu, có thể chỉnh theo sở thích. Mẫu nghe dùng thiết lập gợi ý; ngữ điệu theo giọng gốc.')
        info.setObjectName('muted')
        info.setWordWrap(True)
        form.addRow(info)
        self.add_number(form, 'Tốc độ', 'speed', 0.5, 2.0, 0.05, ' ×')
        self.add_number(form, 'Âm lượng', 'volume', 0, 200, 5, ' %')
        self.add_number(form, 'Nghỉ giữa đoạn', 'gap', 0.0, 3.0, 0.05, ' s')
        for key, label in [('voice_profile','Tự áp dụng thiết lập khi đổi giọng'), ('auto_preview','Tự nghe mẫu truyện khi đổi giọng')]:
            cb = QCheckBox(label)
            cb.setChecked(self.config.get(key, True))
            self.settings_widgets[key] = cb
            form.addRow(cb)
        self.profile_label = QLabel()
        self.profile_label.setWordWrap(True)
        form.addRow(self.profile_label)
        self.settings_widgets['voice_profile'].toggled.connect(lambda enabled: self.apply_voice_profile() if enabled else None)
        self.settings_widgets['auto_preview'].toggled.connect(lambda enabled: self.stop_preview() if not enabled else None)
        self.voice.currentIndexChanged.connect(self.voice_changed)
        self.update_profile_label()
        if not self.config['voice'] and self.config.get('voice_profile',True):
            self.apply_voice_profile()
        form_layout.addWidget(group)
        group=QGroupBox('NHỊP NGHỈ & SẮC GIỌNG')
        form=QFormLayout(group)
        rhythm=QComboBox()
        rhythm.addItems(['Mặc định / tự chỉnh','Tự nhiên (model tự ngắt)','Kể truyện','Nhanh gọn'])
        form.addRow('Thiết lập nhanh',rhythm)
        for key,label in [('pause_custom','Tự chỉnh nghỉ theo dấu câu'),('use_ref_codes','Bám ngữ điệu của giọng mẫu'),
                          ('normalize_volume','Cân độ lớn audio'),('trim_silence','Rút im lặng đầu / cuối đoạn')]:
            cb=QCheckBox(label); cb.setChecked(self.config[key]); self.settings_widgets[key]=cb; form.addRow(cb)
        self.add_number(form,'Sau dấu phẩy / ; / :','pause_comma',0.,3.,.05,' s')
        self.add_number(form,'Sau dấu chấm / ? / !','pause_sentence',0.,5.,.05,' s')
        self.add_number(form,'Sau xuống dòng','pause_newline',0.,10.,.05,' s')
        self.add_number(form,'Cao độ','pitch',-6.,6.,.5,' bán âm')
        rhythm.currentIndexChanged.connect(self.set_rhythm)
        self.settings_widgets['pause_custom'].toggled.connect(self.sync_pause_controls)
        self.sync_pause_controls()
        note=QLabel('Khoảng nghỉ là phần im lặng thêm vào tiếng model đã tạo. Tự chỉnh dấu câu chia nhiều đoạn nhỏ hơn, có thể giảm tốc độ xử lý. Cao độ là hiệu ứng, không tạo ra giọng mới.')
        note.setWordWrap(True); form.addRow(note)
        form_layout.addWidget(group)
        group = QGroupBox('HIỆU NĂNG CPU / GPU')
        form = QFormLayout(group)
        self.accel_mode = QComboBox()
        for key,label in LABELS.items():
            self.accel_mode.addItem(label,key)
        self.accel_mode.setCurrentIndex(max(0,self.accel_mode.findData(self.config.get('accel_mode','auto'))))
        self.accel_mode.currentIndexChanged.connect(self.sync_acceleration_controls)
        form.addRow('Xử lý',self.accel_mode)
        self.add_number(form,'Nhóm GPU','gpu_batch',1,32,1,' đoạn')
        self.benchmark_btn=button('Đo tốc độ & chọn tự động',self.run_benchmark)
        form.addRow(self.benchmark_btn)
        self.preset = QComboBox()
        for label, value in [('Tiết kiệm • 2 luồng', 2), ('Cân bằng • 4 luồng', 4), ('Nhanh • 6 luồng', 6), ('Tùy chỉnh', 0)]:
            self.preset.addItem(label, value)
        preset_index=self.preset.findData(self.config['threads'])
        self.preset.setCurrentIndex(preset_index if preset_index>=0 else self.preset.count()-1)
        form.addRow('Chế độ CPU', self.preset)
        self.add_number(form, 'Luồng CPU', 'threads', 1, 2147483647, 1)
        self.settings_widgets['threads'].setKeyboardTracking(False)
        self.settings_widgets['threads'].setToolTip('Nhập trực tiếp số nguyên dương, không bị giới hạn bởi số nhân máy. Đây là số luồng cho mỗi worker CPU; nhiều hơn không luôn nhanh hơn. GPU thuần không sử dụng thông số này.')
        self.preset.currentIndexChanged.connect(lambda: self.settings_widgets['threads'].setValue(self.preset.currentData()) if self.preset.currentData() else None)
        self.settings_widgets['threads'].valueChanged.connect(self.sync_cpu_preset)
        self.add_number(form, 'Độ dài đoạn', 'chunk_size', 100, 400, 10, ' ký tự')
        info = QLabel('CPU kép dùng 2 worker, mỗi worker theo số luồng đã chọn. GPU thuần không dùng ONNX. Tự động khóa thông số theo kết quả đo; chọn CPU / GPU để chỉnh tay. Batch lớn cần đủ số đoạn và VRAM, không luôn nhanh hơn. Áp dụng khi Bắt đầu / Tiếp tục.')
        info.setWordWrap(True)
        info.setObjectName('muted')
        form.addRow(info)
        form_layout.addWidget(group)
        group = QGroupBox('ĐẦU RA')
        form = QFormLayout(group)
        self.srt_mode=QComboBox()
        self.srt_mode.addItem('Giữ mốc thời gian SRT','timeline')
        self.srt_mode.addItem('Đọc nối tiếp tự nhiên (bỏ mốc giờ)','continuous')
        self.srt_mode.setCurrentIndex(max(0,self.srt_mode.findData(self.config['srt_mode'])))
        form.addRow('Khi nhập SRT',self.srt_mode)
        self.add_number(form,'Co giọng SRT tối đa','srt_max_speed',1.,4.,.1,' ×')
        cb=QCheckBox('Xuất kèm SRT theo đoạn')
        cb.setChecked(self.config['export_srt']); self.settings_widgets['export_srt']=cb; form.addRow(cb)
        note=QLabel('SRT mặc định giữ thời lượng và mốc giờ gốc bằng cách chèn im lặng, co giọng khi cần. Chọn đọc nối tiếp sẽ bỏ khoảng trống giữa các mục, làm audio ngắn hơn. Nếu lời không vừa giới hạn co giọng sẽ báo lỗi, không cắt lời.')
        note.setWordWrap(True); form.addRow(note)
        self.format = QComboBox()
        self.format.addItems(['wav', 'mp3', 'flac', 'm4a'])
        self.format.setCurrentText(self.config['format'])
        form.addRow('Định dạng', self.format)
        self.outdir = QLineEdit(self.config['output_dir'])
        form.addRow(self.outdir)
        form.addRow(button('Chọn thư mục đầu ra…', self.pick_output))
        for key, label in [('same_folder', 'Lưu cạnh file TXT / SRT'), ('recursive', 'Quét cả thư mục con'),
                           ('preserve_tree', 'Giữ cấu trúc thư mục'), ('cleanup', 'Xóa cache khi hoàn tất')]:
            cb = QCheckBox(label)
            cb.setChecked(self.config[key])
            self.settings_widgets[key] = cb
            form.addRow(cb)
        note = QLabel('Tự thêm số khi trùng tên. Giữ cache để xuất lại mà không cần đọc lại.')
        note.setWordWrap(True)
        note.setObjectName('muted')
        form.addRow(note)
        form_layout.addWidget(group)
        group = QGroupBox('NÂNG CAO')
        form = QFormLayout(group)
        self.add_number(form, 'Temperature', 'temperature', 0.1, 1.5, 0.05)
        self.add_number(form, 'Top K', 'top_k', 1, 100, 1)
        self.add_number(form, 'Top P', 'top_p', 0.1, 1.0, 0.05)
        self.add_number(form, 'Chống lặp', 'repetition_penalty', 1.0, 2.0, 0.05)
        self.add_number(form, 'Số lần thử lại', 'retries', 0, 5, 1)
        self.theme = QComboBox()
        self.theme.addItem('Tối', 'dark')
        self.theme.addItem('Sáng', 'light')
        self.theme.setCurrentIndex(max(0, self.theme.findData(self.config['theme'])))
        self.theme.currentIndexChanged.connect(self.apply_theme)
        form.addRow('Giao diện', self.theme)
        form_layout.addWidget(group)
        form_layout.addWidget(button('Lưu thiết lập', self.save))
        form_layout.addWidget(button('Hướng dẫn sử dụng', lambda: self.open_path(ROOT / 'HUONG_DAN.html')))
        form_layout.addStretch()
        # Long personal voice names must not widen the entire settings sidebar.
        for combo in panel.findChildren(QComboBox):
            combo.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
            combo.setMinimumContentsLength(8)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        for form in panel.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)
        splitter.setSizes([840, 340])
        self.apply_theme()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self.refresh()
        self.refresh_samples()
        self.sync_acceleration_controls()
        self.sync_clone_controls()
        self.detect_worker=TaskWorker('hardware')
        self.detect_worker.result.connect(self.show_hardware)
        self.detect_worker.start()
        self.append_log('Sẵn sàng. Model được nạp khi bắt đầu đọc. Cấu hình áp dụng khi thêm tác vụ mới.')
        if any(j['status'] in ('PAUSED', 'STOPPED') for j in self.store.jobs()):
            self.append_log('Có tác vụ chưa hoàn tất. Chọn tác vụ rồi nhấn Tiếp tục / Thử lại.')
            self.tabs.setCurrentIndex(1)

    def add_number(self, form, label, key, lo, hi, step, suffix=''):
        control = QDoubleSpinBox() if isinstance(lo, float) else QSpinBox()
        control.setRange(lo, hi)
        control.setSingleStep(step)
        control.setValue(self.config[key])
        control.setSuffix(suffix)
        self.settings_widgets[key] = control
        form.addRow(label, control)

    def settings(self):
        s = self.config.copy()
        for key, widget in self.settings_widgets.items():
            s[key] = widget.isChecked() if isinstance(widget, QCheckBox) else widget.value()
        s.update(voice=self.voice.currentData(), format=self.format.currentText(),
                 output_dir=self.outdir.text().strip() or str(ROOT / 'output'), theme=self.theme.currentData(),
                 accel_mode=self.accel_mode.currentData(),srt_mode=self.srt_mode.currentData(),clone_quality=self.clone_quality.currentData())
        return s

    def save(self):
        self.config = self.settings()
        atomic_json(ROOT / 'data/config.json', self.config)
        self.statusBar().showMessage('Đã lưu thiết lập', 4000)

    def apply_theme(self):
        dark = self.theme.currentData() == 'dark'
        bg, panel, field, fg, muted, border = ('#10151f', '#18212e', '#111a27', '#e5edf8', '#91a3b8', '#2c3b50') if dark else ('#edf2f8', '#ffffff', '#f7f9fc', '#17263b', '#60738c', '#d3ddeb')
        app = QApplication.instance()
        app.setStyle('Fusion')
        palette = QPalette()
        for role, color in [(QPalette.Window,bg),(QPalette.WindowText,fg),(QPalette.Base,field),(QPalette.AlternateBase,panel),(QPalette.Text,fg),(QPalette.Button,panel),(QPalette.ButtonText,fg),(QPalette.ToolTipBase,panel),(QPalette.ToolTipText,fg),(QPalette.Highlight,'#2468ce'),(QPalette.HighlightedText,'#ffffff')]:
            palette.setColor(role,QColor(color))
        app.setPalette(palette)
        self.setStyleSheet(f'''
            QDialog, QMessageBox {{ background-color: {bg}; color: {fg}; }}
            QComboBox QAbstractItemView {{ background-color: {field}; color: {fg}; selection-background-color: #2468ce; selection-color: white; }}
            QComboBox QAbstractItemView::item {{ min-height: 24px; }}
            QWidget {{ font-family: 'Segoe UI'; font-size: 13px; color: {fg}; }}
            QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {bg}; }}
            QGroupBox {{ background: {panel}; border: 1px solid {border}; border-radius: 10px; margin-top: 18px; padding: 16px 10px 10px; font-weight: 600; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {muted}; }}
            QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {field}; border: 1px solid {border}; border-radius: 6px; padding: 7px; selection-background-color: #247cdd; }}
            QPushButton {{ background: {panel}; border: 1px solid {border}; border-radius: 7px; padding: 9px 12px; }}
            QPushButton:hover {{ border-color: #448de2; }}
            QPushButton:disabled {{ color: {muted}; }}
            QComboBox:disabled, QSpinBox:disabled {{ color: {muted}; background: {bg}; }}
            QPushButton#primary {{ background: #2468ce; color: white; border: 0; font-weight: 600; }}
            QPushButton#primary:hover {{ background: #317ce9; }}
            QLabel#title {{ font-size: 27px; }}
            QLabel#muted {{ color: {muted}; font-size: 12px; }}
            QLabel#chip {{ color: #40cba9; background: {panel}; border-radius: 10px; padding: 8px 12px; font-weight: 600; }}
            QTabWidget::pane {{ border: 1px solid {border}; background: {panel}; border-radius: 8px; }}
            QTabBar::tab {{ padding: 12px 15px; background: {bg}; color: {muted}; }}
            QTabBar::tab:selected {{ color: {fg}; border-bottom: 3px solid #4089ed; }}
            QTableWidget {{ background: {field}; alternate-background-color: {panel}; border: 0; gridline-color: {border}; selection-background-color: #254d7d; }}
            QHeaderView::section {{ background: {panel}; padding: 10px 6px; border: 0; color: {muted}; }}
            QProgressBar {{ border: 1px solid {border}; border-radius: 6px; background: {field}; text-align: center; height: 22px; }}
            QProgressBar::chunk {{ background: #267cce; border-radius: 5px; }}
            QScrollArea {{ border: 0; }}
            QToolTip {{ background: {panel}; color: {fg}; border: 1px solid {border}; }}
        ''')

    def make_table(self):
        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(['Tác vụ', 'Trạng thái', 'Đoạn', 'Audio'])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 4):
            table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        table.verticalHeader().hide()
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        return table

    def fill_table(self, table, jobs):
        current = table.item(table.currentRow(), 0)
        selected = current.data(Qt.UserRole) if current else None
        table.setRowCount(len(jobs))
        for i, j in enumerate(jobs):
            values = [j['name'], STATUS[j['status']], f'{j["completed"] or 0}/{j["total"]}', f'{j["duration"]:.1f}s']
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, j['id'])
                item.setToolTip(j['error'] or j['output'])
                if col == 1:
                    item.setForeground(QColor('#43bd98' if j['status'] == 'DONE' else '#e09d53' if j['status'] == 'FAILED' else '#699fdc'))
                table.setItem(i, col, item)
            if j['id'] == selected:
                table.selectRow(i)

    def refresh(self):
        jobs = self.store.jobs()
        self.fill_table(self.queue, [j for j in jobs if j['status'] != 'DONE'])
        self.fill_table(self.history, [j for j in jobs if j['status'] == 'DONE'])
        running = self.busy()
        self.start_btn.setEnabled(not running)
        self.retry_btn.setEnabled(not running)
        self.remove_btn.setEnabled(not running)
        self.export_btn.setEnabled(not running)
        self.delete_history_btn.setEnabled(not running)
        self.all_samples_btn.setEnabled(not running)
        self.regen_sample_btn.setEnabled(not running)
        self.benchmark_btn.setEnabled(not running)
        self.pause_btn.setEnabled(bool(self.worker and self.worker.isRunning()))
        self.stop_btn.setEnabled(running)
        if running and ((self.worker and self.worker.isRunning() and self.worker.export) or (self.aux and self.aux.isRunning())):
            self.file_progress.setRange(0, 0)
            self.progress_label.setText('Đang xuất lại audio…' if self.worker and self.worker.isRunning() else 'Đang đo tốc độ / tạo mẫu giọng… Xem chi tiết tại Nhật ký.')
            return
        self.file_progress.setRange(0, 100)
        pending = [j for j in jobs if j['status'] != 'DONE']
        if not running and any(j['id'] not in self.run_ids for j in pending):
            self.run_ids = []
        scope = [j for j in jobs if j['id'] in self.run_ids] if self.run_ids else pending
        total = sum(j['total'] + 1 for j in scope)
        completed = sum((j['completed'] or 0) + (j['status'] == 'DONE') for j in scope)
        self.total_progress.setValue(int(100 * completed / total) if total else 0)
        active = next((j for j in scope if j['status'] == 'RUNNING'), None)
        current = active or next((j for j in scope if j['status'] != 'DONE'), None) or (scope[-1] if scope else None)
        self.file_progress.setValue(int(100 * ((current['completed'] or 0) + (current['status']=='DONE')) / (current['total']+1)) if current else 0)
        if active:
            self.progress_label.setText(f'{active["name"]}  •  {active["completed"]}/{active["total"]} đoạn')
            if active['completed'] == active['total']:
                self.progress_label.setText(active['name'] + ' • Đang ghép và xuất audio…')
            chunks = [c for c in self.store.chunks(active['id']) if c['status'] == 'DONE'][-12:]
            avg = sum(c['elapsed'] for c in chunks) / len(chunks) if chunks else 0
            eta = avg * (active['total'] - active['completed'])
            rate = active['duration'] / active['elapsed'] if active['elapsed'] else 0
            self.stats.setText(f'Đã tạo {active["duration"]:.0f}s audio  •  {rate:.2f}× thời gian thực  •  Còn khoảng {eta/60:.1f} phút' if chunks else 'Đang nạp model / tạo đoạn đầu tiên…')
        elif not running:
            failed = sum(j['status'] == 'FAILED' for j in jobs)
            self.progress_label.setText(f'{len(jobs)} tác vụ  •  {sum(j["status"] == "DONE" for j in jobs)} hoàn tất  •  {failed} lỗi')
            self.stats.setText('Đã lưu tiến trình. Giọng áp dụng khi thêm mới; hiệu năng áp dụng khi Bắt đầu.')

    def append_log(self, text):
        self.log.appendPlainText(time.strftime('%H:%M:%S') + '  ' + text)
        logging.info(text)

    def selected(self, history=False):
        table = self.history if history else self.queue
        item = table.item(table.currentRow(), 0)
        if not item:
            raise ValueError('Hãy chọn một tác vụ trong bảng.')
        return item.data(Qt.UserRole)

    def error(self, exc):
        QMessageBox.warning(self, 'Thông báo', str(exc))

    def add_text(self):
        try:
            self.save()
            self.store.add(self.editor.toPlainText(), self.name.text().strip() or time.strftime('Bai_doc_%Y%m%d_%H%M%S'), self.settings(),is_srt=self.editor_srt.isChecked())
            self.tabs.setCurrentIndex(1)
            self.refresh()
        except Exception as exc:
            self.error(exc)

    def add_paths(self, paths):
        self.save()
        count = 0
        errors = []
        seen = set()
        for path in map(Path, paths):
            scan=path.rglob if self.config['recursive'] else path.glob
            files = sorted([*scan('*.txt'),*scan('*.srt')]) if path.is_dir() else [path]
            for file in files:
                if file.suffix.lower() not in ('.txt','.srt') or str(file.resolve()).casefold() in seen:
                    continue
                seen.add(str(file.resolve()).casefold())
                try:
                    self.store.add(read_text(file), file.name, self.settings(), file,
                                   file.relative_to(path) if path.is_dir() else None)
                    count += 1
                except Exception as exc:
                    errors.append(f'{file.name}: {exc}')
        self.append_log(f'Đã thêm {count} file TXT / SRT.')
        self.tabs.setCurrentIndex(1)
        self.refresh()
        if errors:
            self.error('\n'.join(errors[:10]))

    def pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Chọn văn bản / phụ đề', '', 'Văn bản và phụ đề (*.txt *.srt)')
        if paths:
            self.add_paths(paths)

    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Chọn thư mục TXT')
        if folder:
            self.add_paths([folder])

    def pick_output(self):
        folder = QFileDialog.getExistingDirectory(self, 'Chọn thư mục đầu ra', self.outdir.text())
        if folder:
            self.outdir.setText(folder)

    def launch(self, ids, export=None):
        if self.busy():
            return
        if not ids:
            self.error('Chưa có tác vụ chờ. Hãy thêm nội dung hoặc chọn tác vụ để tiếp tục.')
            return
        self.save()
        self.run_ids = list(ids)
        self.worker = Worker(self.store, self.engine, ids, export, self.settings())
        self.worker.event.connect(self.on_event)
        self.worker.finished.connect(self.finished)
        self.started = time.monotonic()
        self.worker.start()
        self.refresh()

    def start(self):
        self.launch([j['id'] for j in self.store.jobs() if j['status'] in ('PENDING', 'PAUSED', 'STOPPED')])

    def retry(self):
        try:
            self.launch([self.selected()])
        except Exception as exc:
            self.error(exc)

    def pause(self):
        if self.worker:
            self.worker.runner.pause.set()
            self.append_log('Sẽ tạm dừng sau đoạn hiện tại. Các đoạn đã tạo được giữ lại.')

    def stop(self):
        if self.aux and self.aux.isRunning():
            self.aux.cancel.set()
            self.append_log('Sẽ dừng sau mẫu / nhóm hiện tại.')
            return
        if self.worker:
            self.worker.runner.stop.set()
            self.append_log('Sẽ dừng an toàn sau đoạn hiện tại. Có thể tiếp tục sau.')

    def on_event(self, kind, value):
        if kind == 'log':
            self.append_log(value)
        elif kind == 'export':
            self.open_path(Path(value).parent)
        elif kind == 'preview':
            if self.requested_preview and Path(value)==preview_path(self.requested_preview) and not self.closing:
                self.play_sample(value)
        elif kind == 'tuned_preview':
            if self.requested_preview=='tuned' and not self.closing:
                self.play_sample(value)
        elif kind == 'samples':
            self.refresh_samples()
        elif kind == 'error':
            self.error(value)
        else:
            self.refresh()

    def finished(self):
        self.refresh()
        self.append_log('Đã kết thúc lượt xử lý. Tiến trình đã được lưu.')
        if self.closing:
            self.close()
        elif self.pending_preview:
            self.preview_timer.start()

    def remove(self):
        try:
            self.store.remove(self.selected())
            self.refresh()
        except Exception as exc:
            self.error(exc)

    def remove_history(self):
        try:
            jid = self.selected(True)
            if self.store.job(jid)['status'] != 'DONE':
                raise ValueError('Chỉ xóa mục đã hoàn tất.')
            self.store.remove(jid)
            self.refresh()
        except Exception as exc:
            self.error(exc)

    def details(self):
        try:
            j = self.store.job(self.selected())
            s = json.loads(j['settings'])
            QMessageBox.information(self, j['name'], f'Giọng: {s["voice"]}\nLuồng CPU: {s["threads"]}\n'
                f'Tốc độ: {s["speed"]}× • Âm lượng: {s["volume"]}%\nOutput: {j["output"]}\n'
                f'Trạng thái: {STATUS[j["status"]]}\n{j["error"]}\n\n{j["text"][:1200]}')
        except Exception as exc:
            self.error(exc)

    def play(self):
        try:
            self.open_path(self.store.job(self.selected(True))['output'])
        except Exception as exc:
            self.error(exc)

    def open_selected_folder(self):
        try:
            self.open_path(Path(self.store.job(self.selected(True))['output']).parent)
        except Exception as exc:
            self.error(exc)

    def reexport(self):
        try:
            self.launch([self.selected(True)], self.settings())
        except Exception as exc:
            self.error(exc)

    def open_path(self, path):
        if not Path(path).exists():
            self.error('Không tìm thấy: ' + str(path))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.add_paths([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])

    def closeEvent(self, event):
        self.stop_preview()
        self.save()
        if self.busy():
            self.closing = True
            self.stop()
            self.progress_label.setText('Đang lưu đoạn hiện tại rồi đóng ứng dụng…')
            event.ignore()
            return
        self.player.stop()
        if hasattr(self,'detect_worker') and self.detect_worker.isRunning():
            self.detect_worker.wait(9000)
        self.engine.close()
        event.accept()

    def busy(self):
        return bool((self.worker and self.worker.isRunning()) or (self.aux and self.aux.isRunning()))

    def sync_cpu_preset(self,value):
        idx=self.preset.findData(value)
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(idx if idx>=0 else self.preset.count()-1)
        self.preset.blockSignals(False)

    def sync_acceleration_controls(self):
        if not hasattr(self,'preset'):
            return
        mode=self.accel_mode.currentData()
        self.preset.setEnabled(mode in ('cpu','cpu2','hybrid'))
        self.settings_widgets['threads'].setEnabled(mode in ('cpu','cpu2','hybrid'))
        self.settings_widgets['gpu_batch'].setEnabled(mode in ('gpu','hybrid'))
        if mode=='auto':
            _,batch=recommendation()
            self.settings_widgets['gpu_batch'].setValue(batch)

    def show_hardware(self,info):
        mode,batch=recommendation(info)
        gpu=f'{info["gpu"]} • {info["vram_mb"]/1024:.0f} GB VRAM' if info['gpu'] else 'Không phát hiện NVIDIA GPU'
        self.gpu_label.setText(f'{gpu}  |  Tự động: {LABELS[mode]}, nhóm {batch}')

    def refresh_samples(self):
        items,_=voices()
        selected=self.sample_table.currentRow()
        self.sample_table.setRowCount(len(items))
        for i,(voice,desc) in enumerate(items):
            path=preview_path(voice)
            for col,value in enumerate([display_name(voice),desc,'Đã lưu' if path.exists() else 'Chưa tạo']):
                item=QTableWidgetItem(value)
                if col==0: item.setData(Qt.UserRole,voice)
                item.setToolTip(str(path) if col==2 else desc)
                self.sample_table.setItem(i,col,item)
        if selected>=0:
            self.sample_table.selectRow(selected)

    def play_sample(self,path):
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.play()
        self.statusBar().showMessage('Đang nghe: '+Path(path).stem,6000)

    def sync_pause_controls(self):
        for key in ('pause_comma','pause_sentence','pause_newline'):
            self.settings_widgets[key].setEnabled(self.settings_widgets['pause_custom'].isChecked())

    def set_rhythm(self,index):
        if index==0: return
        self.settings_widgets['pause_custom'].setChecked(index!=1)
        if index in (2,3):
            values=(.18,.4,.7) if index==2 else (.08,.2,.35)
            for key,value in zip(('pause_comma','pause_sentence','pause_newline'),values):
                self.settings_widgets[key].setValue(value)

    def pick_clone(self):
        path,_=QFileDialog.getOpenFileName(self,'Chọn mẫu giọng','','Audio (*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.mp4)')
        if path: self.clone_file.setText(path)

    def insert_emotion(self,index):
        if index:
            self.editor.insertPlainText(' '+self.emotion_cue.itemText(index)+' ')
            self.emotion_cue.setCurrentIndex(0)
            self.editor.setFocus()

    def sync_clone_controls(self):
        if not hasattr(self,'clone_quality_note'): return
        clone=(self.voice.currentData() or '').startswith('clone:')
        mode=self.clone_quality.currentData()
        self.clone_quality.setEnabled(clone)
        messages={'stable':'Clone: Temperature 0.70, Top K 25, Top P 0.90, chống lặp 1.20; đoạn tối đa 180 ký tự. Ưu tiên ổn định, không bảo đảm hết lỗi phát âm.',
                  'natural':'Clone: Temperature 0.80, Top K 25, Top P 0.95, chống lặp 1.20; đoạn tối đa 220 ký tự. Ngữ điệu theo mẫu, kết quả có thể biến thiên.',
                  'manual':'Clone dùng các thông số Nâng cao bạn tự đặt.'}
        self.clone_quality_note.setText(messages[mode] if clone else 'Chế độ này áp dụng riêng cho giọng cá nhân.')
        for key in ('temperature','top_k','top_p','repetition_penalty'):
            if key in self.settings_widgets: self.settings_widgets[key].setEnabled(not clone or mode=='manual')

    def show_clone_report(self,report):
        text=f"Mẫu {report['seconds']:.1f}s • mức RMS {report['level_db']:.1f} dBFS • tín hiệu rõ khoảng {report['active_seconds']:.1f}s."
        text+='\n'+' '.join(report['warnings']) if report['warnings'] else '\nKhông thấy vấn đề mức âm rõ rệt. Vẫn cần nghe kiểm tra nhạc nền, phát âm và ngữ điệu.'
        self.clone_report.setText(text)

    def inspect_clone(self):
        if self.busy(): self.error('Chờ tác vụ hiện tại kết thúc.'); return
        if not Path(self.clone_file.text()).is_file(): self.error('Chọn file audio mẫu trước.'); return
        self.stop_preview()
        self.aux=TaskWorker('inspect_clone',request=dict(source=self.clone_file.text(),start=self.clone_start.value(),seconds=self.clone_length.value()))
        self.aux.result.connect(self.clone_inspected)
        self.aux.event.connect(self.on_event)
        self.aux.finished.connect(self.finished)
        self.aux.start(); self.refresh()

    def clone_inspected(self,result):
        self.show_clone_report(result['report'])
        if not self.closing: self.play_sample(result['path'])

    def reuse_clone(self):
        key=self.voice.currentData() or ''
        if not key.startswith('clone:'): self.error('Chọn giọng cá nhân trong danh sách Giọng trước.'); return
        try:
            path=voice_file(key); meta=json.loads(path.read_text(encoding='utf-8'))
            self.clone_file.setText(str(path.with_suffix('.wav')))
            self.clone_name.setText(meta['name']+' • bản mới')
            self.clone_start.setValue(0)
            self.clone_length.setValue(meta.get('quality',{}).get('seconds',meta['seconds']))
            self.clone_report.setText('Đã lấy mẫu gốc đã lưu. Nghe kiểm tra, chọn lọc nhiễu phù hợp rồi tạo thành giọng mới để so sánh.')
        except Exception as exc: self.error(exc)

    def start_clone(self):
        if self.busy():
            self.error('Chờ tác vụ hiện tại kết thúc rồi tạo giọng.'); return
        if not self.clone_name.text().strip() or not Path(self.clone_file.text()).is_file():
            self.error('Nhập tên giọng và chọn file audio mẫu.'); return
        self.stop_preview()
        self.aux=TaskWorker('clone',self.engine,request=dict(name=self.clone_name.text(),source=self.clone_file.text(),
            start=self.clone_start.value(),seconds=self.clone_length.value(),denoise=self.clone_denoise.isChecked(),prepare=self.clone_prepare.isChecked()))
        self.aux.event.connect(self.on_event)
        self.aux.result.connect(self.clone_ready)
        self.aux.finished.connect(self.finished)
        self.aux.start(); self.refresh()

    def clone_ready(self,key):
        self.voice.blockSignals(True)
        self.voice.clear()
        for name,desc in voices()[0]:
            self.voice.addItem(('★ ' if name in NARRATORS else 'Cá nhân • ' if name.startswith('clone:') else '')+display_name(name),name)
            self.voice.setItemData(self.voice.count()-1,desc,Qt.ToolTipRole)
        self.voice.setCurrentIndex(self.voice.findData(key))
        self.voice.blockSignals(False)
        self.refresh_samples()
        self.append_log('Đã lưu giọng cá nhân: '+display_name(key))
        report=json.loads(voice_file(key).read_text(encoding='utf-8')).get('quality')
        if report: self.show_clone_report(report)
        if not self.closing: self.voice_changed()

    def preview_tuned(self):
        if self.busy():
            self.error('Chờ lượt xử lý hiện tại kết thúc rồi tạo mẫu tùy chỉnh.'); return
        self.stop_preview()
        self.requested_preview='tuned'
        selected=self.editor.textCursor().selectedText().replace('\u2029','\n')
        if len(selected)>1000:
            self.error('Chọn tối đa 1.000 ký tự để nghe thử.'); return
        self.aux=TaskWorker('tuned',self.engine,request=self.settings() | dict(format='wav',_preview_text=selected))
        self.aux.event.connect(self.on_event)
        self.aux.finished.connect(self.finished)
        self.aux.start(); self.refresh()

    def update_profile_label(self):
        p=profile(self.voice.currentData())
        self.profile_label.setText(f'{p["description"]}\nGợi ý: {p["speed"]:.2f}× • nghỉ {p["gap"]:.2f}s • âm lượng {p["volume"]}%')

    def apply_voice_profile(self):
        for key,value in profile_settings(self.voice.currentData()).items():
            self.settings_widgets[key].setValue(value)

    def stop_preview(self):
        self.preview_timer.stop()
        self.pending_preview=None
        self.requested_preview=None
        self.player.stop()

    def voice_changed(self):
        self.stop_preview()
        self.sync_clone_controls()
        self.update_profile_label()
        if self.settings_widgets['voice_profile'].isChecked():
            self.apply_voice_profile()
        if self.settings_widgets['auto_preview'].isChecked():
            self.pending_preview=self.voice.currentData()
            self.preview_timer.start()

    def preview_pending(self):
        if self.closing or not self.pending_preview:
            return
        if self.busy():
            self.statusBar().showMessage('Mẫu giọng sẽ phát khi lượt xử lý hiện tại kết thúc.',5000)
            return
        voice=self.pending_preview
        self.pending_preview=None
        self.make_preview(voice)

    def preview_current(self):
        self.preview_timer.stop()
        self.pending_preview=None
        self.make_preview(self.voice.currentData())

    def preview_selected(self,force=False):
        row=self.sample_table.currentRow()
        if row<0:
            self.error('Hãy chọn một giọng trong bảng.')
            return
        voice=self.sample_table.item(row,0).data(Qt.UserRole)
        self.voice.setCurrentIndex(self.voice.findData(voice))
        self.preview_timer.stop()
        self.pending_preview=None
        self.make_preview(voice,force)

    def make_preview(self,voice,force=False):
        path=preview_path(voice)
        self.requested_preview=voice
        if not force and valid_preview(voice):
            self.play_sample(path)
            return
        if self.busy():
            self.error('Mẫu này chưa được tạo. Chờ lượt xử lý hiện tại kết thúc rồi nghe thử.')
            return
        self.aux=TaskWorker('preview',self.engine,[voice],force)
        self.aux.event.connect(self.on_event)
        self.aux.finished.connect(self.finished)
        self.aux.start()
        self.refresh()

    def all_previews(self):
        if self.busy():
            return
        self.aux=TaskWorker('preview',self.engine,[v[0] for v in voices()[0]])
        self.aux.event.connect(self.on_event)
        self.aux.finished.connect(self.finished)
        self.aux.start()
        self.refresh()

    def open_samples(self):
        (ROOT/'voice_samples').mkdir(parents=True,exist_ok=True)
        self.open_path(ROOT/'voice_samples')

    def run_benchmark(self):
        if self.busy():
            return
        self.tabs.setCurrentIndex(3)
        self.aux=TaskWorker('benchmark',self.engine)
        self.aux.event.connect(self.on_event)
        self.aux.result.connect(self.benchmark_done)
        self.aux.finished.connect(self.finished)
        self.aux.start()
        self.refresh()

    def benchmark_done(self,report):
        if self.aux.cancel.is_set():
            self.append_log('Đã dừng đo; giữ cấu hình tự động trước đó.')
            return
        best=report['recommended']
        self.accel_mode.setCurrentIndex(self.accel_mode.findData('auto'))
        self.settings_widgets['gpu_batch'].setValue(best['batch'])
        self.show_hardware(report['hardware'])
        self.save()
        lines=[f'{LABELS[r["mode"]]} / nhóm {r["batch"]} / CPU {r.get("threads",4)} luồng: {r["seconds"]:.1f}s ({r["realtime"]:.2f}×)' if r.get('ok') else f'{LABELS[r["mode"]]}: không chạy được' for r in report['results']]
        self.append_log('Đã chọn tự động: '+LABELS[best['mode']])
        QMessageBox.information(self,'Kết quả đo tốc độ','\n'.join(lines)+f'\n\nĐã chọn: {LABELS[best["mode"]]}, nhóm {best["batch"]}.\nThời gian đo không tính nạp model và ghép file.')
