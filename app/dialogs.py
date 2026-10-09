"""Secondary windows: reading preview, pronunciation dictionary, chunk review."""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QDialogButtonBox, QMessageBox)

DICTIONARY_HELP = (
    '# Từ điển phát âm: mỗi dòng “chữ viết = cách đọc”. Không phân biệt hoa thường, chỉ thay trọn từ.\n'
    '# Áp dụng khi THÊM tác vụ mới (tác vụ cũ giữ bản đã chốt). Dòng bắt đầu bằng # là ghi chú.\n'
    '# Ví dụ:\n'
    '# TP.HCM = thành phố Hồ Chí Minh\n'
    '# Kaito = Cai tô\n'
    '# AI = ây ai\n'
)


def _table(headers):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().hide()
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setAlternatingRowColors(True)
    table.setWordWrap(True)
    return table


class ReadingPreviewDialog(QDialog):
    """Text after chunking, dictionary and the SDK's number/symbol normalisation."""
    def __init__(self, parent, rows, total):
        super().__init__(parent)
        self.setWindowTitle('Văn bản sẽ được đọc')
        self.resize(960, 640)
        layout = QVBoxLayout(self)
        shown = f'{len(rows)}/{total}' if len(rows) < total else str(total)
        note = QLabel(f'{shown} đoạn. Cột phải là chữ sau khi chuẩn hóa số, ký hiệu, viết tắt và từ điển phát âm '
                      '— đúng những gì model nhận. Dấu “|” là chỗ SDK tự chia nhỏ thêm. Sửa văn bản gốc hoặc thêm '
                      'vào Từ điển phát âm nếu thấy đọc sai.')
        note.setWordWrap(True)
        layout.addWidget(note)
        boundary = {'chunk': 'đoạn', 'comma': 'phẩy', 'sentence': 'câu', 'newline': 'xuống dòng'}
        table = _table(['#', 'Văn bản gốc', 'Model sẽ đọc', 'Nghỉ sau'])
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [str(row['index']), row['text'], row['normalized'], boundary.get(row['boundary'], row['boundary'])]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if col == 2 and value.replace(' | ', ' ').strip(' ,.') != row['text'].strip(' ,.'):
                    item.setForeground(Qt.darkYellow)
                table.setItem(i, col, item)
        header = table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        for col in (0, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        table.resizeRowsToContents()
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class DictionaryDialog(QDialog):
    def __init__(self, parent, path):
        super().__init__(parent)
        self.path = Path(path)
        self.setWindowTitle('Từ điển phát âm')
        self.resize(640, 520)
        layout = QVBoxLayout(self)
        self.editor = QPlainTextEdit()
        try:
            text = self.path.read_text(encoding='utf-8-sig')
        except OSError:
            text = DICTIONARY_HELP
        self.editor.setPlainText(text)
        layout.addWidget(self.editor)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(self.editor.toPlainText(), encoding='utf-8')
        tmp.replace(self.path)
        self.accept()


class ChunkDialog(QDialog):
    """Listen to single chunks and queue chosen ones for regeneration."""
    def __init__(self, parent, store, jid, play):
        super().__init__(parent)
        self.store, self.jid, self.play = store, jid, play
        self.reset = False
        job = store.job(jid)
        self.setWindowTitle('Các đoạn • ' + job['name'])
        self.resize(900, 600)
        layout = QVBoxLayout(self)
        note = QLabel('Chọn một hoặc nhiều đoạn (giữ Ctrl) để nghe hoặc tạo lại. Đoạn tạo lại được sinh mới; '
                      'tác vụ quay về Hàng đợi, nhấn Tiếp tục / Thử lại để tạo rồi xuất file mới.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = _table(['#', 'Văn bản', 'Thời lượng', 'Tốc độ đọc', 'Trạng thái'])
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.doubleClicked.connect(lambda _: self.listen())
        layout.addWidget(self.table)
        row = QHBoxLayout()
        for text, callback in (('▶ Nghe đoạn', self.listen), ('Tạo lại đoạn đã chọn', self.regenerate)):
            b = QPushButton(text)
            b.clicked.connect(callback)
            row.addWidget(b)
        row.addStretch()
        close = QPushButton('Đóng')
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)
        self.fill()

    def fill(self):
        from tools.clone_quality import syllables
        chunks = self.store.chunks(self.jid)
        rates = [syllables(c['text']) / c['duration'] for c in chunks if c['duration'] and syllables(c['text']) >= 5]
        median = sorted(rates)[len(rates) // 2] if rates else None
        self.table.setRowCount(len(chunks))
        for i, c in enumerate(chunks):
            syl = syllables(c['text'])
            rate = syl / c['duration'] if c['duration'] else 0
            odd = median and syl >= 5 and not .6 < rate / median < 1.6
            values = [str(c['idx'] + 1), c['text'], f'{c["duration"]:.1f}s' if c['duration'] else '—',
                      (f'{rate:.1f} tiếng/s' + (' ⚠' if odd else '')) if rate else '—', c['status']]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, c['idx'])
                if odd:
                    item.setToolTip('Tốc độ đọc khác thường so với các đoạn khác: có thể bị đọc sót hoặc lặp. Nên nghe lại.')
                self.table.setItem(i, col, item)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (0, 2, 3, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

    def chosen(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(Qt.UserRole) for r in rows]

    def listen(self):
        chunks = {c['idx']: c for c in self.store.chunks(self.jid)}
        picked = self.chosen()
        if not picked:
            return
        path = chunks[picked[0]]['path']
        if path and Path(path).exists():
            self.play(path)
        else:
            QMessageBox.information(self, 'Nghe đoạn', 'Đoạn này chưa có audio (đang chờ tạo hoặc cache đã dọn).')

    def regenerate(self):
        picked = self.chosen()
        if not picked:
            return
        try:
            self.store.reset_chunks(self.jid, picked)
            self.reset = True
            self.fill()
        except Exception as exc:
            QMessageBox.warning(self, 'Tạo lại đoạn', str(exc))
