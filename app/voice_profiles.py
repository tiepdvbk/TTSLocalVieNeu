"""Editable starting points for narration, not claims of acoustic optimality."""
STORY_TEXT = (
    'Chiều ấy, An trở về căn nhà nhỏ bên sông. Trên bàn, một lá thư vẫn đợi cậu, '
    'nằm dưới chiếc đèn đã tắt từ lâu.\n'
    'Cậu mở thư. Chỉ có một dòng chữ: Đừng quên lời hẹn dưới gốc bàng. '
    'Ngoài hiên, tiếng bước chân bỗng dừng lại.'
)
# Gain measured on the shared story sample toward -20 LUFS; see voice-calibration.json.
# name: speed, inter-chunk silence (before tempo adjustment), volume %, description
PROFILES = {
    'Thiện Minh': (.97, .30, 97, 'Kể chuyện Bắc • nhịp vừa'),
    'Thiền Tâm Đức': (.95, .40, 90, 'Kể chậm • suy ngẫm'),
    'Thái Sơn': (.97, .30, 103, 'Kể chuyện Nam • nhịp vừa'),
    'Thanh Bình': (.96, .35, 102, 'Truyện dài • thong thả'),
    'Ngọc Linh': (.97, .30, 96, 'Truyện dài • giọng nữ Bắc'),
    'Thục Đoan': (.97, .30, 105, 'Truyện dài • giọng nữ Nam'),
    'Mỹ Duyên': (.98, .30, 90, 'Đọc truyện • nữ Nam'),
    'Quỳnh Anh': (.98, .30, 99, 'Đọc truyện • nữ Bắc'),
    'Đức Trí': (.97, .30, 95, 'Đọc truyện • nam Nam'),
    'Kim Thanh': (.98, .30, 103, 'Đọc truyện • nữ Nam'),
    'Trúc Ly': (.97, .30, 92, 'Tự nhiên • truyện đời thường'),
    'Hải Đăng': (.97, .30, 96, 'Tự nhiên • truyện đời thường'),
    'Ngọc Huyền': (.98, .30, 94, 'Tự nhiên • nữ Bắc'),
    'Quang Sơn': (.98, .30, 100, 'Tự nhiên • nam Trung'),
    'Ngọc Trân': (.98, .30, 98, 'Tự nhiên • nữ Trung'),
    'Adam bựa': (1.0, .25, 93, 'Tự nhiên • đoạn hội thoại'),
    'Adam': (1.0, .25, 100, 'Tự nhiên • nam Nam'),
    'Quốc Tuấn': (.98, .30, 105, 'Tự nhiên • nam Bắc'),
    'Phạm Tuyên': (.98, .30, 99, 'Tự nhiên • nam Bắc'),
    'Xuân Vĩnh': (.98, .30, 98, 'Tự nhiên • nam Bắc'),
    'Đoan Trang': (.98, .30, 103, 'Tự nhiên • nữ Bắc'),
    'Mai Anh': (.95, .35, 104, 'Tin tức • hạ nhịp cho truyện'),
    'Thùy Dung': (.95, .35, 98, 'Tin tức • hạ nhịp cho truyện'),
    'Minh Đức': (.95, .35, 108, 'Tin tức • hạ nhịp cho truyện'),
    'Minh Triết': (.95, .35, 99, 'Tin tức • hạ nhịp cho truyện'),
}
ALIASES = {'Minh Quân': 'Hải Đăng', 'Minh Quân Pro': 'Hải Đăng',
           'Anh Khôi': 'Thiện Minh', 'Mạnh Dũng': 'Quốc Tuấn'}
NARRATORS = tuple(PROFILES)[:10]


def canonical_voice(name):
    return ALIASES.get(name, name)


def profile(name):
    speed, gap, volume, description = PROFILES.get(canonical_voice(name), (1., .30, 100, 'Kể chuyện'))
    return dict(speed=speed, gap=gap, volume=volume, description=description)


def profile_settings(name):
    return {k: v for k, v in profile(name).items() if k != 'description'}
