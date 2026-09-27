# TTSLocalVieNeu — TTS Studio

Ứng dụng Windows đọc truyện tiếng Việt offline, dùng **VieNeu 3.8.3 / v3 Turbo**, 25 giọng, CPU hoặc NVIDIA GPU. Hỗ trợ TXT, hàng đợi, tạm dừng / tiếp tục, WAV / MP3 / FLAC / M4A.

## Tải source và chạy

1. Tải ZIP từ **Code → Download ZIP**, giải nén vào thư mục ngắn có quyền ghi, ví dụ `E:\TTSLocalVieNeu`. Không chạy BAT trực tiếp bên trong ZIP.
2. Nhấp đôi **BUILD.bat**. Lần đầu cần Internet, Windows 10/11 x64 và khoảng 20 GB trống; bản portable cần thêm dung lượng.
3. BAT tự tải công cụ cục bộ, Python 3.12.10, thư viện CPU / CUDA 12.6, SDK và model đúng phiên bản; tạo 25 mẫu truyện, chạy kiểm thử rồi build.
4. Khi báo thành công, mở **VieNeuTTSStudio.exe** ngay trong thư mục. Những lần sử dụng bình thường chạy offline.

Không cần cài sẵn Python, Git, PowerShell 7 hay CUDA Toolkit. NVIDIA GPU cần driver tương thích; máy không có NVIDIA vẫn dùng CPU. Tải model có tiến trình theo file. Mất mạng thì chạy lại BAT; các file model đã tải xong được dùng lại. Lần đầu có thể lâu vì tải vài GB dữ liệu và tạo mẫu cho cả 25 giọng.

Giữ nguyên `.tools`, `.venv`, `.gpu`, `_internal`, `models`, `tools` cạnh EXE ở bản build từ source. **Không chỉ chuyển một file EXE** sang máy khác. Không đổi vị trí thư mục môi trường phát triển đã build; để chuyển máy, dùng bản portable hoặc tải source rồi build lại.

Các tùy chọn trong cửa sổ lệnh:

```bat
BUILD.bat -Portable
BUILD.bat -SetupOnly
BUILD.bat -SkipSamples
BUILD.bat -NoPause
```

`-Portable` tạo thêm `portable\VieNeuTTSStudio`, có hai runtime độc lập và model; chép nguyên thư mục này để dùng trên máy khác. Thư mục portable là sản phẩm build và sẽ bị thay thế khi build lại, nên lưu bài đọc ở ngoài thư mục đó. `-SetupOnly` chỉ chuẩn bị dữ liệu; `-SkipSamples` hoãn tạo mẫu đến lúc nghe trong app. `-NoPause` dùng cho tự động hóa. Sau khi môi trường đã sẵn sàng, `.tools\powershell\pwsh.exe -File build.ps1` chỉ kiểm thử và build, không tải lại model.

## Giọng kể truyện và nghe mẫu

- 25 giọng có sẵn. 10 giọng mang nhãn kể / đọc truyện của upstream được xếp đầu và đánh dấu ★, gồm Thiện Minh, Thiền Tâm Đức, Thái Sơn, Thanh Bình, Ngọc Linh, Thục Đoan, Mỹ Duyên, Quỳnh Anh, Đức Trí, Kim Thanh.
- So với bộ 23 giọng: thêm **Adam bựa, Thiền Tâm Đức**; **Trúc Ly** được thay mẫu gốc. **Minh Quân / Minh Quân Pro → Hải Đăng**, **Anh Khôi → Thiện Minh**, **Mạnh Dũng → Quốc Tuấn** là đổi tên. Thiết lập giọng cũ được chuyển tên khi mở app.
- Đổi giọng tự áp dụng tốc độ, khoảng nghỉ và âm lượng gợi ý; tự phát một đoạn truyện ngắn bằng giọng đó. Có thể tắt riêng hai lựa chọn này.
- Thiết lập gợi ý dựa trên nhãn phong cách của upstream, không phải khẳng định “tốt nhất” cho mọi truyện. Âm lượng từng giọng được gợi ý từ phép đo mẫu truyện hướng tới −20 LUFS (90–108% ở bộ mẫu này); bộ giới hạn đỉnh giúp tránh vỡ tiếng. Đây là mức cố định theo mẫu, không phải chuẩn hóa độ lớn toàn bộ mỗi chương. Chi tiết đo nằm ở `voice-calibration.json`. Bạn có thể chỉnh cả ba thông số và lưu lại. Mở app giữ thiết lập đã lưu; đổi giọng khi bật tự áp dụng sẽ đặt lại gợi ý.
- Mẫu cố định dùng thiết lập gợi ý, được ghi rõ trong giao diện; chỉnh tay không làm thay đổi mẫu. Mẫu được tạo bằng cùng cách chia đoạn, chèn nghỉ và xử lý tốc độ / âm lượng như xuất truyện. Cache tự đổi khi nội dung, cấu hình giọng hoặc SDK thay đổi.
- Trong tab **Nghe thử giọng**, có thể nghe, dừng, tạo lại hoặc bổ sung tất cả mẫu. Nội dung mẫu là một đoạn truyện tự viết, nằm ở `app/voice_profiles.py`.
- Đổi thiết lập chỉ áp dụng khi thêm tác vụ mới. Tác vụ cũ giữ giọng và thiết lập đã chốt; có thể dùng **Xuất lại tốc độ / âm lượng** từ cache.

## Hiệu năng và an toàn tiến trình

GPU có batch 1–32, ghép các đoạn gần độ dài trong cửa sổ hữu hạn, CUDA Graph và tự giảm batch khi thiếu VRAM. GTX 1050 Ti dùng PyTorch 2.8.0 + CUDA 12.6, FP32; cấu hình mặc định GPU nhóm 8 nếu chưa đo. Nút đo tốc độ chọn cấu hình theo máy.

Ô **Luồng CPU** cho nhập số nguyên dương, không khóa ở số nhân máy. CPU kép có hai worker, mỗi worker dùng số luồng đã nhập; GPU thuần không dùng thông số này. Thêm luồng không đảm bảo nhanh hơn.

SQLite lưu cả nhóm hoàn tất trong một giao dịch, kiểm tra hash khi tiếp tục và ghép theo thứ tự văn bản gốc. Tạm dừng đợi phần đã gửi cho worker hoàn tất. Xem [UPGRADE_NOTES.md](UPGRADE_NOTES.md) để biết kết quả đo và giới hạn phép đo.

## Mã nguồn, model và giấy phép

Source không chứa model, môi trường Python, EXE, audio mẫu, log, cấu hình cá nhân, dữ liệu tác vụ hay nội dung truyện. BAT tải / tạo chúng trên máy người dùng. SDK ghim ở `vendor-revision.txt`, model ghim ở `model-revisions.json`, dependency ghim trong hai file `requirements-*-lock.txt` / `requirements-lock.txt`. Nguồn SDK và công cụ tải về được kiểm tra SHA-256. SDK tối giản được cài `--no-deps`; không cài thêm giao diện web Gradio / Librosa của upstream.

Nguồn nền: [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS), [model v3 Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo), [MOSS codec](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano). v3 Turbo là bản mã nguồn mở hiện hành; v4 chỉ có trên dịch vụ của nhà phát triển. Chi tiết bên thứ ba: [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

## Kiểm thử

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe main.py --self-test
.\VieNeuTTSStudio.exe --self-test
.\VieNeuTTSStudio.exe --ui-e2e
.\VieNeuTTSStudio.exe --preview-test
```

Kiểm thử GUI có dữ liệu riêng trong logs, không sửa hàng đợi người dùng. `--self-test` chặn socket để kiểm tra offline; worker GPU bật chế độ offline của Hugging Face. Kết quả kiểm thử ghi trong `logs`. Các bài test dùng FFmpeg do BAT chuẩn bị.
