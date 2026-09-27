# Nâng cấp ngày 27/09/2026

- VieNeu SDK 3.8.3, revision `c1390abbdb2eedcdf58eafb546966c06ce27af71`.
- Model Turbo mới nhất đã kiểm tra: `61b85e3d937fbbacb387714180e8182823512523` (23/09/2026). Chi tiết codec trong `model-revisions.json`.
- Giữ PyTorch 2.8.0 + CUDA 12.6 và FP32 cho GTX 1050 Ti. Không tự cập nhật PyTorch/CUDA trong lúc build.
- Dùng vòng sinh CUDA Graph và lịch sử chống lặp trên GPU của VieNeu mới. Giữ hệ số chống lặp và giọng trong tác vụ.
- Gom đoạn có độ dài gần nhau trong cửa sổ tối đa 512 đoạn; xuất theo chỉ số gốc. Vì vậy số thứ tự tạo đoạn có thể khác thứ tự đọc.
- Mỗi nhóm lưu tiến trình bằng một giao dịch an toàn; nhóm GPU tiếp theo có thể chạy trong lúc lưu tiến trình. Khi tạm dừng, tool hoàn tất và lưu cả nhóm đã gửi trước.
- Khi thiếu VRAM, giảm batch và ghi nhớ mức giảm trong worker. Card <= 6 GB chỉ giữ một dạng CUDA Graph để tránh tích lũy bộ nhớ.
- Ô Luồng CPU nhận số nguyên dương tự nhập đến 2.147.483.647, không giới hạn theo số nhân. Đây là giới hạn nhập liệu, không phải mức máy có thể chạy thực tế. Thông số áp dụng cho từng worker CPU; GPU thuần vẫn dùng một worker GPU.
- Kết quả benchmark bản cũ không được áp dụng tự động cho engine mới. Chưa có benchmark mới thì chế độ tự động dùng GPU nhóm 8 nếu có NVIDIA/runtime, hoặc CPU phù hợp.

## Build và mẫu giọng (bổ sung)

`BUILD.bat` hiện tự chuẩn bị PowerShell 7, Python, hai môi trường CPU / GPU, FFmpeg, SDK ghim commit và model ghim revision. Không cần môi trường cài trước. Xem README cho lựa chọn build, tạo 25 mẫu truyện và portable.

Thêm bộ thiết lập gợi ý từng giọng, nghe mẫu khi đổi giọng và bảo vệ khỏi phát nhầm mẫu khi đổi liên tục. Những số đo dưới đây là phép đo hiệu năng engine trước khi thêm chức năng mẫu truyện; chức năng này không thay đổi thuật toán GPU.

## Đo lại tốc độ

Đã đo trên i7-8750H / GTX 1050 Ti 4 GB, giọng Mỹ Duyên, batch 32, FP32, chống lặp 1,2:

| Bài đo | Thời gian xử lý | Audio gốc sinh ra |
|---|---:|---:|
| Bản 3.6.0, 64 đoạn rải đều trong truyện | 120,87 giây | 415,56 giây |
| Bản cuối 3.8.3, cùng 64 đoạn, ghép theo độ dài | 30,81 giây | 440,64 giây |
| Bản cuối 3.8.3, 256 đoạn rải đều | 101,16 giây | 1.708,64 giây |

Trên mẫu 64 đoạn, thời gian giảm 74,5% (nhanh hơn 3,92 lần). Các số trên không tính nạp model và lượt làm nóng, có tính ghi WAV an toàn; không tính bước xuất file ghép cuối. Cùng văn bản nhưng audio khác thời lượng do phiên bản model/SDK và lấy mẫu theo batch thay đổi. Không suy diễn mức tăng tốc này thành cam kết cho file 4 giờ 32 phút.

Lượt 256 đoạn chạy 8 nhóm GPU, không OOM/CPU fallback; bộ nhớ tensor đỉnh 2.243 MiB, bộ nhớ dự trữ được báo cuối các nhóm cao nhất 2.600 MiB. Tất cả 256 WAV qua kiểm tra định dạng 48 kHz, dữ liệu hữu hạn, không rỗng/không im lặng. Đây là kiểm tra kỹ thuật, không thay thế nghe đánh giá phát âm và ngữ điệu.

Chi tiết máy, chỉ số đoạn nguồn, thời lượng từng đoạn và thời gian từng công đoạn được giữ tại thư mục dự án gốc trong `logs/performance-*/result.json` và `logs/performance-summary.json`.

`tools/measure_performance.py --label ten-lan-do --count 64 --order length` lấy 64 đoạn rải đều từ tác vụ dài nhất, chạy GPU và lưu WAV cùng số đo trong `logs/performance-ten-lan-do`. Chạy bằng `.venv/Scripts/python.exe`. Không sửa tác vụ hoặc cấu hình người dùng.

Số đo tách nạp model, sinh mã âm thanh, giải mã waveform và ghi WAV; độ nhanh của mẫu không phải cam kết cho cả truyện. Thay engine hoặc thứ tự batch có thể thay đổi kết quả lấy mẫu ngẫu nhiên, nên audio không đồng nhất từng byte với bản cũ.
