# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603720
- **Họ và tên:** Hà Quang Vinh
- **SEED** (in ở ô đầu notebook): 6543

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Chia dữ liệu theo SEED 6543: train 3619, val 775, test 775 (106 spam), SUB 40 tin cân bằng. Hàng 4a–4c (LLM, Groq) chỉ đo trên SUB nên F1 test = F1 SUB.

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | 0.902 | 0.857 | 0.06 | 0 |
| 2. BoW + MLP | 0.935 | 0.947 | 4.30 | 0 |
| 3a. Encoder MiniLM + kNN | 0.880 | 1.000 | 17.20 | 0 |
| 3b. Encoder MiniLM + LogReg | 0.976 | 0.974 | 7.11 | 0 |
| 3c. Encoder MiniLM + MLP head | 0.981 | 0.974 | 8.33 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-oss-20b) | 0.927 | 0.927 | 2434.7 | 0* |
| 4b. LLM lớn zero-shot (gpt-oss-120b) | 0.905 | 0.905 | 2392.1 | 0* |
| 4c. LLM nhỏ few-shot | 0.923 | 0.923 | 2563.8 | 0* |

\* Groq không có trong bảng giá `PRICE_PER_1M` nên cột USD ra 0, không phải miễn phí thật.

Đường Pareto (1 − F1 theo p95): BoW + LogReg, BoW + MLP, Encoder + LogReg, Encoder + kNN. Cả 3 LLM nằm ngoài SLA 50 ms. Vì SUB chỉ 40 tin (1 tin sai ≈ 2,5 điểm %), chênh dưới ~0.05 F1 coi như hoà.

## 3. Dự đoán và thực tế

1. **`dl_relu`** — Tôi đoán bỏ ReLU thì F1 giảm vì mất phi tuyến. Thực tế F1 vẫn 0.935, bằng MLP có ReLU. Lệch vì hai lớp Linear liên tiếp tương đương một lớp Linear, nên tôi đoán đúng về mặt lý thuyết nhưng sai về hệ quả: trên BoW, phi tuyến gần như không giúp gì. Phần F1 cao hơn LogReg tự code (0.902) đến từ cách huấn luyện (Adam, mini-batch), không phải từ ReLU.
2. **`llm_models`** — Tôi đoán model lớn có F1 cao hơn model nhỏ. Thực tế model nhỏ 0.927, model lớn 0.905. Lệch vì SUB chỉ 40 tin nên chênh vài tin là nhiễu, và model lớn sai ở một số tin ham có từ khoá mạnh.
3. **`hard_eval`** — Tôi đoán accuracy BoW ~60%, Encoder ~70%, LLM ~85%, và LLM khó nhất ở prompt injection. Thực tế BoW 53–57%, Encoder 64–69%, LLM 90–93%; cả 4a và 4d đúng 100% nhóm prompt injection. LLM sai ở các tin ham chứa từ khoá spam (#2 "FREE pizza", #8 "Cash prize", #9 "free tickets", #47 kể lại tin lừa đảo). Lệch vì tôi đánh giá thấp khả năng đọc hiểu của LLM và đánh giá cao tác động của câu lệnh chèn.

## 4. Phân tích 50 case khó

Bảng gốc ghi "50 case" nhưng sau khi thêm 8 case của tôi, tổng thực tế là 58 (acc BoW + LogReg 0.534 = 31/58).

| Model | F1 test thường (SUB) | F1 case khó | Tụt |
|---|---|---|---|
| BoW + LogReg | 0.857 | 0.341 | −0.52 |
| BoW + MLP | 0.947 | 0.419 | −0.53 |
| Emb + kNN | 1.000 | 0.719 | −0.28 |
| Emb + LogReg | 0.974 | 0.667 | −0.31 |
| Emb + MLP head | 0.974 | 0.632 | −0.34 |
| LLM nhỏ zero-shot | 0.927 | 0.917 | −0.01 |
| LLM nhỏ + prompt phòng thủ | 0.927 | 0.943 | +0.02 |

**BoW + MLP tụt nhiều nhất** (−0.53), BoW + LogReg sát sau. Ở nhóm spam giọng tự nhiên BoW chỉ đúng 8–17%, thao túng/lừa đảo 11–33%, né từ khoá 14%.

- **Case #29** `C0ngr4tulat10ns! Y0u h4ve w0n...` (né từ khoá): BoW tách token bằng regex, "c0ngr4tulat10ns" không nằm trong vocab nên vector gần như toàn 0 và bị đoán ham. Encoder/LLM đọc được nghĩa nên đoán đúng.
- **Case #21** "Hi Mum, I dropped my phone..." (thao túng): không có từ khoá spam nào, chỉ có ý đồ lừa chuyển tiền. BoW chỉ đếm từ nên sai; LLM suy ra ý định nên đúng.
- Chiều ngược lại, LLM nhỏ vẫn sai tin #2 "FREE pizza in the office kitchen" (ham): bị từ khoá FREE kéo về spam.

Điểm test thường quá lạc quan vì cùng phân phối với tập train (từ khoá spam lặp lại), model chỉ cần thuộc từ khoá.

## 5. Thử thách cá nhân

**Đề #4 (Embedding):** chạy `knn_predict` với k = 1, 5, 25, báo F1 và giải thích vì sao k quá nhỏ hoặc quá lớn đều có hại.

Cách làm: dùng embedding MiniLM đã tính sẵn (`E_tr`, `E_te`), gọi `knn_predict` với từng k và tự tính F1 từ TP/FP/FN.

| k | F1 | TP | FP | FN |
|---|---|---|---|---|
| 1 | 0.899 | 98 | 14 | 8 |
| 5 | 0.880 | 99 | 20 | 7 |
| 25 | 0.891 | 102 | 21 | 4 |

Giải thích:
- F1 gần như phẳng theo k (chênh tối đa 0.019, vài tin trên 775). k tăng thì FN giảm (8 → 4) nhưng FP tăng (14 → 21): k là núm đánh đổi precision/recall hơn là "tốt hay xấu".
- k quá nhỏ chỉ nghe 1 láng giềng nên một tin train sai nhãn hoặc lạc chỗ kéo kết quả sai theo (phương sai cao, overfit).
- k quá lớn làm ranh giới quá mượt, bỏ qua cấu trúc cục bộ, và spam chỉ chiếm ~14% nên có nguy cơ bị phe ham lấn phiếu.
- Điều bất ngờ: tôi đoán k lớn làm FN tăng, thực tế FN giảm. Có thể do embedding gom spam thành cụm chặt nên k = 25 vẫn đủ láng giềng spam, còn tin ham sát biên bị kéo nhầm thành spam.
- Chọn k nên dùng tập val rồi báo test một lần; chọn trực tiếp trên test như ở đây là "nhìn trộm" đáp án.

## 6. Phương án cho sản phẩm thật

Yêu cầu: p95 < 50 ms/tin, 10 triệu SMS/ngày (~116 tin/giây).

- **Đạt độ trễ:** cả 5 cách không phải LLM (p95 từ 0.06 đến 17 ms). Cả 3 LLM trượt, p95 ~2,4 giây.
- **Chọn: Encoder MiniLM + LogReg (3b)** — F1 test 0.976, p95 7 ms, gần như không tốn tiền suy luận. 3c hơn 0.005 F1 là trong vùng nhiễu nên không đáng đổi. kNN loại vì F1 test chỉ 0.880 và chỉ ~99 tin/s.
- **Chi phí nếu dùng LLM** (100 token vào, 1 token ra, 10 triệu tin/ngày): gpt-6-luna ≈ $105/ngày ≈ $3.150/tháng; gpt-6.1-terra ≈ $31.500/tháng.
- **Thiết kế lai (router):** Encoder + LogReg chạy trước; p < 0,3 cho qua, p > 0,7 chặn, p ∈ [0,3; 0,7] đẩy sang LLM nhỏ với prompt phòng thủ. Phần lớn tin xử lý ở ~7 ms, không tốn tiền; LLM chỉ lo nhóm lưỡng lự và mạnh đúng chỗ Encoder yếu (spam tự nhiên, thao túng, né từ khoá đều đạt 100% ở 4d). Nếu nhóm lưỡng lự khoảng 10% thì chi phí LLM còn ≈ $315/tháng (con số ước tính, chưa đo).
- **Đánh đổi chấp nhận:** tin mà Encoder tự tin sai vẫn lọt; ngưỡng cần chỉnh dựa trên p của các case sai trong bộ case khó; cần thêm dữ liệu spam thật được gán nhãn và cập nhật theo thời gian. Thêm lớp luật phát hiện prompt injection ngoài LLM.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời gì | Tôi sửa / kiểm chứng gì |
|---|---|---|---|
| Data leakage | Data Leakage là gì, ảnh hưởng đánh giá thế nào? | Giải thích rò rỉ dữ liệu và kết quả đánh giá ảo | Đối chiếu cách chia train/test trong notebook |
| Huấn luyện MLP | Giải thích `zero_grad`, `forward`, `backward`, `step` | Quy trình tính loss, lan truyền ngược, cập nhật trọng số | Kiểm tra từng bước trong code |
| Embedding search | Cách tìm tin tương tự bằng embedding và kNN | Mã hoá văn bản, cosine, Top-K | Kiểm tra các tin truy xuất được |
| Đánh giá mô hình | Precision, Recall, F1 và cách đánh giá | So sánh ưu nhược BoW, Embedding, LLM | Đối chiếu kết quả trên case khó |
| Phân tích kết quả | Model nào xử lý tốt từng nhóm case? | Phân tích tỷ lệ đúng theo nhóm | Kiểm tra tỷ lệ và nhóm còn yếu |
| Prompt injection | Là gì, hạn chế thế nào? | Prompt phòng thủ, tách dữ liệu khỏi chỉ dẫn | So sánh 4a với 4d |
| Phân tích lỗi | Model nào sai ở case tự thêm? | Code lọc dự đoán sai theo model | Đối chiếu nhãn thật, nhãn đoán, nhóm |
| So sánh LLM / API | Gọi API hai model, chi phí có mất phí không? | Cách gọi API, tính phí theo token, giới hạn ngân sách | Kiểm tra API key, model ID, usage |
| Tham số phân loại | Vì sao `temperature=0`, `max_tokens` nhỏ? | Tính ổn định, giới hạn đầu ra, token reasoning | Đối chiếu `PROVIDERS` và kết quả chạy |
| Zero-shot | Vì sao LLM không cần dữ liệu có nhãn? | Lợi ích với bài toán mới, thiếu nhãn | Liên hệ bài toán Spam/Ham |

Phần 1–3 (vùng tự code) tôi chỉ hỏi AI để hiểu khái niệm, không nhờ viết code.

**Điều hiểu ra hôm nay:**
- Điểm cao trên test cùng phân phối không chứng minh model "hiểu" spam; bộ case khó làm F1 của BoW tụt từ ~0.9 xuống ~0.35–0.42.
- Phi tuyến (ReLU) không phải lý do MLP hơn LogReg trên BoW; đầu vào BoW mới là giới hạn. Đổi đầu vào sang embedding mới cải thiện mạnh (0.902 → 0.976).
- LLM mạnh ở ngữ cảnh và ý định nhưng chậm (~2,4 giây) và có thể bị từ khoá kéo đi; không có lựa chọn nào tốt nhất tuyệt đối, phải cân F1, độ trễ và chi phí, và nên dùng thiết kế lai.
- Số đo trên 40 tin rất nhiễu; phải đọc chênh lệch theo số tin sai chứ không theo số lẻ
