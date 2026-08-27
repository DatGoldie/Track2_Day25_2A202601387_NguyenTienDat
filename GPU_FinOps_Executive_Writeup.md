# Báo Cáo Kỹ Thuật & Chiến Lược FinOps Tối Ưu Hóa Chi Phí GPU (NimbusAI)
## GPU FinOps Technical Write-up & Optimization Strategy

> **Tác giả:** FinOps Lead Engineer  
> **Dự án:** NimbusAI Cloud & GPU Infrastructure Cost Optimization  
> **Thời gian:** Tháng 6/2026 (Snapshot Dữ liệu Thực tế)  
> **Đầu ra chính:** Báo cáo chi phí GPU (Baseline vs. Optimized) — Đầu vào Milestone 2  

---

## 1. Tóm tắt Điều hành (Executive Summary)

NimbusAI là một startup công nghệ LLM đang tăng trưởng nóng, dẫn đến hóa đơn GPU gia tăng nhanh chóng. Bằng việc chuyển dịch phương pháp quản lý chi phí từ chỉ số truyền thống `$/GPU-giờ` sang chỉ số hiệu quả kinh tế **`$/1M-token`**, kết hợp áp dụng 4 đòn bẩy FinOps cốt lõi và 5 phần mở rộng chuyên sâu:

- **Chi phí vận hành GPU hàng tháng (Monthly Spend):**
  - **Baseline (Ban đầu):** **$27,133 / tháng**
  - **Optimized (Sau tối ưu):** **$14,626 / tháng**
  - **Tổng chi phí cắt giảm:** **$12,507 / tháng (Tiết kiệm 46.1%)**
- **Đơn giá phục vụ Inference (`$/1M-token`):**
  - **Baseline:** **$6.488 / 1M-token**
  - **Optimized:** **$1.126 / 1M-token**
  - **Tỷ lệ cắt giảm chi phí token:** **82.6%**
- **Dấu chân Carbon (Carbon Footprint):** Cắt giảm **92.1% lượng phát thải CO2e** đối với các workload huấn luyện và batch có thể gián đoạn thông qua *Carbon-Aware Scheduling*.

---

## 2. Bảng Phân Tích Chi Tiết Từng Đòn Bẩy FinOps (Savings by Lever)

| Đòn bẩy (Lever) | Tiết kiệm (USD/tháng) | Tỷ trọng (% tổng tiết kiệm) | Cơ chế kỹ thuật & Tác động kinh tế |
|---|---|---|---|
| **Purchasing Strategy (Spot & Reserved)** | **$10,040** | **80.3%** | Chuyển đổi 5 batch/training job sang Spot Instances với Checkpointing; cam kết 3-Year Reserved cho 3 cụm Inference 24/7. |
| **Inference Levers (Cascade, Cache, Batch)** | **$1,212** | **9.7%** | Tích hợp Prompt Caching (chiết khấu 90% input prefix), Semantic Cascade Routing sang Small Model (rẻ hơn 15x), Batch API (-50%). Giảm đơn giá $/1M-token đi 82.6%. |
| **Right-size Util-Lies (MBU Right-sizing)** | **$655** | **5.2%** | Hạ cấp các GPU bị nghẽn băng thông HBM nhưng dư thừa FLOPs (H100 -> A100/A10G) dựa trên đo lường MBU thực tế. |
| **Kill Idle GPUs (Zombie Instance Reaper)** | **$600** | **4.8%** | Tự động quét và tắt các instance GPU không có workload (GPU-Util < 10%) sau 2 giờ không hoạt động ($20.0/ngày). |
| **TỔNG CỘNG** | **$12,507** | **100.0%** | **Cắt giảm 46.1% tổng hóa đơn GPU hàng tháng.** |

### Biểu đồ Thác nước Tiết kiệm (Savings Waterfall)
File biểu đồ `outputs/savings.png` trực quan hóa phân bổ giá trị tài chính đóng góp từ 4 đòn bẩy trên.

---

## 3. Phân Tích Chuyên Sâu: Hiện Tượng "GPU-Util Lie"

### Bản chất của "GPU-Util Lie"
Trong quản trị hạ tầng AI, lệnh `nvidia-smi` trả về chỉ số `GPU-Util %`. Tuy nhiên, đây là một **chỉ số đo thời gian xung nhịp hoạt động (time-active clock)**, hoàn toàn **không đo lường hiệu suất tính toán thực tế (Compute Throughput)**:
- **Trường hợp điển hình trong đo kiểm:** `gpu-h100-4` ghi nhận **GPU-Util 98.0%**, nhưng chỉ số **MFU (Model FLOPs Utilization) chỉ đạt 20.2%** (và MBU đạt 45.0%). Tương tự, `gpu-a10g-1` ghi nhận **GPU-Util 93.1%** nhưng MFU chỉ đạt **28.5%**.
- **Hệ quả tài chính:** Doanh nghiệp chi trả trọn vẹn $2.50/giờ thuê GPU H100 nhưng chỉ thu về ~1/5 năng lực tính toán thực tế của Tensor Cores, gây thất thoát **$1,080/tháng trên mỗi GPU H100 bị lãng phí**.

### Nguyên nhân gốc rễ (Root Causes) theo Mô hình Roofline:
1. **Memory-Bound Bottleneck (Nghẽn băng thông bộ nhớ):**
   - Trong quá trình Autoregressive Generation (Decode phase của LLM), mỗi token sinh ra cần đọc toàn bộ trọng số mô hình từ HBM vào SRAM với cường độ tính toán cực thấp (**Arithmetic Intensity ~ 1–2 FLOP/byte**).
   - Trong khi đó, điểm uốn (Ridge Point) của GPU H100 là **~295 FLOP/byte (BF16)**. Do đó, GPU hoàn toàn rơi vào vùng *memory-bound*. Các Tensor Cores bị "bỏ đói" (memory stall), ngồi chờ dữ liệu load từ HBM, dù clock của GPU vẫn ghi nhận 98% busy.
2. **Kernel Launch Overhead & Small Batch:**
   - Việc gửi các request đơn lẻ (Batch Size = 1) khiến chi phí overhead gọi CUDA kernel từ CPU lấn át thời gian tính toán của Tensor Cores.
3. **I/O & Dataloader Stalls:**
   - CPU hoặc đĩa NVMe không cấp dữ liệu kịp thời trong các vòng lặp huấn luyện, khiến GPU giữ context mà không thực thi phép nhân ma trận.

---

## 4. Báo Cáo Kết Quả 5 Phần Mở Rộng "Your Turn"

### D.1 — Cải Thiện Chính Sách Mua Sắm (`recommend_tier`)
- **Điểm hòa vốn cam kết (Break-even Utilization):**
  $$\text{Break-even} = 1 - \text{Discount} = 1 - 0.45 = 55\% \quad (\approx 13.2\text{ giờ/ngày})$$
- **Cải tiến logic:** Bổ sung ma trận đánh giá rủi ro gián đoạn (`interruption_rate`) theo chủng loại phần cứng (commodity GPU như A10G/L4 có tỷ lệ thu hồi ~8%, trong khi H100 cluster ổn định hơn ~4%).
- **Kết quả:** Workload có duty cycle $\ge 55\%$ và chạy liên tục được đề xuất 3-Year Reserved; các batch job được phân bổ sang Spot kết hợp Checkpointing.

### D.2 — Right-Sizing Dựa Trên MBU & `$/GB-VRAM`
- Phân tích chi phí bộ nhớ trên danh mục:
  - H100: **$0.0313 / GB-VRAM / giờ** ($2.50 / 80GB) | Băng thông: 3.35 TB/s ($0.75 / TB/s)
  - A100: **$0.0224 / GB-VRAM / giờ** ($1.79 / 80GB) | Băng thông: 2.00 TB/s ($0.90 / TB/s)
  - A10G: **$0.0417 / GB-VRAM / giờ** ($1.00 / 24GB) | Băng thông: 0.60 TB/s ($1.67 / TB/s)
- **Đề xuất Right-sizing:** Với `gpu-h100-4` (MBU 45%, VRAM sử dụng < 40GB), chuyển đổi sang **A100** giúp tiết kiệm **$0.71/giờ = $511.20/tháng (giảm 28.4%)** mà không ảnh hưởng tới latency.

### D.3 — Kinh Tế Học Của Prompt Caching (`cache_is_worth_it`)
- **Công thức điểm hòa vốn số lần đọc (Break-even Read Count):**
  $$\text{Break-even Reads} = \frac{\text{Chi phí ghi Cache (Write Cost)}}{(1 - \text{Tỷ lệ chiết khấu Đọc}) \times \text{Chi phí Đọc chưa cache}} = \frac{0.20}{(1 - 0.10) \times 0.20} \approx 1.11 \text{ lần}$$
- **Kết luận:** Chỉ cần prompt prefix được tái sử dụng từ **2 lần trở lên**, việc lưu cache đã bắt đầu sinh lời. Trong dataset của NimbusAI, tỷ lệ phủ cache đạt **29.7%** với số lượt đọc trung bình > 3.5 lần, chứng minh Prompt Caching mang lại ROI dương ngay từ request thứ hai.

### D.4 — Ngân Sách & Quản Trị Traffic Suy Luận (Reasoning Token Budget)
- **Hệ số Năng Lượng 80x:** Truy vấn Reasoning tiêu tốn năng lượng gấp **~80 lần** so với truy vấn tiêu chuẩn.
- **Thực trạng dữ liệu:** Traffic Reasoning chiếm **24.5% tổng số request**, nhưng chiếm tới **>78% tổng năng lượng tiêu thụ**.
- **Chính sách đề xuất:** Áp dụng Confidence-Gating Router: chỉ kích hoạt chế độ Reasoning khi độ phức tạp truy vấn vượt ngưỡng. Capping reasoning về 10% traffic giúp tiết kiệm thêm hàng nghìn kWh năng lượng và hàng trăm USD mỗi tháng.

### D.5 — Điều Lịch Nhận Thức Carbon (Carbon-Aware Scheduling)
- So sánh lượng phát thải cho 4,227.0 kWh/tháng của 5 batch job:
  - `us-east-1` (Lưới điện hỗn hợp than/khí): **1,606.3 kg CO2e/tháng** ($507.24 tiền điện)
  - `europe-north1` (Thủy điện Na Uy sạch 100%): **126.8 kg CO2e/tháng** ($380.43 tiền điện)
  - `us-east-wa` (Thủy điện Washington): **380.4 kg CO2e/tháng** ($232.49 tiền điện - rẻ nhất)
- **Hiệu quả:** Di chuyển job training/eval sang `europe-north1` giúp **cắt giảm 1,479.5 kg CO2e/tháng (-92.1%)**, đồng thời giảm chi phí điện 25%.

---

## 5. Kế Hoạch Hành Động Dành Cho FinOps Lead (Top 3 Recommendations)

Nếu đảm nhiệm vai trò FinOps Lead tại NimbusAI, 3 hành động tiên quyết cần triển khai trong 30 ngày đầu:

1. **Tuần 1 — Triển Khai Semantic Router & Prompt Caching (Zero-Risk, High-ROI):**
   - Bật Prompt Caching trên toàn bộ System Prompts và Document Context.
   - Cấu hình LiteLLM/vLLM Semantic Router để điều hướng các truy vấn phân loại, trích xuất sang Small Model.
   - *Mục tiêu:* Hạ ngay đơn giá phục vụ inference từ **$6.488 xuống $1.126 / 1M-token**.

2. **Tuần 2 — Kích Hoạt Daemon Dọn Dẹp GPU Rảnh Rỗi & Phân Bổ Showback/Chargeback:**
   - Cài đặt daemon tự động thu hồi instance rảnh rỗi (idle > 2h), thu hồi ngay $600/tháng lãng phí.
   - Khóa cổng Tag Coverage ở mức $\ge 80\%$ (hiện đạt 92%) và kích hoạt cơ chế Chargeback theo chuẩn FOCUS (`outputs/focus_export.csv`) về 4 team (`assistant`, `search`, `eval`, `rag`) để nâng cao trách nhiệm tài chính của từng nhóm phát triển.

3. **Tháng 1 — Tái Cấu Trúc Hợp Đồng Mua Sắm (Spot + 3-Year Reserved Commitment):**
   - Ký kết hợp đồng Reserved 3 năm cho các dịch vụ cốt lõi chạy 24/7 (`job-infer-chat`, `job-infer-rag`, `job-infer-search`).
   - Tích hợp framework checkpointing tự động vào pipeline Airflow/Kubeflow để đưa 100% training batch jobs sang Spot Instances.
   - *Mục tiêu:* Hiện thực hóa khoản tiết kiệm **$10,040 / tháng** trên hợp đồng cloud.

---

_Báo cáo này được chuẩn bị hoàn chỉnh cho Milestone 2 và các phiên thẩm định kiến trúc hạ tầng của NimbusAI._
