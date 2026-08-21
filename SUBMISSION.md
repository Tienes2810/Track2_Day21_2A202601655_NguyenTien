# Nộp bài — Lab Day 21 (2A202601655 — Nguyễn Tiến)

## 1. URL repo GitHub công khai

https://github.com/Tienes2810/Track2_Day21_2A202601655_NguyenTien

## 2. Chuỗi ảnh chụp màn hình (đúng thứ tự BTC)

| # | File | Nội dung |
|---|---|---|
| 1 | `submission/01_mlflow_ui.png` | MLflow UI ≥ 5 thí nghiệm, có `accuracy` và `f1_score` |
| 2 | `submission/02_actions_list.png` | Tab Actions: lần 2.998 mẫu xanh, lần dữ liệu xanh, eval gate đỏ |
| 3 | `submission/03_actions_first_run.png` | 4 job xanh (Unit Test, Train, Eval, Deploy) — 2.998 mẫu |
| 4 | `submission/04_actions_data_run.png` | 4 job xanh, tiêu đề commit dữ liệu 5.996 mẫu |
| 5 | `submission/05_actions_eval_gate.png` | Job Eval thất bại với siêu tham số yếu (`max_depth=3`) |
| 6 | `submission/06_curl_health_predict.png` | `GET /health` → `{"status":"ok"}`; `POST /predict` → `{"prediction":0,"label":"thap"}` |
| 7 | `submission/07_s3_dvc_and_model.png` | S3 prefix `dvc/` và `models/latest/model.pkl` |

VM API: `http://100.31.53.55:8000`

## 3. Báo cáo A4

`submission/REPORT.pdf` (1 trang)

## Pipeline đã chạy

- 2.998 mẫu: accuracy **0.6740**, f1 **0.6730** — https://github.com/Tienes2810/Track2_Day21_2A202601655_NguyenTien/actions/runs/32446803931
- Eval gate (siêu tham số yếu): accuracy **0.5580** — Deploy bị chặn — https://github.com/Tienes2810/Track2_Day21_2A202601655_NguyenTien/actions/runs/32447154865
- 5.996 mẫu (commit dữ liệu): accuracy **0.7440**, f1 **0.7429** — https://github.com/Tienes2810/Track2_Day21_2A202601655_NguyenTien/actions/runs/32447337293
