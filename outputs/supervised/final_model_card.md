# Final Model Card - Device Trust Score

## Model name

`xgb_tuned`.

## Intended use

- Chấm `P_fraud` đã calibration và `DTS` 0-1000 cho tập holdout/production.
- Hỗ trợ KYC, fraud review và credit input theo decision matrix.
- Fraud review bị giới hạn theo operating point top 5% risk.

## Not intended use

- Không dùng DTS để auto reject tín dụng một mình.
- Không dùng score thay thế điều tra fraud, KYC policy, hoặc credit score truyền thống.
- Không dùng cho population khác nếu chưa kiểm tra drift, calibration và proxy bias.

## Data

Train có 51,047 khách hàng có nhãn; holdout có 20,000 khách hàng. Event tables SIM, session, KYC và catalog được tổng hợp về cấp `CustomerID`.

## Validation and metrics

Theo summary hiện có: xgb_tuned AUC khoảng 0.954, PR-AUC khoảng 0.797, Brier khoảng 0.0112, Precision@5% khoảng 0.508; stacking_tuned nhỉnh hơn rất nhỏ nhưng phức tạp hơn để vận hành.

Metric chi tiết cần lấy lại từ notebook/output supervised nếu muốn audit từng fold.

## Calibration

Pipeline dùng xác suất fraud đã calibration trước khi đổi sang PDO/DTS. `P_fraud` cao tương ứng `DTS` thấp.

## Decision thresholds

- `kyc_fast_cutoff`: 1000.00000000
- `kyc_reject_cutoff`: 766.00000000
- `fraud_stepup_cutoff_p`: 0.00826757
- `fraud_review_cutoff_p`: 0.09120521
- `fraud_decline_cutoff_p`: 1.00000000
- `credit_low_cutoff`: 858.00000000
- `credit_high_cutoff`: 1000.00000000
- `expected_review_rate`: 5.00%
- `manual_review_rate`: 4.06%
- `decline_or_block_rate`: 1.06%
- `manual_review_or_decline_rate`: 5.13%

## Reason for model choice

`xgb_tuned` được chọn mặc định vì hiệu năng gần tương đương stacking nhưng đơn giản hơn để deploy, kiểm thử và giải thích bằng SHAP trực tiếp.

## Limitations

- Dữ liệu mang tính synthetic/competition nên cần kiểm chứng ngoài thực tế.
- Label fraud có thể đến trễ, nhất là mule/subscription fraud.
- Một số feature có thể là proxy cho hành vi hợp pháp như roaming, port-in hoặc thiết bị giá rẻ.

## Monitoring

Theo dõi PSI feature/score, calibration drift, fraud rate theo DTS bucket, review precision, recall theo FraudType khi label đến, và các feature dễ bị game như SIM swap, shared device, VPN/datacenter, KYC missing.
