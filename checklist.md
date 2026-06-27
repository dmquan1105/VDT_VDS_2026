# Checklist Week 3 - Track B Supervised

Mục tiêu Week 3: xây supervised fraud model trên `dts_train`, hiệu chỉnh xác suất, đổi sang DTS 0-1000, và chuẩn bị model card/reason codes để review với mentor.

Trạng thái file này: kế hoạch thực hiện, không phải log việc đã hoàn thành.

## 0. Setup Và Reproducibility

- [ ] Kiểm tra dependency cần cho Track B: `scikit-learn`, `pyarrow`, `matplotlib`, `seaborn`, `shap`.
- [ ] Nếu dùng LightGBM/XGBoost thì thêm dependency rõ ràng; nếu không cài được, dùng `HistGradientBoostingClassifier` hoặc `GradientBoostingClassifier` của sklearn.
- [ ] Tạo thư mục output nếu chưa có: `outputs/`, `figures/`, `artifacts/`, `reports/`.
- [ ] Đặt seed cố định cho CV/model.
- [ ] Ghi lại lệnh chạy tái lập được trong README hoặc model card.

Acceptance check:

- [ ] Có một entrypoint chạy lại được pipeline supervised, ưu tiên `scripts/train_supervised.py`.
- [ ] Notebook/script có thể restart/run lại từ đầu, không phụ thuộc biến tạm trong kernel.

## 1. Feature Matrix Supervised

Mục tiêu: tạo `train_features` và `holdout_features` cùng schema, một dòng mỗi `CustomerID`.

- [ ] Build feature matrix cho `dts_train` và `dts_holdout`.
- [ ] Gộp feature từ `sim_events`, `device_sessions`, `kyc_records`, `device_catalog`.
- [ ] Đảm bảo một dòng / `CustomerID`.
- [ ] Loại khỏi input model: `CustomerID`, `FraudFlag`, `FraudType`, `Churn`.
- [ ] Tạo feature nhiều cửa sổ thời gian bắt buộc: 7/30/90 ngày cho các tín hiệu phù hợp.
- [ ] Join `device_catalog` qua TAC và tạo tín hiệu emulator/clone/low-tier/rooted/shared-device.
- [ ] Kiểm tra train/holdout có cùng cột feature, cùng dtype hợp lý.
- [ ] Lưu feature snapshot ra `outputs/`.

Output:

- `outputs/train_features_supervised.parquet`
- `outputs/holdout_features_supervised.parquet`
- `outputs/supervised_feature_schema.csv`

Acceptance check:

- [ ] `train_features` có 51,047 dòng.
- [ ] `holdout_features` có 20,000 dòng.
- [ ] `CustomerID` unique ở cả hai file.
- [ ] Không có label leakage trong input model.

## 2. Validation Và Metrics

Đây là phần cần làm sớm để tránh leakage.

- [ ] Dùng `StratifiedKFold(n_splits=5)` theo `FraudFlag`.
- [ ] Fit preprocessing/model bên trong từng fold.
- [ ] Dùng cùng CV split cho Logistic và Gradient Boosting để so sánh công bằng.
- [ ] Tính PR-AUC, ROC-AUC, recall@5%, precision@5%.
- [ ] Tính recall@5% theo từng `FraudType`.
- [ ] Lưu OOF predictions kèm fold/model/prediction.
- [ ] Chọn metric chính: PR-AUC và recall@5% vì fraud chỉ khoảng 3.2%.

Output:

- `outputs/supervised_oof_predictions.csv`
- `outputs/supervised_cv_metrics.csv`
- `outputs/fraudtype_error_analysis.csv`

Acceptance check:

- [ ] Mọi transform có khả năng học tham số đều fit trong fold.
- [ ] OOF prediction có đủ số dòng train và không trùng `CustomerID`.
- [ ] Review rate 5% được tính bằng top 5% score trong từng OOF/model.

## 3. Logistic Regression Baseline

Mục tiêu: baseline dễ giải thích và đối chiếu với nghiệp vụ.

- [ ] Chạy Logistic Regression baseline với `class_weight="balanced"` hoặc sample weight.
- [ ] Preprocessing numeric/categorical rõ ràng.
- [ ] Thêm WOE encoding cho categorical/numeric bins.
- [ ] Đọc coefficient/top features và đối chiếu trực giác nghiệp vụ.
- [ ] So sánh với mốc mentor: Logistic PR-AUC khoảng 0.61.

Output:

- `outputs/logistic_cv_metrics.csv`
- `outputs/logistic_coefficients.csv`
- `artifacts/logistic_pipeline.joblib`

Acceptance check:

- [ ] Logistic chạy đủ 5 fold.
- [ ] Có bảng coefficient/reasonable direction cho các feature quan trọng.

## 4. Gradient Boosting Model Chính

Mục tiêu: model chính để vượt baseline Logistic.

- [ ] Chạy Gradient Boosting baseline bằng sklearn.
- [ ] Thử LightGBM hoặc XGBoost nếu cài được; nếu không, ghi rõ fallback.
- [ ] Xử lý imbalance bằng class weight, sample weight, `scale_pos_weight`, hoặc threshold/review-rate analysis.
- [ ] Tuning nhẹ, ưu tiên ít tham số nhưng có validation rõ.
- [ ] So sánh công bằng với Logistic bằng cùng CV split.
- [ ] Mục tiêu tham chiếu: Gradient Boosting mentor PR-AUC khoảng 0.76, AUC khoảng 0.94.
- [ ] Mục tiêu cải thiện: vượt PR-AUC 0.76 hoặc tăng recall `sim_swap_ato` tại review 5%.

Output:

- `outputs/gbm_cv_metrics.csv`
- `outputs/model_comparison.csv`
- `artifacts/final_model.joblib`

Acceptance check:

- [ ] Có bảng model comparison gồm Logistic và GBM.
- [ ] Final model được chọn dựa trên OOF PR-AUC và recall@5%, không dựa trên holdout label.

## 5. Error Analysis Theo FraudType

Mục tiêu: biết model bắt tốt/yếu loại fraud nào.

- [ ] Tính recall@5% cho `sim_swap_ato`.
- [ ] Tính recall@5% cho `mule`.
- [ ] Tính recall@5% cho `device_farm`.
- [ ] Tính recall@5% cho `subscription_fraud`.
- [ ] Xem false negatives gần ngưỡng top 5%.
- [ ] Phân tích kỹ `sim_swap_ato` và `mule` vì đây là hai archetype mentor nhấn mạnh.
- [ ] Ghi nhận pattern của hard negatives/false positives nếu thấy rõ.

Output:

- `outputs/fraudtype_error_analysis.csv`
- `outputs/top5_review_analysis.csv`

Acceptance check:

- [ ] Có bảng theo `FraudType` gồm count, recall@5%, missed_count.
- [ ] Có nhận xét ngắn về archetype yếu nhất và feature cần bổ sung.

## 6. Calibration

Mục tiêu: biến raw model score thành xác suất đáng tin hơn.

- [ ] Thử Platt/sigmoid calibration.
- [ ] Thử isotonic calibration nếu đủ mẫu trong fold.
- [ ] So sánh Brier score và reliability curve.
- [ ] Chọn calibrator ổn định, không chỉ tối ưu một fold.
- [ ] Tránh calibration leakage: calibrator phải học từ prediction/validation đúng cách, không fit trên cùng data mà raw model đã fit.

Output:

- `outputs/calibration_metrics.csv`
- `figures/reliability_curve.png`
- `artifacts/final_calibrator.joblib`

Acceptance check:

- [ ] Có lý do chọn sigmoid/isotonic.
- [ ] Calibrated `P_fraud` nằm trong `[0, 1]`.

## 7. Scorecard DTS 0-1000 Theo PDO

Mục tiêu: chuyển calibrated `P_fraud` thành điểm DTS.

- [ ] Input duy nhất cho scorecard supervised: calibrated `P_fraud`.
- [ ] Quy ước: `P_fraud` cao thì DTS thấp.
- [ ] Dùng base score 600, PDO 50; nếu đổi thì ghi rõ lý do.
- [ ] Clip DTS vào `[0, 1000]`.
- [ ] Check monotonicity giữa `P_fraud` và DTS.

Output:

- `outputs/calibrated_scorecard.csv`

Acceptance check:

- [ ] DTS nằm trong `[0, 1000]`.
- [ ] Customer có `P_fraud` cao hơn không được có DTS cao hơn nếu chỉ khác theo scorecard transform.

## 8. SHAP Và Reason Codes

Mục tiêu: giải thích model cho reviewer.

- [ ] Tạo global SHAP feature importance cho final GBM.
- [ ] Tạo local SHAP cho case study.
- [ ] Sinh 3-5 reason codes cho mỗi customer holdout.
- [ ] Map feature kỹ thuật thành câu nghiệp vụ dễ hiểu.
- [ ] Nếu SHAP gặp vấn đề với model fallback, dùng permutation importance cho global và heuristic top-contribution/reason mapping, nhưng phải ghi rõ limitation.

Reason code examples:

- SIM swap gần đây.
- Tỷ lệ IP datacenter cao.
- Thiết bị dùng chung nhiều account.
- KYC yếu.
- Geo-velocity bất thường.

Output:

- `outputs/shap_feature_importance.csv`
- `outputs/reason_codes_holdout.csv`
- `figures/shap_summary.png`

Acceptance check:

- [ ] Reason codes không hiện tên feature thô khó hiểu nếu đưa vào report.
- [ ] Mỗi reason code có mapping từ feature nguồn sang ý nghĩa nghiệp vụ.

## 9. Case Studies

Mục tiêu: minh họa local explainability.

- [ ] Chọn ít nhất 3 case study đúng yêu cầu report.
- [ ] Ưu tiên có mặt `sim_swap_ato`, `mule`, `device_farm`.
- [ ] Nếu có thời gian, thêm `subscription_fraud` vào appendix.
- [ ] Mỗi case gồm: raw signals, model score, DTS, SHAP/reason codes, kết luận reviewer.

Output:

- `reports/week3_case_studies.md` hoặc một section trong model card.
- `figures/case_*` nếu cần visualization.

Acceptance check:

- [ ] Mỗi case nói được vì sao model score cao/thấp.
- [ ] Có ít nhất một case false negative hoặc gần ngưỡng nếu có thời gian.

## 10. Final Train Và Draft Holdout Scoring

Mục tiêu: tạo pipeline score holdout sẵn sàng cho Track B; file này có thể dùng làm draft cho lần nộp cuối ở Week 4.

- [ ] Chọn final model bằng OOF PR-AUC + recall@5%.
- [ ] Refit preprocessing/model trên toàn bộ `dts_train`.
- [ ] Score `dts_holdout`.
- [ ] Apply calibration.
- [ ] Convert `P_fraud` sang DTS.
- [ ] Save draft submission.

Output:

- `outputs/track_b_holdout_submission.csv`

Schema:

- `CustomerID`
- `P_fraud`
- `DTS`

Acceptance check:

- [ ] 20,000 rows.
- [ ] `CustomerID` unique.
- [ ] `P_fraud` in `[0, 1]`.
- [ ] `DTS` in `[0, 1000]`.
- [ ] No missing.

## 11. Model Card Và Báo Cáo Week 3

Mục tiêu: có tài liệu nộp mentor và để mình giải thích được pipeline.

Nội dung cần có:

- [ ] Data và label distribution.
- [ ] Feature groups.
- [ ] Validation design.
- [ ] Leakage controls.
- [ ] Logistic vs Gradient Boosting comparison.
- [ ] Imbalance handling.
- [ ] Error analysis theo `FraudType`.
- [ ] Calibration result.
- [ ] PDO scorecard definition.
- [ ] SHAP summary.
- [ ] Reason codes.
- [ ] Limitations.
- [ ] Monitoring: PSI, calibration drift, fraud-type drift.

Output:

- `reports/week3_model_card.md` hoặc `.typ`
- `reports/week3_model_card.pdf`

Acceptance check:

- [ ] Mọi con số trong report trace được về file output/script.
- [ ] Có câu trả lời cho checkpoint: vì sao PR-AUC phù hợp hơn ROC-AUC khi positive rate khoảng 3.2%.
- [ ] Có câu trả lời cho checkpoint: calibration thay đổi gì khi DTS được dùng làm credit input.

## 12. Stretch Sau Khi Đạt Baseline

Chỉ làm nếu các mục bắt buộc đã xong và chạy lại được.

- [ ] Thêm interaction features cho `sim_swap_ato` và `mule`.
- [ ] Thử feature selection hoặc regularization nhẹ.
- [ ] Thử ensemble Logistic + GBM nếu có lợi ích rõ trên OOF.
- [ ] Thêm PSI train-vs-holdout cho feature chính và score.
