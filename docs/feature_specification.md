# Feature Specification

## Device Trust Score (DTS)

### Phiên bản

v1.0

### Mục tiêu

Tài liệu này mô tả các feature được thiết kế cho bài toán Device Trust Score (DTS).

Các feature được xây dựng từ các nguồn dữ liệu:

- `dts_train`
- `dts_holdout`
- `sim_events`
- `device_sessions`
- `kyc_records`
- `device_catalog`

Mục tiêu của bộ feature là lượng hóa các tín hiệu rủi ro liên quan đến:

1. Identity Confidence
2. SIM Stability
3. Device Integrity
4. Behavioral Consistency

Các tín hiệu này được thiết kế để hỗ trợ phát hiện các fraud archetype chính:

- SIM Swap ATO
- Device Farm
- Mule Account
- Subscription Fraud

---

# 1. Identity Confidence

Đánh giá mức độ đáng tin cậy của danh tính khách hàng.

| Feature                     | Công thức                                 | Lý do nghiệp vụ                                             | Lớp dữ liệu |
| --------------------------- | ----------------------------------------- | ----------------------------------------------------------- | ----------- |
| `kyc_level_ord`             | `none=0, basic=1, full=2`                 | KYC mạnh giúp giảm khả năng danh tính giả                   | Identity    |
| `has_face_score`            | `FaceMatchScore IS NOT NULL`              | Sự tồn tại của FaceMatchScore phản ánh mức xác minh cao hơn | Identity    |
| `has_iddoc_score`           | `IDDocMatchScore IS NOT NULL`             | Có xác minh giấy tờ là tín hiệu nhận dạng mạnh              | Identity    |
| `face_match_score`          | `FaceMatchScore`                          | Điểm khớp khuôn mặt với giấy tờ                             | Identity    |
| `iddoc_match_score`         | `IDDocMatchScore`                         | Điểm khớp giấy tờ định danh                                 | Identity    |
| `identity_confidence_score` | `weighted(KYCLevel, FaceScore, DocScore)` | Tổng hợp độ tin cậy danh tính thành một thước đo duy nhất   | Identity    |

---

# 2. SIM Stability

Đánh giá mức độ ổn định của SIM và số điện thoại.

| Feature                    | Công thức                             | Lý do nghiệp vụ                                                       | Lớp dữ liệu |
| -------------------------- | ------------------------------------- | --------------------------------------------------------------------- | ----------- |
| `phone_number_age_days`    | `reference_date - activation_date`    | SIM hoặc số điện thoại mới thường xuất hiện trong các fraud archetype | SIM         |
| `sim_swap_count_90d`       | `count(EventType='sim_swap')`         | Đổi SIM nhiều lần là dấu hiệu bất thường                              | SIM         |
| `days_since_last_sim_swap` | `reference_date - max(sim_swap_date)` | SIM vừa bị swap gần đây liên quan trực tiếp tới ATO                   | SIM         |
| `iccid_count`              | `nunique(ICCID)`                      | Nhiều ICCID trên cùng khách hàng phản ánh lịch sử thay SIM            | SIM         |
| `port_in_flag`             | `exists(port_in)`                     | Chuyển mạng giữ số gần đây có thể làm thay đổi profile rủi ro         | SIM         |
| `recent_sim_change_flag`   | `days_since_last_sim_swap < 30`       | Rule nghiệp vụ phổ biến trong các hệ thống fraud detection            | SIM         |

---

# 3. Device Integrity

Đánh giá độ tin cậy và tính ổn định của thiết bị.

| Feature                  | Công thức                     | Lý do nghiệp vụ                                                 | Lớp dữ liệu |
| ------------------------ | ----------------------------- | --------------------------------------------------------------- | ----------- |
| `num_imeis_90d`          | `nunique(IMEI)`               | Đổi nhiều thiết bị trong thời gian ngắn là tín hiệu rủi ro      | Device      |
| `max_customers_per_imei` | `max(customer_count_of_imei)` | Thiết bị dùng chung nhiều khách hàng là tín hiệu mạnh của fraud | Device      |
| `shared_imei_flag`       | `max_customers_per_imei > 1`  | Phát hiện shared device                                         | Device      |
| `rooted_session_ratio`   | `mean(RootedFlag)`            | Thiết bị root/jailbreak thường xuất hiện trong fraud ecosystem  | Device      |
| `rooted_flag_any`        | `max(RootedFlag)`             | Cho biết khách hàng từng sử dụng thiết bị root hay chưa         | Device      |
| `device_tier`            | `join(TAC -> DeviceTier)`     | Phân khúc thiết bị phản ánh mức độ tin cậy khác nhau            | Device      |
| `device_age_days`        | `CurrentEquipmentDays`        | Thiết bị rất mới thường có risk profile khác biệt               | Device      |
| `observed_device_days`   | `nunique(SessionDate)`        | Đo độ ổn định và hiện diện của thiết bị theo thời gian          | Device      |

---

# 4. Behavioral Consistency

Đánh giá mức độ ổn định của hành vi sử dụng.

| Feature                  | Công thức                                     | Lý do nghiệp vụ                                                      | Lớp dữ liệu |
| ------------------------ | --------------------------------------------- | -------------------------------------------------------------------- | ----------- |
| `distinct_ip_count`      | `nunique(IP)`                                 | Xuất hiện trên nhiều IP khác nhau có thể phản ánh hành vi bất thường | Behavior    |
| `distinct_country_count` | `nunique(CountryCode)`                        | Di chuyển qua nhiều quốc gia trong thời gian ngắn làm tăng rủi ro    | Behavior    |
| `datacenter_ratio`       | `count(IPType='datacenter') / total_sessions` | Datacenter IP thường liên quan tới automation hoặc device farm       | Behavior    |
| `vpn_proxy_ratio`        | `count(IPType='vpn_proxy') / total_sessions`  | Che giấu danh tính hoặc vị trí địa lý                                | Behavior    |
| `non_residential_ratio`  | `(vpn + datacenter) / total_sessions`         | Đo mức độ sử dụng hạ tầng mạng phi dân dụng                          | Behavior    |
| `home_cell_ratio`        | `mean(IsHomeCell)`                            | Người dùng hợp pháp thường có mức độ ổn định vị trí cao hơn          | Behavior    |
| `night_session_ratio`    | `sessions[0h-5h] / total_sessions`            | Hoạt động đêm quá nhiều có thể phản ánh automation                   | Behavior    |
| `active_days_90d`        | `nunique(SessionDate)`                        | Số ngày hoạt động trong cửa sổ quan sát                              | Behavior    |
| `avg_sessions_per_day`   | `total_sessions / active_days_90d`            | Cường độ sử dụng bất thường có thể là dấu hiệu farm hoặc bot         | Behavior    |
| `geo_velocity_flag`      | `multiple CountryCode in same day`            | Impossible travel hoặc location anomaly                              | Behavior    |

---

# 5. Liên hệ giữa Feature và Fraud Archetype

## SIM Swap ATO

### Đặc điểm nghiệp vụ

- SIM vừa bị swap
- Nhiều IP
- Đổi vị trí bất thường
- Số điện thoại tương đối mới

### Feature liên quan

| Feature                    |
| -------------------------- |
| `sim_swap_count_90d`       |
| `days_since_last_sim_swap` |
| `phone_number_age_days`    |
| `distinct_ip_count`        |
| `geo_velocity_flag`        |

---

## Device Farm

### Đặc điểm nghiệp vụ

- Rooted/Jailbroken device
- Datacenter IP
- Nhiều tài khoản trên cùng thiết bị
- KYC yếu

### Feature liên quan

| Feature                  |
| ------------------------ |
| `rooted_session_ratio`   |
| `rooted_flag_any`        |
| `datacenter_ratio`       |
| `vpn_proxy_ratio`        |
| `shared_imei_flag`       |
| `max_customers_per_imei` |

---

## Mule Account

### Đặc điểm nghiệp vụ

- Thiết bị dùng chung
- SIM xuất hiện trên nhiều máy
- Hoạt động nhiều vị trí
- Hành vi ban đêm cao

### Feature liên quan

| Feature                  |
| ------------------------ |
| `shared_imei_flag`       |
| `num_imeis_90d`          |
| `distinct_country_count` |
| `night_session_ratio`    |
| `active_days_90d`        |

---

## Subscription Fraud

### Đặc điểm nghiệp vụ

- Danh tính yếu
- SIM mới
- Thiết bị mới
- KYC ở mức trung bình

### Feature liên quan

| Feature                     |
| --------------------------- |
| `kyc_level_ord`             |
| `face_match_score`          |
| `iddoc_match_score`         |
| `identity_confidence_score` |
| `phone_number_age_days`     |
| `device_age_days`           |

---

### Tổng số feature baseline

```text
18 features
```

Bộ feature này bao phủ đầy đủ bốn fraud archetype chính và tận dụng các tín hiệu từ SIM, Device, KYC và Behavioral data theo đúng mục tiêu của bài toán Device Trust Score.
