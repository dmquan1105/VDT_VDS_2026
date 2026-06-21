# Feature Specification

## Device Trust Score (DTS)

### Phiên bản

v1.1

### Mục tiêu

Tài liệu này mô tả bộ feature được thiết kế cho bài toán Device Trust Score (DTS). Các feature được xây dựng ở mức `CustomerID`, phục vụ phát hiện rủi ro gian lận và chấm điểm độ tin cậy thiết bị trong bối cảnh dữ liệu viễn thông, thiết bị, hành vi truy cập và KYC.

Các nguồn dữ liệu chính:

- `dts_train`
- `dts_holdout`
- `sim_events`
- `device_sessions`
- `kyc_records`
- `device_catalog`

Mục tiêu của bộ feature là lượng hóa các nhóm tín hiệu rủi ro:

1. Identity Confidence
2. SIM Stability
3. Device Integrity
4. Behavioral Consistency

Các tín hiệu này hỗ trợ phát hiện bốn fraud archetype chính:

- SIM Swap ATO
- Device Farm
- Mule Account
- Subscription Fraud

### Nguyên tắc xây dựng

- Mỗi feature được tổng hợp về cấp khách hàng thông qua khóa `CustomerID`.
- `reference_date` được xác định là ngày mới nhất giữa `sim_events.EventDate` và `device_sessions.SessionDate`.
- Các feature hành vi được tính trên toàn bộ cửa sổ quan sát và bổ sung thêm cửa sổ 30 ngày gần nhất để phản ánh rủi ro mới phát sinh.
- Pipeline có thể giữ lại các biến hồ sơ khách hàng từ `customer_df` nếu bật `include_customer_features=True`; các nhãn `FraudFlag`, `FraudType`, `Churn` được loại khỏi nhóm feature đầu vào.

---

# 1. Identity Confidence

Đánh giá mức độ đáng tin cậy của danh tính khách hàng dựa trên mức KYC và kết quả đối sánh giấy tờ, khuôn mặt.

| Feature              | Công thức / logic                  | Lý do nghiệp vụ                                             | Lớp dữ liệu |
| -------------------- | ---------------------------------- | ----------------------------------------------------------- | ----------- |
| `kyc_level_ord`      | `none=0, basic=1, full=2`          | KYC mạnh giúp giảm khả năng danh tính giả                   | Identity    |
| `has_face_score`     | `FaceMatchScore IS NOT NULL`       | Sự tồn tại của FaceMatchScore phản ánh mức xác minh cao hơn | Identity    |
| `has_iddoc_score`    | `IDDocMatchScore IS NOT NULL`      | Có xác minh giấy tờ là tín hiệu nhận dạng mạnh              | Identity    |
| `face_match_score`   | `FaceMatchScore`                   | Điểm khớp khuôn mặt với giấy tờ                             | Identity    |
| `id_doc_match_score` | `IDDocMatchScore`                  | Điểm khớp giấy tờ định danh                                 | Identity    |

---

# 2. SIM Stability

Đánh giá mức độ ổn định của SIM và số điện thoại. Nhóm feature này tập trung vào tuổi số điện thoại, lịch sử đổi SIM, số lượng ICCID và tín hiệu chuyển mạng giữ số.

| Feature                    | Công thức / logic                                      | Lý do nghiệp vụ                                                       | Lớp dữ liệu |
| -------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------- | ----------- |
| `phone_number_age_days`    | `reference_date - min(number_activation.EventDate)`    | SIM hoặc số điện thoại mới thường xuất hiện trong các fraud archetype | SIM         |
| `sim_swap_count_total`     | `count(EventType='sim_swap')`                          | Tổng số lần đổi SIM phản ánh mức bất ổn dài hạn                       | SIM         |
| `sim_swap_count_90d`       | `count(sim_swap trong 90 ngày gần nhất)`               | Đổi SIM trong giai đoạn gần đây là tín hiệu rủi ro mạnh               | SIM         |
| `sim_swap_count_12m`       | `count(sim_swap trong 365 ngày gần nhất)`              | Bổ sung góc nhìn trung hạn về lịch sử thay SIM                        | SIM         |
| `days_since_last_sim_swap` | `reference_date - max(sim_swap.EventDate)`             | SIM vừa bị swap gần đây liên quan trực tiếp tới ATO                   | SIM         |
| `iccid_count`              | `nunique(ICCID)`                                       | Nhiều ICCID trên cùng khách hàng phản ánh lịch sử thay SIM            | SIM         |
| `port_in_flag`             | `max(EventType='port_in')`                             | Chuyển mạng giữ số có thể làm thay đổi profile rủi ro                 | SIM         |
| `recent_sim_change_flag`   | `days_since_last_sim_swap <= 30`                       | Rule nghiệp vụ phổ biến để nhận diện đổi SIM gần đây                  | SIM         |

---

# 3. Device Integrity

Đánh giá độ tin cậy, tính ổn định và mức độ bất thường của thiết bị. Nhóm này kết hợp tín hiệu từ session, IMEI, TAC và catalog thiết bị.

| Feature                         | Công thức / logic                                                         | Lý do nghiệp vụ                                                        | Lớp dữ liệu |
| ------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------- |
| `num_imeis_90d`                 | `nunique(IMEI)`                                                           | Đổi nhiều thiết bị trong thời gian quan sát là tín hiệu rủi ro         | Device      |
| `max_customers_per_imei`        | `max(nunique(CustomerID) theo IMEI)`                                      | Thiết bị dùng chung nhiều khách hàng là tín hiệu mạnh của fraud        | Device      |
| `num_accounts_linked_to_device` | Alias của `max_customers_per_imei`                                        | Diễn giải trực tiếp số tài khoản lớn nhất liên kết với thiết bị        | Device      |
| `shared_imei_flag`              | `max_customers_per_imei > 1`                                              | Phát hiện thiết bị được dùng bởi nhiều tài khoản                       | Device      |
| `high_shared_imei_flag`         | `max_customers_per_imei >= 4`                                             | Ngưỡng rủi ro cao dựa trên EDA khi IMEI gắn với nhiều khách hàng       | Device      |
| `rooted_session_ratio`          | `mean(RootedFlag)`                                                        | Thiết bị root/jailbreak thường xuất hiện trong hệ sinh thái gian lận   | Device      |
| `is_rooted`                     | `max(RootedFlag)`                                                         | Cho biết khách hàng từng sử dụng thiết bị root hay chưa                | Device      |
| `observed_device_days`          | `nunique(SessionDate)`                                                    | Đo mức độ hiện diện và ổn định của thiết bị theo thời gian             | Device      |
| `device_tier_mean`              | `mean(DeviceTier)`                                                        | Phân khúc thiết bị trung bình phản ánh chất lượng thiết bị thường dùng | Device      |
| `device_tier_min`               | `min(DeviceTier)`                                                         | Ghi nhận phân khúc thiết bị tốt nhất từng xuất hiện                    | Device      |
| `device_tier_max`               | `max(DeviceTier)`                                                         | Ghi nhận phân khúc thiết bị rủi ro nhất từng xuất hiện                 | Device      |
| `low_tier_session_ratio`        | `mean(DeviceTier >= 3)`                                                   | Tỷ lệ session trên thiết bị low-tier, có thể liên quan tới fraud farm  | Device      |
| `low_tier_device_flag`          | `max(DeviceTier >= 3)`                                                    | Cờ nhận diện khách hàng từng dùng thiết bị low-tier                    | Device      |
| `device_catalog_missing_ratio`  | `mean(missing DeviceTier/DeviceBrand/DeviceModel/DeviceOS)`               | Tỷ lệ session không map được catalog, có thể phản ánh TAC lạ hoặc dữ liệu thiết bị yếu | Device      |
| `device_catalog_missing_any`    | `max(missing DeviceTier/DeviceBrand/DeviceModel/DeviceOS)`                | Cờ nhận diện khách hàng từng có thiết bị thiếu thông tin catalog       | Device      |
| `is_emulator`                   | `max(pattern từ DeviceBrand/DeviceModel/DeviceOS)`                        | Nhận diện emulator, thiết bị generic, clone hoặc OS bất thường         | Device      |
| `emulator_session_ratio`        | `mean(is_emulator_session)`                                               | Đo cường độ sử dụng emulator/generic device                            | Device      |
| `tac_customer_count_max`        | `max(nunique(CustomerID) theo TAC)`                                       | TAC xuất hiện ở nhiều khách hàng có thể là dấu hiệu clone/farm         | Device      |
| `tac_imei_count_max`            | `max(nunique(IMEI) theo TAC)`                                             | TAC có nhiều IMEI giúp nhận diện cụm thiết bị cùng model               | Device      |
| `tac_customer_per_imei_max`     | `max(tac_customer_count / tac_imei_count)`                                | Chuẩn hóa mức độ lan truyền khách hàng trên từng cụm TAC               | Device      |
| `tac_grey_clone_flag`           | `is_emulator OR (low_tier_device_flag AND high_shared_imei_flag) OR (is_rooted AND high_shared_imei_flag)` | Cờ tổng hợp cho thiết bị grey-market, clone, emulator hoặc farm        | Device      |
| `tac_risk_score`                | `2*is_emulator + low_tier_device_flag + high_shared_imei_flag + is_rooted` | Điểm rủi ro thiết bị tổng hợp từ các tín hiệu mạnh                     | Device      |

---

# 4. Behavioral Consistency

Đánh giá mức độ ổn định của hành vi sử dụng theo IP, quốc gia, loại mạng, thời điểm truy cập và cường độ hoạt động. Các feature không có hậu tố `_30d` được tính trên toàn bộ cửa sổ quan sát; các feature có hậu tố `_30d` tập trung vào 30 ngày gần nhất.

## 4.1. Hành vi trên toàn bộ cửa sổ quan sát

| Feature                  | Công thức / logic                                     | Lý do nghiệp vụ                                                      | Lớp dữ liệu |
| ------------------------ | ----------------------------------------------------- | -------------------------------------------------------------------- | ----------- |
| `distinct_ip_count`      | `nunique(IP)`                                         | Xuất hiện trên nhiều IP khác nhau có thể phản ánh hành vi bất thường | Behavior    |
| `distinct_country_count` | `nunique(CountryCode)`                                | Di chuyển qua nhiều quốc gia trong thời gian ngắn làm tăng rủi ro    | Behavior    |
| `datacenter_ratio`       | `mean(IPType='datacenter')`                           | Datacenter IP thường liên quan tới automation hoặc device farm       | Behavior    |
| `vpn_proxy_ratio`        | `mean(IPType='vpn_proxy')`                            | VPN/proxy có thể được dùng để che giấu danh tính hoặc vị trí         | Behavior    |
| `non_residential_ratio`  | `datacenter_ratio + vpn_proxy_ratio`                  | Đo mức độ sử dụng hạ tầng mạng phi dân dụng                          | Behavior    |
| `home_cell_ratio`        | `mean(IsHomeCell)`                                    | Người dùng hợp pháp thường có mức độ ổn định vị trí cao hơn          | Behavior    |
| `night_session_ratio`    | `mean(SessionHour between 0 and 5)`                   | Hoạt động đêm quá nhiều có thể phản ánh automation                   | Behavior    |
| `active_days_90d`        | `nunique(SessionDate)`                                | Số ngày hoạt động trong cửa sổ quan sát                              | Behavior    |
| `total_sessions`         | `count(session)`                                      | Tổng cường độ hoạt động của khách hàng                               | Behavior    |
| `avg_sessions_per_day`   | `total_sessions / active_days_90d`                    | Cường độ sử dụng bất thường có thể là dấu hiệu farm hoặc bot         | Behavior    |
| `geo_velocity_alerts`    | `sum(country_count_per_day > 1)`                      | Số ngày phát sinh impossible travel hoặc location anomaly             | Behavior    |
| `geo_velocity_flag`      | `max(country_count_per_day > 1)`                      | Cờ nhận diện khách hàng từng có nhiều quốc gia trong cùng ngày        | Behavior    |
| `days_since_first_seen`  | `reference_date - min(SessionDate)`                   | Khách hàng/thiết bị mới xuất hiện thường có ít lịch sử tin cậy hơn   | Behavior    |

## 4.2. Hành vi trong 30 ngày gần nhất

| Feature                        | Công thức / logic                                           | Lý do nghiệp vụ                                                          | Lớp dữ liệu |
| ------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------ | ----------- |
| `distinct_ip_30d`              | `nunique(IP trong 30 ngày gần nhất)`                        | Tăng số lượng IP gần đây có thể báo hiệu takeover hoặc automation        | Behavior    |
| `distinct_country_30d`         | `nunique(CountryCode trong 30 ngày gần nhất)`               | Phát hiện thay đổi vị trí địa lý bất thường trong giai đoạn gần          | Behavior    |
| `datacenter_ratio_30d`         | `mean(IPType='datacenter' trong 30 ngày gần nhất)`          | Nhấn mạnh mức sử dụng hạ tầng datacenter mới phát sinh                   | Behavior    |
| `vpn_proxy_ratio_30d`          | `mean(IPType='vpn_proxy' trong 30 ngày gần nhất)`           | Đo mức che giấu vị trí trong hành vi gần đây                             | Behavior    |
| `non_residential_ratio_30d`    | `datacenter_ratio_30d + vpn_proxy_ratio_30d`                | Tổng hợp mức sử dụng mạng phi dân dụng gần đây                           | Behavior    |
| `home_cell_ratio_30d`          | `mean(IsHomeCell trong 30 ngày gần nhất)`                   | Suy giảm ổn định vị trí gần đây có thể là tín hiệu rủi ro                | Behavior    |
| `night_session_ratio_30d`      | `mean(SessionHour between 0 and 5 trong 30 ngày gần nhất)`  | Tăng hoạt động đêm gần đây có thể liên quan tới bot/farm                 | Behavior    |
| `active_days_30d`              | `nunique(SessionDate trong 30 ngày gần nhất)`               | Đo mức độ hiện diện gần đây                                              | Behavior    |
| `total_sessions_30d`           | `count(session trong 30 ngày gần nhất)`                     | Tổng cường độ hoạt động gần đây                                          | Behavior    |
| `avg_sessions_per_day_30d`     | `total_sessions_30d / active_days_30d`                      | Phát hiện burst activity trong giai đoạn ngắn                            | Behavior    |
| `geo_velocity_alerts_30d`      | `sum(country_count_per_day > 1 trong 30 ngày gần nhất)`     | Đếm số ngày có dấu hiệu impossible travel gần đây                        | Behavior    |
| `geo_velocity_flag_30d`        | `max(country_count_per_day > 1 trong 30 ngày gần nhất)`     | Cờ nhận diện location anomaly mới phát sinh                              | Behavior    |

---

# 5. Feature truyền thẳng từ hồ sơ khách hàng

Trong `build_feature_matrix`, nếu `include_customer_features=True`, pipeline giữ lại các biến sẵn có trong `customer_df` sau khi loại bỏ nhãn:

- `FraudFlag`
- `FraudType`
- `Churn`

Nhóm biến này không được tính lại trong `src/features/*_features.py`, nhưng có thể được đưa vào ma trận feature cuối cùng như các tín hiệu nền về hồ sơ thuê bao, nhân khẩu học, dịch vụ hoặc lịch sử tương tác nếu tồn tại trong bảng nguồn.

---

# 6. Liên hệ giữa Feature và Fraud Archetype

## SIM Swap ATO

### Đặc điểm nghiệp vụ

- SIM vừa bị swap hoặc có lịch sử swap dày đặc.
- Số điện thoại mới hoặc thiếu lịch sử ổn định.
- IP, quốc gia và vị trí truy cập thay đổi bất thường.
- Hành vi bất thường xuất hiện mạnh trong 30 ngày gần nhất.

### Feature liên quan

| Feature                    |
| -------------------------- |
| `sim_swap_count_total`     |
| `sim_swap_count_90d`       |
| `sim_swap_count_12m`       |
| `days_since_last_sim_swap` |
| `recent_sim_change_flag`   |
| `phone_number_age_days`    |
| `distinct_ip_count`        |
| `distinct_ip_30d`          |
| `geo_velocity_flag`        |
| `geo_velocity_flag_30d`    |

---

## Device Farm

### Đặc điểm nghiệp vụ

- Rooted/jailbroken device.
- Emulator, thiết bị generic, clone hoặc TAC bất thường.
- Datacenter/VPN/proxy IP.
- Nhiều tài khoản trên cùng thiết bị hoặc cùng cụm TAC.
- KYC yếu hoặc thiếu tín hiệu xác minh mạnh.

### Feature liên quan

| Feature                         |
| ------------------------------- |
| `rooted_session_ratio`          |
| `is_rooted`                     |
| `is_emulator`                   |
| `emulator_session_ratio`        |
| `datacenter_ratio`              |
| `vpn_proxy_ratio`               |
| `datacenter_ratio_30d`          |
| `vpn_proxy_ratio_30d`           |
| `shared_imei_flag`              |
| `high_shared_imei_flag`         |
| `max_customers_per_imei`        |
| `num_accounts_linked_to_device` |
| `tac_customer_count_max`        |
| `tac_customer_per_imei_max`     |
| `tac_grey_clone_flag`           |
| `tac_risk_score`                |

---

## Mule Account

### Đặc điểm nghiệp vụ

- Thiết bị dùng chung giữa nhiều tài khoản.
- SIM xuất hiện qua nhiều ICCID hoặc nhiều thiết bị.
- Hoạt động ở nhiều vị trí/quốc gia.
- Cường độ sử dụng bất thường hoặc hoạt động ban đêm cao.

### Feature liên quan

| Feature                    |
| -------------------------- |
| `shared_imei_flag`         |
| `high_shared_imei_flag`    |
| `num_imeis_90d`            |
| `iccid_count`              |
| `distinct_country_count`   |
| `distinct_country_30d`     |
| `night_session_ratio`      |
| `night_session_ratio_30d`  |
| `active_days_90d`          |
| `active_days_30d`          |
| `avg_sessions_per_day`     |
| `avg_sessions_per_day_30d` |

---

## Subscription Fraud

### Đặc điểm nghiệp vụ

- Danh tính yếu hoặc xác minh KYC thấp.
- SIM/số điện thoại mới.
- Thiết bị mới, low-tier hoặc thiếu lịch sử quan sát.
- Hành vi sử dụng chưa đủ ổn định để tạo niềm tin.

### Feature liên quan

| Feature                    |
| -------------------------- |
| `kyc_level_ord`            |
| `has_face_score`           |
| `has_iddoc_score`          |
| `face_match_score`         |
| `id_doc_match_score`       |
| `phone_number_age_days`    |
| `days_since_first_seen`    |
| `observed_device_days`     |
| `device_tier_mean`         |
| `device_tier_max`          |
| `low_tier_device_flag`     |
| `low_tier_session_ratio`   |
| `device_catalog_missing_ratio` |
| `device_catalog_missing_any`   |

---

# 7. Tổng hợp số lượng feature engineered

```text
Identity Confidence:       5 features
SIM Stability:             8 features
Device Integrity:         22 features
Behavioral Consistency:   25 features
--------------------------------------
Tổng cộng:                60 features
```

Bộ feature này bao phủ bốn fraud archetype chính và tận dụng tín hiệu từ SIM, thiết bị, KYC, TAC/catalog và hành vi truy cập. Các feature mới bổ sung giúp tăng khả năng phát hiện rủi ro theo thời gian gần, nhận diện thiết bị dùng chung ở mức nghiêm trọng, phát hiện emulator/clone device và lượng hóa các cụm thiết bị bất thường theo TAC.
