# Business Understanding

## 1. Device Trust Score (DTS)

### DTS là gì?

Device Trust Score (DTS) là một điểm số tổng hợp (0–1000) lượng hóa mức độ tin cậy của một thiết bị/thuê bao, được xây dựng từ dữ liệu viễn thông và dữ liệu định danh.

## 2. Use Cases

### KYC

DTS hỗ trợ KYC theo mức rủi ro (risk-based). Thiết bị có điểm cao được luồng eKYC nhanh; điểm thấp phải bổ sung xác minh đầy đủ hoặc review thủ công.

### Fraud Detection

Kết hợp DTS với xác suất gian lận từ mô hình để ra quyết định approve / step-up authentication / review / decline, với ràng buộc vận hành là tỷ lệ review không vượt quá 5%. (Tỷ lệ review = số giao dịch bị review thủ công / tổng số giao dịch).

### Credit Input

DTS được dùng làm một biến đầu vào (feature) cho mô hình chấm điểm tín dụng - không thay thế điểm tín dụng. Yêu cầu: điểm phải được hiệu chỉnh (calibrated) và ổn định theo thời gian (kiểm tra PSI).

## 3. Device Fingerprinting

Khi con người giao tiếp với nhau, họ sẽ để lại dấu vết (footprint) về hành vi, thói quen, sở thích, v.v. Tương tự, khi một thiết bị được sử dụng, nó cũng sẽ để lại dấu vết kỹ thuật số (digital footprint) thông qua các tín hiệu và hành vi của nó. Device Intelligence là quá trình thu thập, phân tích và hiểu các tín hiệu này để đánh giá mức độ tin cậy của thiết bị.

## 5. eKYC

### KYC

Know Your Customer (KYC) là quy trình dùng để xác minh danh tính của khách hàng trong tài chính, ngân hàng. Ngoài ra, đây cũng là một công cụ giúp các tổ chức này đề phòng những rủi ro, gian lận và đánh giá sát sao những hoạt động giao dịch của khách hàng.

### eKYC

Electronic KYC: là một quy trình xác minh danh tính khách hàng trực tuyến. Thay vì phải điền vào các biểu mẫu dài dòng và mang theo nhiều giấy tờ, khách hàng chỉ cần sử dụng điện thoại thông minh để chụp ảnh chân dung và giấy tờ tùy thân. Hệ thống eKYC sẽ sử dụng các thuật toán thông minh để so sánh hình ảnh của khách hàng với cơ sở dữ liệu và xác nhận danh tính một cách nhanh chóng và an toàn.

## 6. Fraud Archetypes

### SIM Swap ATO (30%)

Chiếm đoạt tài khoản qua đổi SIM: SIM rất mới, vừa swap, nhiều IP, có cảnh báo geo-velocity.

#### Mô tả

SIM Swap ATO (Account Takeover) là hình thức chiếm đoạt tài khoản bằng cách giành quyền kiểm soát số điện thoại của nạn nhân thông qua quá trình đổi SIM. Sau khi thực hiện thành công SIM swap, kẻ gian có thể nhận OTP và các mã xác thực gửi qua SMS để truy cập vào tài khoản ngân hàng, ví điện tử hoặc các dịch vụ trực tuyến khác.

#### Attack Flow

1. Kẻ gian thu thập thông tin cá nhân của nạn nhân.
2. Kẻ gian yêu cầu nhà mạng thực hiện SIM swap hoặc cấp lại SIM.
3. SIM gốc của nạn nhân bị vô hiệu hóa.
4. Kẻ gian nhận được SIM mới gắn với số điện thoại của nạn nhân.
5. OTP và mã xác thực được gửi đến SIM mới.
6. Kẻ gian đăng nhập và chiếm đoạt tài khoản (Account Takeover).

#### Đặc điểm hành vi

Các trường hợp SIM Swap ATO thường có một số đặc điểm:

- Xuất hiện hoạt động đăng nhập hoặc giao dịch ngay sau khi xảy ra SIM swap.
- Thiết bị hoặc mạng truy cập thay đổi bất thường so với lịch sử trước đó.
- Số lượng địa chỉ IP tăng đột biến trong thời gian ngắn.
- Xuất hiện các truy cập từ vị trí địa lý xa lạ hoặc không phù hợp với lịch sử sử dụng của khách hàng.
- Có dấu hiệu "impossible travel" hoặc geo-velocity bất thường, ví dụ các phiên đăng nhập từ hai quốc gia cách xa nhau trong khoảng thời gian không khả thi về mặt di chuyển.
- Hành vi sử dụng sau SIM swap khác đáng kể so với hành vi thông thường của chủ thuê bao.

### Device Farm (25%)

Farm máy ảo / danh tính tổng hợp: emulator hoặc rooted, IP datacenter, nhiều SIM trên một máy, KYC yếu, điểm khớp giấy tờ thấp.

#### Mô tả

Device Farm là hình thức gian lận sử dụng một hoặc nhiều thiết bị vật lý hoặc máy ảo (emulator) để vận hành số lượng lớn tài khoản. Các thiết bị này thường được quản lý tập trung, sử dụng nhiều SIM hoặc danh tính khác nhau nhằm tạo ra các tài khoản có vẻ độc lập nhưng thực chất do cùng một tổ chức hoặc cá nhân kiểm soát.

Trong môi trường tài chính số, Device Farm thường được sử dụng để thực hiện đăng ký tài khoản hàng loạt, lạm dụng khuyến mãi, tạo danh tính tổng hợp (synthetic identity) hoặc hỗ trợ các hoạt động rửa tiền và gian lận tín dụng.

#### Attack Flow

1. Kẻ gian chuẩn bị hạ tầng gồm nhiều thiết bị vật lý hoặc emulator.
2. Thu thập hoặc tạo ra nhiều danh tính khác nhau.
3. Sử dụng nhiều SIM hoặc số điện thoại để đăng ký tài khoản.
4. Thực hiện eKYC ở mức tối thiểu hoặc sử dụng giấy tờ giả.
5. Vận hành hàng loạt tài khoản từ cùng một cụm thiết bị.
6. Khai thác hệ thống để nhận ưu đãi, thực hiện giao dịch gian lận hoặc tạo đầu vào cho các hình thức gian lận khác.

#### Đặc điểm hành vi

Các trường hợp Device Farm thường có các đặc điểm:

- Một thiết bị được sử dụng bởi nhiều tài khoản khác nhau.
- Một nhóm tài khoản có hành vi hoạt động tương đồng về thời gian và mạng truy cập.
- Thiết bị có dấu hiệu bị can thiệp như root hoặc jailbreak.
- Xuất hiện tỷ lệ cao các kết nối từ datacenter IP, VPN hoặc proxy.
- Nhiều tài khoản mới được tạo trong thời gian ngắn.
- Chất lượng định danh thấp, KYC ở mức cơ bản hoặc có điểm xác thực giấy tờ thấp.
- Hành vi sử dụng có tính lặp lại và tự động hóa cao.
- Nhiều SIM hoặc số điện thoại được gắn với cùng một thiết bị trong khoảng thời gian ngắn.
- Thiết bị thường không có lịch sử sử dụng ổn định như khách hàng thông thường.

### Mule Account (25%)

Tài khoản trung gian: thiết bị gắn nhiều tài khoản, SIM xuất hiện trên nhiều máy, di chuyển nhiều, hoạt động đêm cao.

#### Mô tả

Mule Account (hay Money Mule Account) là tài khoản được sử dụng làm trung gian để nhận, giữ hoặc chuyển tiền cho các hoạt động gian lận hoặc rửa tiền.

Khác với Device Farm hay SIM Swap ATO, chủ sở hữu của Mule Account thường là người thật với danh tính hợp lệ. Trong một số trường hợp, chủ tài khoản biết rõ mình đang tham gia hoạt động bất hợp pháp; trong các trường hợp khác, họ bị lừa hoặc bị thuê để cho mượn tài khoản.

Do sử dụng danh tính thật và có lịch sử hoạt động hợp pháp, Mule Account thường khó phát hiện hơn các hình thức gian lận khác.

#### Attack Flow

1. Kẻ gian tuyển dụng hoặc lợi dụng một người đóng vai trò "money mule".
2. Tiền từ các hoạt động gian lận được chuyển vào tài khoản của mule.
3. Mule thực hiện chuyển tiếp tiền tới các tài khoản khác hoặc rút tiền mặt.
4. Quá trình được lặp lại qua nhiều lớp trung gian nhằm che giấu nguồn gốc dòng tiền.
5. Tài khoản mule trở thành mắt xích trung chuyển trong mạng lưới gian lận hoặc rửa tiền.

#### Đặc điểm hành vi

Các Mule Account thường có các đặc điểm sau:

- Thiết bị hoặc số điện thoại có liên hệ với nhiều tài khoản khác nhau.
- Xuất hiện trên nhiều thiết bị hoặc nhiều SIM hơn mức thông thường của một khách hàng cá nhân.
- Hoạt động ở nhiều địa điểm địa lý khác nhau trong thời gian ngắn.
- Tỷ lệ hoạt động vào ban đêm cao hơn nhóm khách hàng thông thường.
- Hành vi sử dụng thiếu tính ổn định, thay đổi thiết bị hoặc môi trường truy cập thường xuyên.
- Có dấu hiệu chia sẻ thiết bị hoặc chia sẻ số điện thoại giữa nhiều tài khoản.
- Các đặc điểm định danh (KYC) thường hợp lệ, khiến tài khoản khó bị phát hiện bằng các biện pháp xác minh danh tính truyền thống.

#### Vì sao Mule Account khó phát hiện?

Mule Account thường sử dụng danh tính thật và có lịch sử hoạt động hợp pháp. Các tín hiệu bất thường chủ yếu xuất hiện ở mức hành vi (behavioral signals) thay vì mức định danh (identity signals). Vì vậy các phương pháp phát hiện bất thường đơn giản hoặc chỉ dựa trên KYC thường có hiệu quả hạn chế đối với nhóm này.

### Subscription Fraud (20%)

Đăng ký gian lận: mọi thứ đều mới (số, SIM, thiết bị), điểm KYC trung bình. Cố ý thiết kế để gần như không phải outlier - anomaly detection rất khó bắt.

#### Mô tả

Subscription Fraud là hình thức gian lận trong đó kẻ gian đăng ký tài khoản hoặc dịch vụ mới bằng danh tính giả, danh tính tổng hợp (synthetic identity) hoặc thông tin bị đánh cắp nhằm đạt được quyền truy cập vào sản phẩm tài chính hoặc dịch vụ viễn thông.

Khác với các hình thức gian lận dựa trên chiếm đoạt tài khoản hiện hữu, Subscription Fraud xảy ra ngay từ giai đoạn onboarding. Kẻ gian cố gắng tạo ra một hồ sơ khách hàng có vẻ hợp lệ để vượt qua các bước kiểm tra ban đầu.

#### Attack Flow

1. Kẻ gian chuẩn bị danh tính giả, danh tính tổng hợp hoặc thông tin cá nhân bị đánh cắp.
2. Đăng ký số điện thoại, SIM và thiết bị mới.
3. Thực hiện KYC hoặc eKYC ở mức đủ để vượt qua quy trình onboarding.
4. Tạo tài khoản và bắt đầu sử dụng dịch vụ.
5. Thực hiện hành vi gian lận sau khi tài khoản đã được chấp nhận.

#### Đặc điểm hành vi

Các trường hợp Subscription Fraud thường có các đặc điểm sau:

- Số điện thoại, SIM và thiết bị đều mới được tạo hoặc mới xuất hiện trong hệ thống.
- Thiếu lịch sử hoạt động dài hạn để chứng minh độ tin cậy.
- Thông tin định danh có vẻ hợp lệ nhưng chưa đủ mạnh để tạo sự tin tưởng cao.
- Chất lượng KYC thường ở mức trung bình thay vì quá thấp hoặc quá cao.
- Không xuất hiện nhiều dấu hiệu bất thường rõ rệt về thiết bị, vị trí địa lý hay mạng truy cập.
- Hành vi ban đầu được thiết kế để giống khách hàng mới hợp pháp nhằm tránh bị phát hiện.
- Nhiều tín hiệu rủi ro chỉ xuất hiện khi xem xét đồng thời nhiều khía cạnh của hồ sơ khách hàng thay vì từ một dấu hiệu riêng lẻ.

#### Phân biệt với khách hàng mới hợp pháp

Một thách thức quan trọng của Subscription Fraud là sự chồng lấn lớn với nhóm khách hàng mới hợp pháp. Việc sử dụng các quy tắc đơn giản như "SIM mới" hoặc "thiết bị mới" có thể dẫn đến tỷ lệ false positive cao và làm ảnh hưởng đến trải nghiệm onboarding của khách hàng thật.
