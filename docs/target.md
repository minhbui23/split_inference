# Đề Cương Chi Tiết Các Tham Số Đánh Giá Hiệu Quả Hệ Thống Suy Luận Phân Tán

**Ngày:** 21 tháng 5 năm 2025

**Mục Tiêu Đánh Giá:** Chứng minh các quyết định thiết kế cốt lõi (đa luồng, sử dụng message queue, kiến trúc xử lý phân tán) mang lại lợi ích cụ thể, giúp hệ thống hoạt động hiệu quả, ổn định và có khả năng mở rộng, thay vì chỉ tập trung vào các con số hiệu năng tuyệt đối phụ thuộc vào tài nguyên.

---

## I. Đánh Giá Hiệu Quả Thiết Kế Đa Luồng Tại Client Worker

**Quyết định thiết kế:** Tách biệt luồng I/O (`IOWorker`) và luồng tính toán (`InferenceWorker`) trong mỗi Client Worker, giao tiếp qua hàng đợi nội bộ (`input_data_queue`, `output_data_queue`).
**Mục tiêu của thiết kế:**
* Tối đa hóa việc sử dụng CPU cho tác vụ suy luận (inference) bằng cách giảm thiểu thời gian `InferenceWorker` phải chờ đợi I/O.
* Cho phép `IOWorker` xử lý việc nhận/gửi dữ liệu qua RabbitMQ một cách độc lập, không bị block bởi quá trình tính toán kéo dài của `InferenceWorker`.
* Tạo bộ đệm (buffer) cho dữ liệu giữa I/O và tính toán, giúp điều hòa tốc độ và tăng khả năng đáp ứng của client.

### 1.1. Mức Độ Sử Dụng CPU của Client Pod (CPU Utilization)
* **Cách đo:**
    * Sử dụng `kubectl top pods -n split-infer` (hoặc namespace tương ứng) để theo dõi %CPU utilization của các client pod theo thời gian khi hệ thống đang chịu tải (xử lý video/dữ liệu).
* **Ý nghĩa tham số:**
    * Phản ánh mức độ "bận rộn" của CPU trên client pod.
    * CPU Utilization cao (ví dụ, 70-90% một cách ổn định) khi có tải cho thấy luồng `InferenceWorker` đang được cung cấp đủ công việc và tài nguyên tính toán đang được khai thác hiệu quả.
    * CPU Utilization thấp trong khi vẫn có message chờ xử lý (trong RabbitMQ hoặc hàng đợi nội bộ) có thể chỉ ra vấn đề trong việc điều phối luồng, `IOWorker` không cung cấp kịp dữ liệu, hoặc `InferenceWorker` bị block bởi nguyên nhân khác.
* **Cách chứng minh hiệu quả thiết kế & Điểm tham chiếu:**
    * **Lập luận:** Thiết kế đa luồng cho phép `InferenceWorker` tập trung vào các tác vụ tính toán nặng, trong khi `IOWorker` đảm nhiệm việc giao tiếp mạng. Điều này giúp CPU được sử dụng chủ yếu cho inference, thay vì xen kẽ giữa tính toán và chờ I/O như trong mô hình đơn luồng.
    * **So sánh (với chính nó ở các trạng thái tải khác nhau hoặc so sánh lý thuyết):**
        * Quan sát CPU utilization khi client xử lý nhiều request so với khi ít request. Mức tăng tương ứng với tải cho thấy hệ thống phản ứng tốt.
        * Lý thuyết: So với kịch bản đơn luồng (tuần tự I/O rồi mới tính toán), CPU utilization trung bình của thiết kế đa luồng được kỳ vọng sẽ cao hơn và ổn định hơn do giảm thiểu thời gian CPU ở trạng thái chờ (idle).
    * **Ví dụ diễn giải:** "Khi hệ thống hoạt động dưới tải cao, CPU utilization của các client pod duy trì ổn định ở mức X%, cho thấy `InferenceWorker` được tối ưu để thực hiện tính toán gần như liên tục, minh chứng cho hiệu quả của việc tách biệt luồng I/O và luồng tính toán."

### 1.2. Thời Gian Chờ và Độ Dài Hàng Đợi Nội Bộ Của Client (`input_data_queue`)
* **Cách đo (Cần bổ sung logging trong `client.py` hoặc `Client_worker.py`):**
    * Ghi lại số lượng item trong `input_data_queue` (ví dụ: `input_data_queue.qsize()`) ở đầu mỗi vòng lặp xử lý của `InferenceWorker`.
    * Đo thời gian từ lúc một item được `IOWorker` đưa vào `input_data_queue` (`put`) cho đến khi `InferenceWorker` lấy ra (`get`) để xử lý.
* **Ý nghĩa tham số:**
    * Phản ánh sự cân bằng và hiệu quả phối hợp giữa tốc độ cung cấp dữ liệu của `IOWorker` và tốc độ xử lý của `InferenceWorker`.
    * **Thời gian chờ thấp và độ dài hàng đợi ngắn/ổn định:** Cho thấy `IOWorker` và `InferenceWorker` đang hoạt động đồng bộ tốt, dữ liệu được chuyển giao nhanh chóng, `InferenceWorker` không bị "đói" dữ liệu. Overhead của hàng đợi nội bộ là nhỏ.
    * **Thời gian chờ cao và/hoặc độ dài hàng đợi liên tục tăng:**
        * Nếu CPU của `InferenceWorker` cao: Cho thấy `IOWorker` rất hiệu quả trong việc cung cấp dữ liệu "gối đầu", tạo buffer lớn. `InferenceWorker` là điểm nghẽn do giới hạn tính toán của nó (có thể do tài nguyên hoặc bản chất công việc), nhưng ít nhất nó không bị đói dữ liệu. Thiết kế đa luồng vẫn đang giúp tối đa hóa việc sử dụng CPU.
        * Nếu CPU của `InferenceWorker` thấp: Có thể `IOWorker` chậm, hàng đợi nội bộ quá nhỏ, hoặc có vấn đề về tranh chấp tài nguyên/GIL.
* **Cách chứng minh hiệu quả thiết kế & Điểm tham chiếu:**
    * **Lập luận:** Hàng đợi nội bộ đóng vai trò như một bộ đệm quan trọng, giúp điều hòa tốc độ giữa luồng nhận dữ liệu (I/O-bound) và luồng xử lý (CPU-bound). Mục tiêu là giữ cho hàng đợi này có độ dài ổn định và thời gian chờ của tác vụ trong hàng đợi là tối thiểu so với thời gian xử lý chính của tác vụ đó, đảm bảo `InferenceWorker` luôn có dữ liệu để làm việc.
    * **So sánh tương đối (với thời gian xử lý của `InferenceWorker`):**
        * "Hiệu quả" được xác định khi thời gian chờ trong hàng đợi nội bộ chiếm một tỷ lệ nhỏ so với thời gian xử lý của `InferenceWorker`. Ví dụ: Nếu `InferenceWorker` mất 100ms để xử lý một batch, thời gian chờ 5-10ms (5-10%) trong hàng đợi nội bộ được xem là hiệu quả. Nếu thời gian chờ cũng là 50-100ms (50-100%), cần phân tích thêm (như đã nói ở trên, có thể vẫn là dấu hiệu tốt nếu CPU `InferenceWorker` cao).
    * **Ví dụ diễn giải:** "Thời gian chờ trung bình của một batch dữ liệu trong hàng đợi nội bộ `input_data_queue` là Y ms, trong khi thời gian xử lý trung bình của batch đó bởi `InferenceWorker` là Z ms. Tỷ lệ Y/Z cho thấy mức độ 'gối đầu' dữ liệu hiệu quả, đảm bảo `InferenceWorker` không phải chờ đợi I/O trực tiếp từ RabbitMQ và giảm thiểu thời gian CPU ở trạng thái nghỉ."

---

## II. Đánh Giá Hiệu Quả Sử Dụng Message Queue (RabbitMQ)

**Quyết định thiết kế:** Sử dụng RabbitMQ làm message broker cho việc giao tiếp bất đồng bộ và decoupling giữa các client layer trong pipeline xử lý phân tán.
**Mục tiêu của thiết kế:**
* Tách rời (decouple) các công đoạn xử lý (client layers), cho phép chúng hoạt động độc lập và với tốc độ khác nhau.
* Tạo bộ đệm (buffer) giữa các công đoạn, giúp hệ thống xử lý các biến động về tải hoặc tốc độ xử lý tạm thời.
* Tăng khả năng chịu lỗi: sự cố ở một layer không làm sập ngay lập tức các layer khác.
* Tạo nền tảng cho việc mở rộng từng phần của pipeline.

### 2.1. Thời Gian Lan Truyền Dữ Liệu Giữa Các Layer (Inter-Layer Propagation Time)
* **Cách đo:** Dựa vào log "Packet propagation from prev layer" (tính từ `l1_processed_timestamp`) trong `InferenceWorker` của các client layer > 1.
* **Ý nghĩa tham số:**
    * Đo lường tổng thời gian từ khi một client layer hoàn thành xử lý và gửi dữ liệu đi (publish lên RabbitMQ) cho đến khi client layer tiếp theo nhận được message đó và sẵn sàng cho vào hàng đợi nội bộ của mình để xử lý.
    * Bao gồm: thời gian serialization (`pickle.dumps`), độ trễ mạng giữa client và RabbitMQ server, thời gian message được RabbitMQ xử lý và lưu trữ (nếu có), độ trễ mạng giữa RabbitMQ server và client nhận, thời gian client nhận và deserialization (`pickle.loads`).
* **Cách chứng minh hiệu quả thiết kế & Điểm tham chiếu:**
    * **Lập luận:** Thời gian này là overhead của việc giao tiếp phân tán qua message queue. Mục tiêu là giữ cho overhead này ở mức chấp nhận được so với thời gian xử lý hữu ích của mỗi layer, đổi lại lấy các lợi ích của decoupling và buffering.
    * **Đánh giá "hiệu quả" của một giá trị X (ví dụ: 0.15s):**
        1.  **Phân tích thành phần (nếu có thể):** Ước lượng độ trễ mạng (ping giữa các K8s node), thời gian serialization/deserialization (có thể đo riêng với kích thước dữ liệu trung bình). Phần còn lại có thể là thời gian message nằm trong RabbitMQ (do consumer chậm hoặc RabbitMQ bận) và overhead của thư viện Pika.
        2.  **So sánh tương đối với thời gian xử lý của layer nhận:** Nếu layer nhận xử lý một batch mất 0.5s, thì 0.15s lan truyền (chiếm 30%) có thể là một đánh đổi hợp lý cho sự linh hoạt mà RabbitMQ mang lại. Nhưng nếu layer nhận chỉ xử lý trong 0.1s, thì 0.15s lan truyền là quá lớn, chỉ ra rằng phần giao tiếp đang là điểm nghẽn.
        3.  **Tính ổn định:** Biến động thấp của thời gian lan truyền qua các phép đo khác nhau cho thấy sự ổn định của hệ thống mạng và RabbitMQ.
    * **So sánh lý thuyết (với kịch bản giao tiếp đồng bộ trực tiếp):** Không có RabbitMQ, client A gọi trực tiếp client B (blocking call). Độ trễ có thể thấp hơn (chỉ là độ trễ mạng + ser/deser), NHƯNG client A sẽ bị block hoàn toàn nếu client B chậm hoặc lỗi. Thời gian lan truyền qua RabbitMQ là "chi phí" cho sự tách rời và khả năng chịu lỗi. Cần biện luận chi phí này là xứng đáng.
    * **Ví dụ diễn giải:** "Thời gian lan truyền dữ liệu trung bình giữa Layer X và Layer Y là A giây. Phân tích sơ bộ cho thấy B% thời gian này là do độ trễ mạng và C% là do serialization/deserialization. Phần còn lại là overhead của RabbitMQ và hàng đợi. Với thời gian xử lý trung bình của Layer Y là D giây, overhead giao tiếp chiếm E%, đây là mức chấp nhận được để đạt được lợi ích về decoupling, cho phép các layer hoạt động độc lập và hệ thống có khả năng xử lý biến động tải tốt hơn."

### 2.2. Độ Sâu Hàng Đợi RabbitMQ (`intermediate_queue_X`) và Hành Vi Khi Có Biến Động Tải
* **Cách đo:**
    * Sử dụng RabbitMQ Management UI (thường ở port 15672) để theo dõi số lượng message (Ready, Unacknowledged) trong các hàng đợi `intermediate_queue_` theo thời gian.
    * Hoặc sử dụng API của RabbitMQ (tương tự cách `delete_old_queues` được hiện thực trong `Utils.py`).
* **Ý nghĩa tham số & Cách chứng minh hiệu quả thiết kế:**
    * **Khả năng Buffering và Decoupling:**
        * Khi client producer (ví dụ Layer 1) hoạt động nhanh hơn client consumer (Layer 2), hàng đợi `intermediate_queue_2` sẽ tăng lên. Đây là hành vi **mong muốn và hiệu quả** của RabbitMQ, cho thấy nó đang làm tốt vai trò buffer, giúp Layer 1 không bị chặn và có thể tiếp tục công việc.
        * Khi Layer 2 bắt kịp (hoặc được scale lên), hàng đợi sẽ giảm xuống. Khả năng "co giãn" này của độ sâu hàng đợi chính là minh chứng cho sự decoupling.
    * **Khả năng chịu lỗi tạm thời của consumer:** Nếu một client consumer (ví dụ Layer 2) tạm thời bị chậm hoặc dừng, client producer (Layer 1) vẫn có thể tiếp tục publish message vào hàng đợi (cho đến khi hàng đợi đạt giới hạn lưu trữ). Điều này ngăn chặn sự cố lan truyền ngược (cascading failure).
    * **Xác định điểm nghẽn (Bottleneck Identification):** Nếu một hàng đợi liên tục có độ sâu lớn và không giảm, điều đó chỉ rõ client consumer của hàng đợi đó đang là điểm nghẽn của pipeline.
* **Điểm tham chiếu (So sánh với kịch bản không có queue hoặc queue không hiệu quả):**
    * **Lập luận:** Nếu không có cơ chế hàng đợi hiệu quả, sự không đồng đều về tốc độ xử lý giữa các layer sẽ dẫn đến việc layer nhanh hơn phải chờ layer chậm hơn, làm giảm hiệu suất tổng thể, hoặc gây mất dữ liệu nếu không có buffer.
    * **Ví dụ diễn giải:** "Trong quá trình thử nghiệm với tải đột biến, chúng tôi quan sát thấy hàng đợi `intermediate_queue_X` đã tăng lên để chứa các message chưa được xử lý, cho phép layer trước đó tiếp tục hoạt động mà không bị gián đoạn. Sau khi tải giảm hoặc layer xử lý tiếp theo được bổ sung tài nguyên, hàng đợi đã dần được xử lý hết. Điều này minh chứng vai trò quan trọng của RabbitMQ trong việc đảm bảo tính ổn định và khả năng đáp ứng linh hoạt của pipeline."

---

## III. Đánh Giá Hiệu Quả Thiết Kế Xử Lý Phân Tán

**Quyết định thiết kế:** Chia tác vụ suy luận thành nhiều phần (layers) và xử lý song song/tuần tự trên nhiều Client Worker độc lập, được điều phối bởi một Server trung tâm.
**Mục tiêu của thiết kế:**
* Xử lý được các mô hình hoặc khối lượng công việc lớn hơn khả năng của một máy đơn lẻ.
* Tăng thông lượng (throughput) tổng thể của hệ thống thông qua xử lý song song.
* Cho phép mở rộng (scale) từng phần của pipeline một cách độc lập.

### 3.1. Thông Lượng Tổng Thể (FPS) của Pipeline so với Số Lượng Worker/Stage
* **Cách đo:**
    * Đo FPS tổng thể của pipeline (ví dụ, output của layer cuối cùng, hoặc FPS đầu vào nếu là xử lý stream liên tục) sử dụng `FPSLogger`.
    * Thực hiện các thử nghiệm bằng cách thay đổi số lượng client worker cho một hoặc nhiều layer (ví dụ, thay đổi số `replicas` trong file K8s deployment cho `client-layer1` hoặc `client-layer2`, và cập nhật cấu hình `server.clients` trong `config.yaml` nếu cần để server biết số lượng client dự kiến). **Quan trọng: giữ nguyên cấu hình tài nguyên (CPU/Memory) cho mỗi pod worker khi thay đổi số lượng worker.**
* **Ý nghĩa tham số & Cách chứng minh hiệu quả thiết kế:**
    * Nếu việc tăng số lượng worker (ví dụ, tăng số replica cho Layer 2 từ 1 lên 2) dẫn đến sự gia tăng FPS tổng thể của hệ thống (dù có thể không phải là tăng tuyến tính hoàn hảo do overhead của việc điều phối và giao tiếp), điều đó chứng tỏ kiến trúc phân tán của bạn có khả năng tận dụng tài nguyên bổ sung để tăng hiệu suất.
    * **Lập luận:** Thiết kế phân tán cho phép chúng tôi song song hóa việc xử lý ở các công đoạn hoặc trong cùng một công đoạn (nếu một layer có nhiều replica). Bằng cách thêm worker, chúng tôi có thể tăng khả năng xử lý của một công đoạn cụ thể hoặc toàn bộ pipeline, từ đó cải thiện thông lượng chung.
* **Điểm tham chiếu (So sánh với chính nó ở các cấu hình phân tán khác nhau):**
    * So sánh FPS khi chạy với 1 worker/stage vs. N workers/stage.
    * Lưu ý đến định luật Amdahl: tốc độ tăng sẽ có giới hạn do phần công việc không thể song song hóa được (ví dụ: một số phần xử lý tuần tự, overhead giao tiếp). Mục tiêu là cho thấy có sự cải thiện tích cực.
    * **Ví dụ diễn giải:** "Khi tăng số lượng worker song song cho Layer X từ 1 lên 2 replica (mỗi replica được cấp cùng một lượng tài nguyên), FPS tổng thể của pipeline đã tăng từ A FPS lên B FPS (tăng C%). Điều này cho thấy thiết kế phân tán có khả năng mở rộng hiệu quả và tận dụng tốt tài nguyên được bổ sung để cải thiện thông lượng."

### 3.2. Khả Năng Xác Định Điểm Nghẽn (Bottleneck) trong Pipeline Phân Tán
* **Cách đo:**
    * Kết hợp theo dõi FPS của từng layer (qua `FPSLogger` của mỗi client) với việc giám sát độ sâu của các hàng đợi RabbitMQ (`intermediate_queue_X`) giữa các layer đó.
* **Ý nghĩa tham số & Cách chứng minh hiệu quả thiết kế:**
    * Khả năng của hệ thống cho phép bạn chỉ ra được: "Layer X đang là điểm nghẽn bởi vì `FPSLogger` của nó báo cáo tốc độ xử lý thấp hơn đáng kể so với layer trước đó, đồng thời hàng đợi RabbitMQ đầu vào của Layer X (`intermediate_queue_X`) đang có xu hướng đầy lên hoặc có độ sâu lớn."
    * **Lập luận:** Một hệ thống phân tán được thiết kế tốt phải cho phép khả năng quan sát (observability) để xác định các vấn đề về hiệu năng. Việc bạn có thể dễ dàng xác định điểm nghẽn thông qua các số liệu này chính là một minh chứng cho tính hiệu quả của việc module hóa và sử dụng message queue.
* **Điểm tham chiếu (Không có tham chiếu cụ thể, mà là khả năng của hệ thống):**
    * Đây không phải là so sánh với một giá trị X, mà là chứng minh rằng hệ thống cung cấp đủ thông tin để chẩn đoán.
    * **Ví dụ diễn giải:** "Dữ liệu giám sát từ `FPSLogger` và RabbitMQ Management UI cho thấy Client Layer A xử lý ở tốc độ 50 FPS, trong khi Client Layer B chỉ đạt 30 FPS, và hàng đợi `intermediate_queue_B` (đầu vào của Layer B) có số lượng message tồn đọng cao. Dựa trên các số liệu này, chúng tôi có thể xác định Layer B hiện là điểm nghẽn của hệ thống, và cần được ưu tiên tối ưu hóa hoặc cấp thêm tài nguyên. Khả năng chẩn đoán này là một ưu điểm của kiến trúc module hóa và giao tiếp qua hàng đợi mà chúng tôi đã xây dựng."

---

**Lưu Ý Chung Khi Thực Hiện Đánh Giá:**

* **Môi trường thử nghiệm:** Tất cả các phép đo liên quan đến hiệu năng phân tán và độ trễ mạng nên được thực hiện trên môi trường Kubernetes thực tế mà bạn triển khai, không phải trên Docker Compose local (trừ khi chỉ đánh giá logic nội bộ của một client đơn lẻ).
* **Kiểm soát điều kiện:** Khi so sánh các kịch bản, cố gắng chỉ thay đổi một yếu tố tại một thời điểm (ví dụ: số lượng replica của một layer, cấu hình tài nguyên cho một loại pod) và giữ các yếu tố khác (dữ liệu đầu vào, cấu hình RabbitMQ, cấu hình chung của hệ thống) không đổi.
* **Chạy thử nghiệm đủ lâu:** Để các chỉ số, đặc biệt là các giá trị trung bình, có thời gian hội tụ và ổn định.
* **Trực quan hóa dữ liệu:** Sử dụng biểu đồ, đồ thị để trình bày sự thay đổi của các tham số theo thời gian, theo tải, hoặc theo sự thay đổi cấu hình. Điều này sẽ giúp báo cáo của bạn dễ hiểu và thuyết phục hơn.
* **Giải thích kết quả:** Luôn giải thích tại sao một giá trị hoặc một xu hướng lại chứng tỏ quyết định thiết kế của bạn là hiệu quả, dựa trên mục tiêu thiết kế ban đầu và các điểm tham chiếu/so sánh đã xác định.