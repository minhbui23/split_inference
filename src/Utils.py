
import time
import pika
from requests.auth import HTTPBasicAuth
import requests
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FPSLogger:
    def __init__(self, layer_id: int, logger_obj, log_interval_seconds: int = 10, log_prefix: str = "Batch"):

        self.layer_id = layer_id
        self.logger = logger_obj
        self.log_interval_seconds = log_interval_seconds
        self.log_prefix = log_prefix

        self.total_frames_processed = 0
        self.total_inference_time = 0.0  # Tính bằng giây
        self.last_cumulative_log_time = time.time()
        
        self._current_batch_start_time = None # Thời điểm bắt đầu xử lý batch hiện tại

    def start_batch_timing(self):
        """Ghi nhận thời điểm bắt đầu xử lý một batch."""
        self._current_batch_start_time = time.time()

    def end_batch_and_log_fps(self, frames_in_current_batch: int):
        """
        Ghi nhận thời điểm kết thúc xử lý batch, tính FPS cho batch,
        cập nhật tổng tích lũy, và log FPS cho batch và có thể cả FPS tích lũy.
        Args:
            frames_in_current_batch: Số lượng frame đã xử lý trong batch này.
        """
        if self._current_batch_start_time is None:
            if self.logger: # Kiểm tra logger tồn tại trước khi sử dụng
                self.logger.log_warning(f"[FPSLogger L{self.layer_id}] end_batch_and_log_fps() được gọi mà không có start_batch_timing(). Bỏ qua log FPS cho batch này.")
            return

        batch_processing_time = time.time() - self._current_batch_start_time
        self._current_batch_start_time = None  # Reset cho lần gọi tiếp theo

        self.total_frames_processed += frames_in_current_batch
        self.total_inference_time += batch_processing_time

        # Log FPS cho batch hiện tại
        if self.logger: # Kiểm tra logger tồn tại
            if batch_processing_time > 0:
                batch_fps = frames_in_current_batch / batch_processing_time
                self.logger.log_info(
                    f"[InferenceThread L{self.layer_id}] {self.log_prefix}: {frames_in_current_batch} frames processed in {batch_processing_time:.4f}s (FPS: {batch_fps:.2f})."
                )
            else:
                self.logger.log_info(
                    f"[InferenceThread L{self.layer_id}] {self.log_prefix}: {frames_in_current_batch} frames processed very quickly."
                )

        # Log FPS tích lũy định kỳ
        current_time = time.time()
        if self.total_frames_processed > 0 and \
           (current_time - self.last_cumulative_log_time >= self.log_interval_seconds) and \
           self.logger: # Kiểm tra logger tồn tại
            if self.total_inference_time > 0:
                cumulative_avg_fps = self.total_frames_processed / self.total_inference_time
                self.logger.log_info(
                    f"[InferenceThread L{self.layer_id}] CUMULATIVE: {self.total_frames_processed} frames in {self.total_inference_time:.4f}s (Avg FPS: {cumulative_avg_fps:.2f})."
                )
            self.last_cumulative_log_time = current_time

    def log_overall_fps(self, process_description: str = "Processing finished"):
        """Logs FPS trung bình tổng thể cuối cùng."""
        if not self.logger: # Kiểm tra logger tồn tại
            print(f"[FPSLogger L{self.layer_id}] Logger not available for final FPS log.")
            return

        if self.total_frames_processed == 0:
            self.logger.log_info(f"[InferenceThread L{self.layer_id}] {process_description}. No frames were processed or timed.")
            return
            
        if self.total_inference_time > 0:
            overall_avg_fps = self.total_frames_processed / self.total_inference_time
            self.logger.log_info(
                f"[InferenceThread L{self.layer_id}] {process_description}. OVERALL: {self.total_frames_processed} frames in {self.total_inference_time:.4f}s (Avg FPS: {overall_avg_fps:.2f})."
            )
        else:
            self.logger.log_info(
                f"[InferenceThread L{self.layer_id}] {process_description}. {self.total_frames_processed} frames processed, but total inference time was zero or not recorded."
            )



def delete_old_queues(address, username, password, virtual_host):
    # Đảm bảo virtual_host được mã hóa đúng chuẩn URL (quan trọng nếu vhost không phải '/')
    encoded_vhost = requests.utils.quote(virtual_host, safe='')
    mgmt_url = f'http://{address}:15672/api/queues/{encoded_vhost}'
    logging.info(f"Cleanup: Lấy danh sách queue từ API: {mgmt_url}")

    try:
        # Gọi API Management để lấy danh sách queues
        response = requests.get(mgmt_url, auth=HTTPBasicAuth(username, password), timeout=10) # Đặt timeout 10 giây
        response.raise_for_status() # Kiểm tra lỗi HTTP (4xx, 5xx)
        queues = response.json()
        logging.info(f"Cleanup: Lấy thành công {len(queues)} queue(s) qua API.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Cleanup: Lỗi khi kết nối hoặc lấy danh sách queue từ Management API: {e}")
        return False # Không thể tiếp tục nếu không lấy được danh sách

    connection = None # Khởi tạo connection là None
    deleted_count = 0
    try:
        # Thiết lập kết nối AMQP để thực hiện xóa
        credentials = pika.PlainCredentials(username, password)
        params = pika.ConnectionParameters(
            host=address,
            port=5672,
            virtual_host=virtual_host,
            credentials=credentials,
            heartbeat=30,
            blocked_connection_timeout=30
        )
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        logging.info("Cleanup: Đã kết nối AMQP để dọn dẹp.")

        # Các tiền tố của queue do ứng dụng này tạo ra cần được dọn dẹp
        # Bao gồm cả 'reply_' theo logic gốc
        app_queue_prefixes = ("reply_", "intermediate_queue_", "result", "rpc_queue")

        for queue_info in queues:
            queue_name = queue_info.get('name')
            if not queue_name:
                continue

            # Kiểm tra xem tên queue có bắt đầu bằng một trong các tiền tố cần xóa không
            if queue_name.startswith(app_queue_prefixes):
                logging.info(f"Cleanup: Chuẩn bị xóa queue ứng dụng: {queue_name}")
                try:
                    # Thực hiện xóa queue qua kênh AMQP
                    channel.queue_delete(queue=queue_name)
                    deleted_count += 1
                    logging.info(f"Cleanup: Đã xóa thành công queue: {queue_name}")
                except pika.exceptions.ChannelClosedByBroker as e:
                    # Lỗi này CÓ THỂ xảy ra nếu cố xóa queue 'reply_' mà client đã tạo là exclusive
                    logging.warning(f"Cleanup: Không thể xóa queue {queue_name}. Broker đã đóng channel: {e}. Queue có thể đang được dùng hoặc là exclusive.")
                    # Cố gắng mở lại channel để tiếp tục với các queue khác
                    if connection.is_open and (not channel or channel.is_closed):
                        try:
                            channel = connection.channel()
                            logging.info("Cleanup: Đã mở lại channel sau lỗi xóa.")
                        except Exception as reopen_e:
                            logging.error(f"Cleanup: Không thể mở lại channel: {reopen_e}. Dừng dọn dẹp.")
                            break # Thoát vòng lặp nếu không mở lại được channel
                except Exception as e:
                    logging.error(f"Cleanup: Lỗi không xác định khi xóa queue {queue_name}: {e}")
                    # Cũng nên thử mở lại channel
                    if connection.is_open and (not channel or channel.is_closed):
                        try:
                           channel = connection.channel()
                           logging.info("Cleanup: Đã mở lại channel sau lỗi không xác định.")
                        except Exception as reopen_e:
                           logging.error(f"Cleanup: Không thể mở lại channel: {reopen_e}. Dừng dọn dẹp.")
                           break
            # else:
                # Bỏ qua các queue không khớp tiền tố
                # logging.debug(f"Cleanup: Bỏ qua queue không thuộc ứng dụng: {queue_name}")

        logging.info(f"Cleanup: Kết thúc dọn dẹp. Số queue đã xóa: {deleted_count}")
        return True

    except pika.exceptions.AMQPConnectionError as e:
        logging.error(f"Cleanup: Lỗi kết nối AMQP để dọn dẹp: {e}")
        return False
    except Exception as e:
        logging.error(f"Cleanup: Lỗi không xác định trong quá trình dọn dẹp AMQP: {e}")
        import traceback
        traceback.print_exc() # In chi tiết lỗi nếu có lỗi lạ
        return False
    finally:
        # Luôn đảm bảo đóng kết nối AMQP sau khi xong
        if connection and connection.is_open:
            connection.close()
            logging.info("Cleanup: Đã đóng kết nối AMQP dọn dẹp.")

