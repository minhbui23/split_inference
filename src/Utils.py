
import pika
from requests.auth import HTTPBasicAuth
import requests
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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