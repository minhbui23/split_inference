import os
import sys
import base64
import pika
import pickle
import torch # Mặc dù không dùng trực tiếp, giữ lại cho ngữ cảnh
import torch.nn as nn # Mặc dù không dùng trực tiếp, giữ lại cho ngữ cảnh

import src.Model # Cần thiết nếu có type hinting hoặc kiểm tra
import src.Log


class Server:
    def __init__(self, config):
        # RabbitMQ
        self.config = config
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.cut_layer_config = config["server"]["cut-layer"] # Đổi tên để tránh nhầm lẫn với biến cục bộ
        self.expected_clients_per_layer = config["server"]["clients"] # Đây là list số client cho mỗi "nhóm" layer
        # Tổng số client mong đợi là tổng của các giá trị trong list này.
        # Ví dụ: nếu clients: [1, 1] nghĩa là có 2 client, mỗi client xử lý 1 phần.
        self.total_expected_clients = sum(self.expected_clients_per_layer)


        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue') # Hàng đợi server lắng nghe yêu cầu

        # Lưu thông tin client: client_id -> {'layer_id': ..., 'reply_to': ..., 'correlation_id': ...}
        self.client_info = {} 
        self.registered_clients_count = 0

        self.channel.basic_qos(prefetch_count=1)
        # self.reply_channel = self.connection.channel() # Channel này sẽ được dùng để gửi reply
        # Không cần reply_channel riêng nếu publish từ channel chính là an toàn.
        # Để đơn giản, dùng self.channel cho cả consume request và publish reply.

        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        self.data_source_config = config["data"] 
        self.imgsz = config["imgsz"] # Kích thước ảnh đầu vào cho model

        self.debug_mode = config["debug-mode"]
        log_path = config["log-path"]
        self.logger = src.Log.Logger(f"{log_path}/server_app.log") # Server cũng nên có logger riêng
        self.logger.log_info(f"Application start. Server is waiting for {self.total_expected_clients} clients.")

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        action = message.get("action")
        client_id = str(message.get("client_id")) # Đảm bảo client_id là string
        layer_id = message.get("layer_id") # Client gửi layer_id của nó

        self.logger.log_info(f"[<<<] Received message from client {client_id} (Layer {layer_id}): Action '{action}'")


        if action == "REGISTER":
            if not props.reply_to:
                self.logger.log_warning(f"Client {client_id} registered without a reply_to queue. Cannot send START signal.")
                # Ack message để loại bỏ nó khỏi queue
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            if client_id not in self.client_info:
                self.client_info[client_id] = {
                    'layer_id': layer_id,
                    'reply_to': props.reply_to, # Lưu lại reply_to queue
                    'correlation_id': props.correlation_id # Lưu lại correlation_id
                }
                self.registered_clients_count += 1
                self.logger.log_info(f"Client {client_id} (Layer {layer_id}) registered. Reply_to: {props.reply_to}. Total registered: {self.registered_clients_count}/{self.total_expected_clients}")
            else:
                # Cập nhật thông tin nếu client đăng ký lại (ví dụ reply_to có thể thay đổi nếu client khởi động lại)
                self.client_info[client_id]['reply_to'] = props.reply_to
                self.client_info[client_id]['correlation_id'] = props.correlation_id
                self.client_info[client_id]['layer_id'] = layer_id # Cập nhật layer_id nếu có thay đổi
                self.logger.log_warning(f"Client {client_id} (Layer {layer_id}) re-registered or sent duplicate registration.")


            # Kiểm tra xem đã đủ client đăng ký chưa
            # Logic này cần xem xét lại nếu `self.expected_clients_per_layer` phức tạp.
            # Hiện tại, chỉ cần tổng số client.
            if self.registered_clients_count >= self.total_expected_clients:
                # Đảm bảo không gửi notify nhiều lần nếu có re-registration
                # Có thể thêm một cờ self.notified = False ban đầu
                self.logger.log_info("All expected clients are connected. Sending notifications.")
                self.notify_clients()
            else:
                self.logger.log_info(f"Waiting for more clients. Currently {self.registered_clients_count}/{self.total_expected_clients}.")


        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_rpc_reply(self, reply_to_queue, correlation_id, message_body):
        """Gửi phản hồi RPC đến hàng đợi được chỉ định bởi client."""
        try:
            # Server không cần khai báo reply_to_queue, client đã làm việc đó.
            self.channel.basic_publish(
                exchange='', # Gửi đến default exchange
                routing_key=reply_to_queue, # routing_key là tên của reply_to_queue
                properties=pika.BasicProperties(
                    correlation_id=correlation_id # Gửi lại correlation_id để client khớp yêu cầu/phản hồi
                ),
                body=message_body
            )
            self.logger.log_info(f"[>>>] Sent RPC reply to queue '{reply_to_queue}' (CorrID: {correlation_id})")
        except Exception as e:
            self.logger.log_error(f"Error sending RPC reply to queue '{reply_to_queue}': {e}")


    def notify_clients(self):
        default_splits = {
            # Giá trị đầu tiên là split_layer index cho model, 
            # giá trị thứ hai là list các layer index cần save output (cho model.forward_head)
            "a": (10, [4, 6, 9]), 
            "b": (16, [9, 12, 15]),
            "c": (22, [15, 18, 21])
        }
        # Lấy cấu hình chia lớp từ config file, ví dụ 'a'
        split_config_key = self.cut_layer_config 
        if split_config_key not in default_splits:
            self.logger.log_error(f"Invalid cut-layer configuration: '{split_config_key}'. Using default 'a'.")
            split_config_key = "a" # Fallback to default
            
        current_split_params = default_splits[split_config_key]
        
        model_file_path = f"{self.model_name}.pt" # Model nên nằm ở thư mục server có thể truy cập
        encoded_model_weights = None
        if os.path.exists(model_file_path):
            self.logger.log_info(f"Loading model {self.model_name} from '{model_file_path}'.")
            with open(model_file_path, "rb") as f:
                file_bytes = f.read()
                encoded_model_weights = base64.b64encode(file_bytes).decode('utf-8')
        else:
            self.logger.log_error(f"Model file {model_file_path} does not exist. Cannot send model to clients.")
            # Không nên sys.exit() ở đây, có thể gửi lỗi cho client hoặc xử lý khác.
            # Hiện tại, sẽ gửi response không có model weights. Client cần xử lý điều này.

        batch_frame_from_config = self.config["server"]["batch-frame"] # Lấy từ config

        # Duyệt qua các client đã đăng ký và gửi thông báo START
        for client_id, info in self.client_info.items():
            client_layer_id = info['layer_id'] # Layer ID mà client này đã đăng ký
            reply_to_queue = info['reply_to']
            correlation_id = info['correlation_id']

            # Logic xác định `splits` và `save_layers` cho từng client có thể phức tạp hơn
            # nếu bạn muốn mỗi client nhận một phần khác nhau của model dựa trên layer_id của nó.
            # Hiện tại, tất cả client nhận cùng một `current_split_params[0]` (split_layer index)
            # và `current_split_params[1]` (save_layers_indices).
            # Điều này phù hợp nếu `SplitDetectionModel` được thiết kế để client tự biết
            # nó là head hay tail dựa trên `layer_id` và `num_layers` mà server gửi.

            response_payload = {
                "action": "START",
                "message": "Server accepts the connection. Ready to start inference.",
                "model": encoded_model_weights, # Có thể là None nếu file không tồn tại
                "splits": current_split_params[0], # split_layer index cho SplitDetectionModel
                "save_layers": current_split_params[1], # List các layer index cần save output
                "batch_frame": batch_frame_from_config,
                "num_layers": self.total_expected_clients, # Tổng số client tham gia                                       
                "model_name": self.model_name,
                "data": self.data_source_config, # Nguồn dữ liệu (ví dụ: video.mp4)
                "debug_mode": self.debug_mode,
                "imgsz": self.imgsz # Kích thước ảnh đầu vào cho model
            }
            
            self.logger.log_info(f"Preparing to send START to client {client_id} (Layer {client_layer_id}) on queue {reply_to_queue}")
            self.send_rpc_reply(reply_to_queue, correlation_id, pickle.dumps(response_payload))

        # Sau khi gửi thông báo, có thể reset để tránh gửi lại nếu có client đăng ký muộn/lại
        # self.client_info = {} 
        # self.registered_clients_count = 0
        # Hoặc dùng một cờ self.notifications_sent = True

    def start(self):
        self.logger.log_info("Server's RPC consumer starting. Waiting for client registrations...")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.log_info("Server's RPC consumer stopped by KeyboardInterrupt.")
        except Exception as e:
            self.logger.log_error(f"Server's RPC consumer stopped due to an error: {e}")
        finally:
            if self.connection.is_open:
                self.logger.log_info("Closing server Pika connection.")
                self.connection.close()
            self.logger.log_info("Server shutdown complete.")

