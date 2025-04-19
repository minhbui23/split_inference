import pickle
import time
import base64
import threading 

import pika
import torch # Giả sử vẫn cần cho type hinting hoặc lỗi import nếu bỏ
import torch.nn as nn # Không cần thiết trực tiếp ở đây

import src.Log
from src.Model import SplitDetectionModel
from ultralytics import YOLO

class RpcClient:
    def __init__(self, client_id, layer_id, address, username, password, virtual_host, inference_func, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.virtual_host = virtual_host

        self.inference_func_ref = inference_func 
        self.device = device

        self.channel = None
        self.connection = None
        self.response_data = None # Dùng để lưu dữ liệu response
        self.model = None
        self.reply_queue_name = f"reply_{self.client_id}"
        self._reply_received_event = threading.Event() 
        self._consumer_tag = None 

        self.connect() 

    def _setup_reply_consumer(self):
        """Khai báo queue và đăng ký consumer cho reply."""
        try:
            # Khai báo queue reply
            self.channel.queue_declare(self.reply_queue_name, durable=False, exclusive=True) # exclusive=True để queue tự xóa khi client mất kết nối
             # Đăng ký consumer, chỉ định hàm callback _handle_rpc_reply
             # auto_ack=False để xác nhận thủ công sau khi xử lý
            self._consumer_tag = self.channel.basic_consume(
                queue=self.reply_queue_name,
                on_message_callback=self._handle_rpc_reply,
                auto_ack=False
            )
            src.Log.print_with_color(f"Client {self.client_id} waiting for reply on {self.reply_queue_name}", "yellow")
        except Exception as e:
            src.Log.print_with_color(f"Error setting up reply consumer: {e}", "red")
            self._reply_received_event.set() # Set event để không bị block vô hạn nếu lỗi

    def _handle_rpc_reply(self, ch, method, properties, body):
        """Hàm callback xử lý message START từ server."""
        src.Log.print_with_color(f"RPC Reply received!", "blue")
        try:
            # Xử lý message, load model
            if self._process_start_message(body):
                 # Chỉ ack nếu xử lý START thành công và load model xong
                ch.basic_ack(delivery_tag=method.delivery_tag)
                src.Log.print_with_color(f"START message processed successfully.", "green")
            else:
                # Nếu xử lý START thất bại, không ack (hoặc có thể nack)
                # Message có thể bị consume lại hoặc vào dead-letter nếu cấu hình
                 src.Log.print_with_color(f"Failed to process START message.", "red")

        except Exception as e:
            src.Log.print_with_color(f"Error processing RPC reply: {e}", "red")

        finally:
            self._reply_received_event.set()
            if self._consumer_tag:
                 try:
                      ch.basic_cancel(consumer_tag=self._consumer_tag, callback=self._on_cancelok)
                 except Exception as cancel_e:
                      src.Log.print_with_color(f"Error cancelling consumer {self._consumer_tag}: {cancel_e}", "yellow")

    def _on_cancelok(self, frame):
         """Callback khi basic_cancel thành công."""
         src.Log.print_with_color(f"RPC reply consumer {self._consumer_tag} cancelled.", "yellow")

    def _process_start_message(self, body):
        """Xử lý nội dung message START và load model. Trả về True nếu thành công."""
        try:
            self.response_data = pickle.loads(body) 
            src.Log.print_with_color(f"[<<<] Client received: {self.response_data['message']}", "blue")
            action = self.response_data["action"]

            if action == "START":
                model_name = self.response_data["model_name"]
                splits = self.response_data["splits"]
                model_data_encoded = self.response_data["model"]

                if model_data_encoded is not None:
                    decoder = base64.b64decode(model_data_encoded)
                    with open(f"{model_name}.pt", "wb") as f:
                        f.write(decoder)
                    src.Log.print_with_color(f"Loaded {model_name}.pt", "green")
                else:
                    src.Log.print_with_color(f"Do not load model file.", "yellow")

                # Load model gốc và tạo model đã chia
                pretrain_model = YOLO(f"{model_name}.pt").model
                self.model = SplitDetectionModel(pretrain_model, split_layer=splits)
                src.Log.print_with_color(f"Split model created for layer {self.layer_id}.", "green")
                return True # Xử lý thành công
            else:
                src.Log.print_with_color(f"Received non-START action: {action}", "yellow")
                return False # Action không mong muốn
        except Exception as e:
            src.Log.print_with_color(f"Error in _process_start_message: {e}", "red")
            return False # Xử lý lỗi

    def wait_for_start_command(self, timeout=300):
        """Chờ nhận message START từ server bằng basic_consume."""
        if not self.channel or self.channel.is_closed:
             src.Log.print_with_color("Cannot wait for reply, channel is closed.", "red")
             return None, None # Trả về None nếu channel đóng

        self._reply_received_event.clear() # Reset event trước khi chờ
        self._setup_reply_consumer() # Thiết lập consumer

        start_time = time.time()
        while not self._reply_received_event.is_set():
             if timeout and time.time() - start_time > timeout:
                  src.Log.print_with_color("Timeout waiting for RPC reply.", "red")
                  # Hủy consumer nếu timeout
                  if self._consumer_tag:
                       try:
                            self.channel.basic_cancel(self._consumer_tag)
                       except Exception as cancel_e:
                            src.Log.print_with_color(f"Error cancelling consumer on timeout: {cancel_e}", "yellow")
                  return None, None # Trả về None nếu timeout


             self.connection.process_data_events(time_limit=0.1)

        return self.model, self.response_data

    def connect(self):
        """Thiết lập kết nối và channel."""
        try:
             if self.connection and self.connection.is_open:
                 return # Đã kết nối rồi thì thôi
             credentials = pika.PlainCredentials(self.username, self.password)
             self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, self.virtual_host, credentials, heartbeat=600, blocked_connection_timeout=300))
             self.channel = self.connection.channel()
             src.Log.print_with_color("Connection and channel established.", "green")
        except Exception as e:
             src.Log.print_with_color(f"Failed to connect: {e}", "red")
             self.connection = None
             self.channel = None


    def send_to_server(self, message):
        """Gửi message đăng ký đến server."""
        if not self.channel or self.channel.is_closed:
             # Nếu chưa connect hoặc channel đã đóng, thử connect lại
             src.Log.print_with_color("Attempting to reconnect before sending...", "yellow")
             self.connect()
             if not self.channel: # Nếu vẫn không connect được thì báo lỗi
                  src.Log.print_with_color("Cannot send message, channel unavailable.", "red")
                  return False # Trả về False nếu không gửi được

        try:
            # Khai báo queue đích ('rpc_queue') - nên khai báo ở server là chính
            # self.channel.queue_declare('rpc_queue', durable=False) # Có thể bỏ qua ở client

            # Gửi message
            self.channel.basic_publish(exchange='',
                                       routing_key='rpc_queue',
                                       body=pickle.dumps(message),
                                       properties=pika.BasicProperties(
                                            reply_to = self.reply_queue_name 
                                       ))
            src.Log.print_with_color("Registration message sent to server.", "red")
            return True
        except Exception as e:
            src.Log.print_with_color(f"Error sending message to server: {e}", "red")
            return False

    def close_connection(self):
         """Đóng kết nối."""
         try:
              if self.connection and self.connection.is_open:
                   src.Log.print_with_color("Closing connection.", "yellow")
                   self.connection.close()
         except Exception as e:
              src.Log.print_with_color(f"Error closing connection: {e}", "red")
