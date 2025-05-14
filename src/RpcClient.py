# src/RpcClient.py (Cập nhật để đọc cấu trúc START phẳng từ server)
import pika
import uuid
import pickle
import time
import base64
import os

class RpcClient:
    def __init__(self, client_id, layer_id, address, username, password, virtual_host,
                 logger_ref):
        self.client_id = client_id
        self.layer_id = layer_id
        self.logger_ref = logger_ref
        
        self.initial_params = None
        self.response_payload = None # Sẽ lưu toàn bộ dict response từ server
        self.correlation_id = None

        self.credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=address,
                port=5672,
                virtual_host=virtual_host,
                credentials=self.credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
        )
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue_name = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue_name,
            on_message_callback=self.on_response_message,
            auto_ack=True
        )
        self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Initialized. Waiting on queue '{self.callback_queue_name}'.")

    def on_response_message(self, ch, method, props, body):
        if self.correlation_id == props.correlation_id:
            try:
                self.response_payload = pickle.loads(body) # response_payload là dict được server gửi
                self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Received RPC response. CorrID: {props.correlation_id}")

                if not isinstance(self.response_payload, dict):
                    self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] Error: Response is not a dictionary! Got type: {type(self.response_payload)}")
                    self.initial_params = {"error": f"Response was not a dict: {type(self.response_payload)}"}
                    return

                action = self.response_payload.get("action")
                server_message = self.response_payload.get("message") # Lấy message từ server (nếu có)
                if server_message:
                     self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Message from server: {server_message}")
                
                self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Action from server: {action}")

                if action == "START":
                    # Server gửi các tham số ở dạng phẳng, trực tiếp trong self.response_payload
                    self.initial_params = {
                        "model_name": self.response_payload.get("model_name"),
                        "num_layers": self.response_payload.get("num_layers"),
                        "splits": self.response_payload.get("splits"),
                        "save_layers": self.response_payload.get("save_layers"),
                        "batch_frame": self.response_payload.get("batch_frame"),
                        "encoded_model": self.response_payload.get("model"),    # Server dùng "model"
                        "data_source": self.response_payload.get("data"),      # Server dùng "data"
                        "debug_mode": self.response_payload.get("debug_mode", False),
                        "imgsz": self.response_payload.get("imgsz")             # Server dùng "imgsz"
                    }
                    
                    # Kiểm tra và xử lý các tham số cần thiết
                    if self.initial_params.get("model_name") is None:
                        self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] 'model_name' is missing in START parameters.")
                        self.initial_params["error"] = "'model_name' missing in START parameters"
                    else:
                         self.initial_params["model_save_path"] = f'{self.initial_params["model_name"]}.pt'

                    # Lưu model nếu có
                    encoded_model_str = self.initial_params.get("encoded_model")
                    model_save_file_path = self.initial_params.get("model_save_path")

                    if encoded_model_str and model_save_file_path:
                        if os.path.exists(model_save_file_path):
                            self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Model file {model_save_file_path} already exists. Skipping download.")
                        else:
                            try:
                                model_bytes = base64.b64decode(encoded_model_str)
                                with open(model_save_file_path, "wb") as f:
                                    f.write(model_bytes)
                                self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Model successfully saved to {model_save_file_path}")
                            except Exception as e:
                                self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] Error decoding/saving model: {e}")
                                self.initial_params["error"] = f"Model saving error: {e}"
                    elif not encoded_model_str:
                        self.logger_ref.log_warning(f"[RpcClient L{self.layer_id}] No 'encoded_model' string found in START parameters.")
                    elif not model_save_file_path: # Thường xảy ra nếu model_name thiếu
                         self.logger_ref.log_warning(f"[RpcClient L{self.layer_id}] 'model_save_path' could not be determined.")

                elif action == "REGISTERED":
                    self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Successfully registered with server.")
                    # self.initial_params vẫn là None, wait_response sẽ tiếp tục chờ START
                
                elif action == "ERROR":
                    error_msg = server_message if server_message else self.response_payload.get('details', 'Unknown error from server')
                    self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] Received ERROR from server: {error_msg}")
                    self.initial_params = {"error": error_msg}
                
                else:
                    self.logger_ref.log_warning(f"[RpcClient L{self.layer_id}] Received unhandled action: {action}")

            except pickle.UnpicklingError as e:
                self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] Failed to unpickle response body: {e}. Body (first 100 bytes): {body[:100]}")
                self.initial_params = {"error": f"Unpickling error: {e}"}
            except KeyError as e: # Bắt lỗi nếu một key bắt buộc (ví dụ "action") không có trong self.response_payload
                self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] Missing key in response_payload: {e}. Payload: {self.response_payload}")
                self.initial_params = {"error": f"Missing key in server response: {e}"}
            except Exception as e:
                self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] Exception in on_response_message: {e}")
                self.initial_params = {"error": f"General RpcClient error: {e}"}
        else:
            self.logger_ref.log_warning(f"[RpcClient L{self.layer_id}] Received message with mismatched correlation ID. Expected: {self.correlation_id}, Got: {props.correlation_id}. Body (first 100 bytes): {body[:100]}")

    def send_to_server(self, message_dict):
        self.response_payload = None
        self.initial_params = None
        self.correlation_id = str(uuid.uuid4())
        
        self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Sending message to 'rpc_queue_server'. CorrID: {self.correlation_id}, Action: {message_dict.get('action')}")

        try:
            self.channel.basic_publish(
                exchange='',
                routing_key='rpc_queue',
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue_name,
                    correlation_id=self.correlation_id,
                ),
                body=pickle.dumps(message_dict)
            )
            self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Message for action '{message_dict.get('action')}' published.")
        except Exception as e:
            self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] Failed to publish message: {e}")
            # Cân nhắc thêm cơ chế retry hoặc báo lỗi nghiêm trọng hơn

    def wait_response(self, timeout_seconds=300):
        self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Waiting for server response (timeout: {timeout_seconds}s)...")
        start_time = time.time()

        while True: # Vòng lặp sẽ được ngắt bởi return hoặc timeout
            # Xử lý các sự kiện Pika
            self.connection.process_data_events(time_limit=1)

            # Kiểm tra điều kiện thoát dựa trên self.initial_params (được set trong on_response_message)
            if self.initial_params:
                if "error" in self.initial_params:
                    self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] Exiting wait_response due to error: {self.initial_params['error']}")
                    return False # Có lỗi
                if self.initial_params.get("model_name") is not None: # Điều kiện thành công chính
                    self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] START signal processed. Parameters received.")
                    return True # Thành công
            
            # Kiểm tra xem có nhận được "REGISTERED" và cần reset để chờ "START" không
            # self.response_payload sẽ là dict đầy đủ từ server
            if self.response_payload and self.response_payload.get("action") == "REGISTERED":
                if self.initial_params is None: # Chỉ reset nếu chưa nhận START/ERROR
                    self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Received REGISTERED, resetting response to wait for START.")
                    self.response_payload = None # Reset để vòng lặp tiếp tục chờ message START thật sự
                # Nếu initial_params đã có (ví dụ lỗi từ trước), không reset nữa

            # Kiểm tra timeout
            if time.time() - start_time > timeout_seconds:
                self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] Timeout waiting for server response after {timeout_seconds}s.")
                # Thử kiểm tra self.initial_params một lần cuối trước khi báo timeout
                if self.initial_params and self.initial_params.get("model_name"): return True
                if self.initial_params and self.initial_params.get("error"): return False
                return False # Timeout

        # Code không nên tới đây nếu logic vòng lặp while True là đúng
        return False


    def close(self):
        try:
            if self.connection and self.connection.is_open:
                self.logger_ref.log_info(f"[RpcClient L{self.layer_id}] Closing RabbitMQ connection.")
                self.connection.close()
        except Exception as e:
            self.logger_ref.log_error(f"[RpcClient L{self.layer_id}] Error closing RabbitMQ connection: {e}")