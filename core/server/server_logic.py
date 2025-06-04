import os
import pickle
import base64
import time
import pika

from core.utils.logger import Logger

class Server:
    def __init__(self, config):
        """Initialize the Server with configuration settings.

        Args:
            config (dict): Configuration dictionary containing RabbitMQ, model, and server settings.
        """
        self.config = config
        self.logger = Logger(f"{config['log-path']}/server_app.log")
        self._initialize_rabbitmq()
        self._initialize_server_settings()
        self._initialize_client_tracking()
        self.encoded_model_payload = self._load_and_encode_model()
        self.should_stop_server = False

    def _initialize_rabbitmq(self):
        """Set up RabbitMQ connection and channel."""
        rabbit_config = self.config["rabbit"]
        credentials = pika.PlainCredentials(
            username=rabbit_config["username"],
            password=rabbit_config["password"]
        )
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=rabbit_config["address"],
                port=5672,
                virtual_host=rabbit_config["virtual-host"],
                credentials=credentials
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self._on_request)

    def _initialize_server_settings(self):
        """Initialize server-specific settings from config."""
        server_config = self.config["server"]
        self.model_name = server_config["model"]
        self.cut_layer_config = server_config["cut-layer"]
        self.expected_clients_per_layer = server_config["clients"]
        self.total_expected_clients = sum(self.expected_clients_per_layer)
        self.num_layers = len(self.expected_clients_per_layer)
        self.data_source = self.config["data"]
        self.debug_mode = self.config["debug-mode"]

    def _initialize_client_tracking(self):
        """Initialize structures to track client registrations and states."""
        self.client_info = {}  # client_id -> {'layer_id', 'reply_to', 'correlation_id'}
        self.registered_clients_count = 0
        self.started_clients = set()
        self.initial_start_sent = False
        self.logger.log_info(
            f"Server initialized, waiting for {self.total_expected_clients} clients."
        )

    def _load_and_encode_model(self):
        """Load and encode the model file as base64.

        Returns:
            str: Base64-encoded model string, or None if loading fails.
        """
        model_file_path = f"{self.model_name}.pt"
        if not os.path.exists(model_file_path):
            self.logger.log_error(f"Model file {model_file_path} does not exist.")
            return None
        try:
            with open(model_file_path, "rb") as f:
                file_bytes = f.read()
            encoded_weights = base64.b64encode(file_bytes).decode('utf-8')
            self.logger.log_info(f"Model '{self.model_name}' loaded and encoded.")
            return encoded_weights
        except Exception as e:
            self.logger.log_error(f"Error encoding model file {model_file_path}: {e}")
            return None

    def _on_request(self, ch, method, props, body):
        """Handle incoming RPC requests from clients.

        Args:
            ch: Pika channel.
            method: Pika method frame.
            props: Pika properties.
            body: Message body.
        """
        try:
            message = pickle.loads(body)
            action = message.get("action")
            client_id = str(message.get("client_id"))
            layer_id = message.get("layer_id")
            self.logger.log_info(
                f"Received action '{action}' from client {client_id} (Layer {layer_id})"
            )

            if action == "REGISTER":
                self._handle_register(client_id, layer_id, props)
            else:
                self.logger.log_warning(f"Unknown action '{action}' from client {client_id}")

        except Exception as e:
            self.logger.log_error(f"Error processing request: {e}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _handle_register(self, client_id, layer_id, props):
        """Process REGISTER action from a client.

        Args:
            client_id (str): Unique client identifier.
            layer_id (int): Layer ID of the client.
            props: Pika properties containing reply_to and correlation_id.
        """
        if not props.reply_to:
            self.logger.log_warning(
                f"Client {client_id} registered without reply_to queue."
            )
            return

        # Update or add client info
        if client_id not in self.client_info:
            self.client_info[client_id] = {
                'layer_id': layer_id,
                'reply_to': props.reply_to,
                'correlation_id': props.correlation_id
            }
            self.registered_clients_count += 1
            self.logger.log_info(
                f"Client {client_id} (Layer {layer_id}) registered. "
                f"Total: {self.registered_clients_count}/{self.total_expected_clients}"
            )
        else:
            self.client_info[client_id].update({
                'reply_to': props.reply_to,
                'correlation_id': props.correlation_id,
                'layer_id': layer_id
            })
            self.logger.log_warning(f"Client {client_id} re-registered.")

        # Notify clients when all are registered or handle late registrations
        if not self.initial_start_sent and self.registered_clients_count == self.total_expected_clients:
            self.logger.log_info("All clients registered. Notifying all.")
            self._notify_all_clients()
            self.initial_start_sent = True
        elif self.initial_start_sent and client_id not in self.started_clients:
            self.logger.log_info(f"Late client {client_id}. Sending START.")
            self._send_start_signal(client_id)

    def _send_rpc_reply(self, reply_to_queue, correlation_id, message_body):
        """Send an RPC reply to the client's reply_to queue.

        Args:
            reply_to_queue (str): Queue name to send the reply to.
            correlation_id (str): Correlation ID to match request/reply.
            message_body (bytes): Serialized message body.
        """
        try:
            self.channel.basic_publish(
                exchange='',
                routing_key=reply_to_queue,
                properties=pika.BasicProperties(correlation_id=correlation_id),
                body=message_body
            )
            self.logger.log_info(f"Sent RPC reply to '{reply_to_queue}' (CorrID: {correlation_id})")
        except Exception as e:
            self.logger.log_error(f"Error sending RPC reply to '{reply_to_queue}': {e}")

    def _send_start_signal(self, client_id):
        """Send START signal to a specific client.

        Args:
            client_id (str): ID of the client to notify.
        """
        if client_id in self.started_clients:
            self.logger.log_info(f"Client {client_id} already started. Skipping.")
            return

        info = self.client_info.get(client_id)
        if not info:
            self.logger.log_warning(f"Unregistered client {client_id}. Skipping.")
            return

        # Default split configurations for model
        default_splits = {
            "a": (10, [4, 6, 9]),
            "b": (16, [9, 12, 15]),
            "c": (22, [15, 18, 21])
        }
        split_key = self.cut_layer_config if self.cut_layer_config in default_splits else "a"
        if split_key != self.cut_layer_config:
            self.logger.log_error(f"Invalid cut-layer '{self.cut_layer_config}'. Using 'a'.")

        split_params = default_splits[split_key]
        client_config = self.config["client"]

        # Prepare response payload
        response_payload = {
            "action": "START",
            "message": "Server accepts the connection. Ready to start inference.",
            "model": self.encoded_model_payload,
            "splits": split_params[0],
            "save_layers": split_params[1],
            "num_layers": self.num_layers,
            "model_name": self.model_name
        }

        self._send_rpc_reply(
            info['reply_to'],
            info['correlation_id'],
            pickle.dumps(response_payload)
        )
        self.started_clients.add(client_id)
        self.logger.log_info(f"Sent START signal to client {client_id}")

    def _notify_all_clients(self):
        """Notify all registered clients to start."""
        for client_id in self.client_info:
            self._send_start_signal(client_id)

    def start(self):
        """Start the server's RPC consumer to listen for client registrations."""
        self.logger.log_info("Starting server's RPC consumer...")

        run_duration = self.config.get("app", {}).get("run_duration_seconds", 0)
        start_time = time.time()

        try:
            # Thay vì channel.start_consuming(), dùng vòng lặp với process_data_events
            # để có thể kiểm tra điều kiện dừng.
            while not self.should_stop_server:
                # Xử lý sự kiện mạng trong một khoảng thời gian ngắn
                # Timeout này quan trọng để vòng lặp không bị block quá lâu
                # và có thể kiểm tra điều kiện dừng thường xuyên.
                try:
                    if self.connection and self.connection.is_open:
                        self.connection.process_data_events(time_limit=1.0) # Xử lý sự kiện trong 1 giây
                    else:
                        self.logger.log_warning("Connection not open in main server loop. Attempting to re-establish or stop.")
                        self.should_stop_server = True # Dừng nếu mất kết nối
                        break
                except pika.exceptions.AMQPConnectionError as e:
                    self.logger.log_error(f"AMQP Connection Error in server loop: {e}. Stopping server.", exc_info=True)
                    self.should_stop_server = True
                    break
                except Exception as e_proc:
                    self.logger.log_error(f"Error in process_data_events: {e_proc}. Stopping server.", exc_info=True)
                    self.should_stop_server = True
                    break

                # Kiểm tra điều kiện dừng theo thời gian
                if run_duration > 0 and (time.time() - start_time) >= run_duration:
                    self.logger.log_info(f"Run duration of {run_duration} seconds reached.")
                    self._notify_all_clients_to_stop() # Hàm này sẽ set self.should_stop_server = True
                    # Vòng lặp sẽ thoát ở lần kiểm tra self.should_stop_server tiếp theo
                
                # Có thể thêm một sleep nhỏ ở đây nếu process_data_events trả về ngay lập tức
                # và không có sự kiện nào, để tránh CPU busy-loop.
                # Tuy nhiên, time_limit=1.0 đã làm việc đó rồi.

        except KeyboardInterrupt:
            self.logger.log_info("Server_logic: KeyboardInterrupt received. Initiating shutdown.")
            self._notify_all_clients_to_stop() # Thông báo cho client dừng
        except Exception as e:
            self.logger.log_error(f"Server_logic: Unhandled exception in start(): {e}", exc_info=True)
            self._notify_all_clients_to_stop() # Cố gắng thông báo cho client dừng
        finally:
            self.logger.log_info("Server_logic: Main loop ended. Cleaning up...")
            if self.connection and self.connection.is_open:
                try:
                    self.connection.close()
                    self.logger.log_info("Server_logic: RabbitMQ connection closed.")
                except Exception as e_close:
                    self.logger.log_error(f"Server_logic: Error closing RabbitMQ connection: {e_close}", exc_info=True)
            self.logger.log_info("Server_logic: Shutdown complete.")


    # Stop signal handling
    def _send_stop_signal_to_client(self, client_id, client_info):
        """Sends a STOP signal to a specific client's callback queue."""
        reply_to_queue = client_info.get('reply_to')
        if not reply_to_queue:
            self.logger.log_warning(f"Client {client_id} has no reply_to queue. Cannot send STOP.")
            return

        stop_payload = {"action": "STOP", "reason": "Server-initiated shutdown (timer or explicit command)."}
        stop_body = pickle.dumps(stop_payload)
        try:
            self.channel.basic_publish(
                exchange='', # Direct to queue
                routing_key=reply_to_queue,
                # No correlation_id needed for server-initiated commands unless a specific protocol is designed
                properties=pika.BasicProperties(), 
                body=stop_body
            )
            self.logger.log_info(f"Sent STOP signal to client {client_id} on queue {reply_to_queue}")
        except Exception as e:
            self.logger.log_error(f"Failed to send STOP to client {client_id} on queue {reply_to_queue}: {e}")

    def _notify_all_clients_to_stop(self): # Hàm mới (hoặc có thể đổi tên _broadcast_stop_signal)
        """Iterates through all clients and sends them a STOP signal."""
        self.logger.log_info("Initiating STOP signal for all registered clients...")
        clients_to_notify = list(self.client_info.items()) # Avoid issues if dict changes during iteration
        for client_id, info in clients_to_notify:
            self._send_stop_signal_to_client(client_id, info)
        
        self.should_stop_server = True
        self.logger.log_info("All clients notified to STOP. Server will shut down soon.")
