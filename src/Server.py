import os
import pickle
import base64
import pika
import src.Model
import src.Log

class Server:
    def __init__(self, config):
        """Initialize the Server with configuration settings.

        Args:
            config (dict): Configuration dictionary containing RabbitMQ, model, and server settings.
        """
        self.config = config
        self.logger = src.Log.Logger(f"{config['log-path']}/server_app.log")
        self._initialize_rabbitmq()
        self._initialize_server_settings()
        self._initialize_client_tracking()
        self.encoded_model_payload = self._load_and_encode_model()

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
            "batch_frame": self.config["server"]["batch-frame"],
            "num_layers": self.num_layers,
            "model_name": self.model_name,
            "data": self.data_source,
            "debug_mode": self.debug_mode,
            "imgsz": client_config.get("imgsz", (640, 640)),
            "internal_queue_size": client_config.get("internal_queue_size"),
            "io_prefetch_count": client_config.get("io_prefetch_count", 5),
            "rabbit_retry_delay": client_config.get("rabbit_retry_delay", 5),
            "io_process_events_timeout": client_config.get("io_process_events_timeout", 0.1),
            "io_output_q_timeout": client_config.get("io_output_q_timeout", 0.05),
            "ack_queue_process_delay": client_config.get("ack_queue_process_delay", 0.05)
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
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.log_info("Server stopped by KeyboardInterrupt.")
        except Exception as e:
            self.logger.log_error(f"Server stopped due to error: {e}")
        finally:
            if self.connection.is_open:
                self.connection.close()
            self.logger.log_info("Server shutdown complete.")