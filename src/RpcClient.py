import pika
import uuid
import pickle
import time
import base64
import os

class RpcClient:
    def __init__(self, client_id, layer_id, address, username, password, virtual_host, logger_ref):
        """Initialize the RPC client for communication with the server.

        Args:
            client_id (str): Unique identifier for the client.
            layer_id (int): Layer ID of the client.
            address (str): RabbitMQ server address.
            username (str): RabbitMQ username.
            password (str): RabbitMQ password.
            virtual_host (str): RabbitMQ virtual host.
            logger_ref: Logger instance for logging client activities.
        """
        self.client_id = client_id
        self.layer_id = layer_id
        self.logger = logger_ref
        self.initial_params = None
        self.response_payload = None
        self.correlation_id = None
        self._initialize_rabbitmq(address, username, password, virtual_host)

    def _initialize_rabbitmq(self, address, username, password, virtual_host):
        """Set up RabbitMQ connection and callback queue."""
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=address,
                port=5672,
                virtual_host=virtual_host,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
        )
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue_name = result.method.queue
        self.channel.basic_consume(
            queue=self.callback_queue_name,
            on_message_callback=self._on_response,
            auto_ack=True
        )
        self.logger.log_info(
            f"[RpcClient L{self.layer_id}] Initialized. Waiting on queue '{self.callback_queue_name}'."
        )

    def _on_response(self, ch, method, props, body):
        """Handle incoming response messages from the server.

        Args:
            ch: Pika channel.
            method: Pika method frame.
            props: Pika properties.
            body: Message body.
        """
        if props.correlation_id != self.correlation_id:
            self.logger.log_warning(
                f"[RpcClient L{self.layer_id}] Mismatched correlation ID. "
                f"Expected: {self.correlation_id}, Got: {props.correlation_id}. "
                f"Body (first 100 bytes): {body[:100]}"
            )
            return

        try:
            self.response_payload = pickle.loads(body)
            self.logger.log_info(
                f"[RpcClient L{self.layer_id}] Received RPC response. CorrID: {props.correlation_id}"
            )
            self._process_response()
        except pickle.UnpicklingError as e:
            self._handle_response_error(f"Unpickling error: {e}", body)
        except KeyError as e:
            self._handle_response_error(f"Missing key in response: {e}", self.response_payload)
        except Exception as e:
            self._handle_response_error(f"General error: {e}")

    def _process_response(self):
        """Process the server response based on the action."""
        if not isinstance(self.response_payload, dict):
            self._handle_response_error(
                f"Response is not a dictionary. Got type: {type(self.response_payload)}"
            )
            return

        action = self.response_payload.get("action")
        server_message = self.response_payload.get("message")
        if server_message:
            self.logger.log_info(f"[RpcClient L{self.layer_id}] Server message: {server_message}")

        self.logger.log_info(f"[RpcClient L{self.layer_id}] Action: {action}")

        if action == "START":
            self._handle_start_action()
        elif action == "REGISTERED":
            self.logger.log_info(f"[RpcClient L{self.layer_id}] Successfully registered with server.")
        elif action == "ERROR":
            error_msg = server_message or self.response_payload.get('details', 'Unknown error')
            self._handle_response_error(f"Server error: {error_msg}")
        else:
            self.logger.log_warning(f"[RpcClient L{self.layer_id}] Unhandled action: {action}")

    def _handle_start_action(self):
        """Handle START action by extracting and saving parameters."""
        params = {
            "model_name": self.response_payload.get("model_name"),
            "num_layers": self.response_payload.get("num_layers"),
            "splits": self.response_payload.get("splits"),
            "save_layers": self.response_payload.get("save_layers"),
            "batch_frame": self.response_payload.get("batch_frame"),
            "encoded_model": self.response_payload.get("model"),
            "data_source": self.response_payload.get("data"),
            "debug_mode": self.response_payload.get("debug_mode", False),
            "imgsz": self.response_payload.get("imgsz")
        }

        if not params["model_name"]:
            self._handle_response_error("'model_name' missing in START parameters")
            return

        params["model_save_path"] = f'{params["model_name"]}.pt'
        self._save_model(params["encoded_model"], params["model_save_path"])
        self.initial_params = params

    def _save_model(self, encoded_model, save_path):
        """Save the encoded model to a file.

        Args:
            encoded_model (str): Base64-encoded model string.
            save_path (str): Path to save the model file.
        """
        if not encoded_model:
            self.logger.log_warning(f"[RpcClient L{self.layer_id}] No encoded model provided.")
            return
        if not save_path:
            self.logger.log_warning(f"[RpcClient L{self.layer_id}] Model save path not determined.")
            return
        if os.path.exists(save_path):
            self.logger.log_info(f"[RpcClient L{self.layer_id}] Model file {save_path} already exists.")
            return

        try:
            model_bytes = base64.b64decode(encoded_model)
            with open(save_path, "wb") as f:
                f.write(model_bytes)
            self.logger.log_info(f"[RpcClient L{self.layer_id}] Model saved to {save_path}")
        except Exception as e:
            self._handle_response_error(f"Error saving model: {e}")

    def _handle_response_error(self, error_msg, payload=None):
        """Log an error and set initial_params with the error.

        Args:
            error_msg (str): Error message to log.
            payload: Optional payload to include in the log.
        """
        log_msg = f"[RpcClient L{self.layer_id}] Error: {error_msg}"
        if payload:
            log_msg += f" Payload: {payload}"
        self.logger.log_error(log_msg)
        self.initial_params = {"error": error_msg}

    def send_to_server(self, message_dict):
        """Send a message to the server via RPC.

        Args:
            message_dict (dict): Message to send, including action and client info.
        """
        self.response_payload = None
        self.initial_params = None
        self.correlation_id = str(uuid.uuid4())

        try:
            self.channel.basic_publish(
                exchange='',
                routing_key='rpc_queue',
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue_name,
                    correlation_id=self.correlation_id
                ),
                body=pickle.dumps(message_dict)
            )
            self.logger.log_info(
                f"[RpcClient L{self.layer_id}] Sent message to 'rpc_queue'. "
                f"CorrID: {self.correlation_id}, Action: {message_dict.get('action')}"
            )
        except Exception as e:
            self.logger.log_error(f"[RpcClient L{self.layer_id}] Failed to publish message: {e}")

    def wait_response(self, timeout_seconds=300):
        """Wait for a server response with a timeout.

        Args:
            timeout_seconds (int): Maximum time to wait for a response.

        Returns:
            bool: True if successful START received, False on error or timeout.
        """
        self.logger.log_info(
            f"[RpcClient L{self.layer_id}] Waiting for server response (timeout: {timeout_seconds}s)..."
        )
        start_time = time.time()

        while True:
            self.connection.process_data_events(time_limit=1)

            if self.initial_params:
                if "error" in self.initial_params:
                    self.logger.log_error(
                        f"[RpcClient L{self.layer_id}] Error: {self.initial_params['error']}"
                    )
                    return False
                if self.initial_params.get("model_name"):
                    self.logger.log_info(
                        f"[RpcClient L{self.layer_id}] START signal processed."
                    )
                    return True

            if self.response_payload and self.response_payload.get("action") == "REGISTERED":
                if self.initial_params is None:
                    self.logger.log_info(
                        f"[RpcClient L{self.layer_id}] REGISTERED received, waiting for START."
                    )
                    self.response_payload = None

            if time.time() - start_time > timeout_seconds:
                self.logger.log_error(
                    f"[RpcClient L{self.layer_id}] Timeout after {timeout_seconds}s."
                )
                return bool(self.initial_params and self.initial_params.get("model_name"))

    def close(self):
        """Close the RabbitMQ connection."""
        try:
            if self.connection and self.connection.is_open:
                self.logger.log_info(f"[RpcClient L{self.layer_id}] Closing RabbitMQ connection.")
                self.connection.close()
        except Exception as e:
            self.logger.log_error(f"[RpcClient L{self.layer_id}] Error closing connection: {e}")