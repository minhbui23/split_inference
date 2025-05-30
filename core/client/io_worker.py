
import threading
import time
import json 
import queue
import pika
from core.utils.fps_logger import FPSLogger
from core.utils.data_transfer import RedisManager, HybridDataTransfer


class BaseIOWorker(threading.Thread):
    """Base class for I/O (Network and Redis) processing threads.

    Manages RabbitMQ/Redis connections and the main loop for communicating
    with other threads via queues.

    Args:
        layer_id (int): The ID of the layer this worker belongs to.
        num_layers (int): The total number of layers in the pipeline.
        rabbit_conn_params (dict): RabbitMQ connection parameters.
        redis_conn_params (dict): Redis connection parameters.
        initial_params (dict): Initialization parameters from the server.
        input_q (queue.Queue): Queue for receiving data to be sent.
        output_q (queue.Queue): Queue for holding processed data to be sent.
        ack_trigger_q (queue.Queue): Queue for sending ACK signals.
        stop_evt (threading.Event): Event to safely stop the thread.
        logger: The logger instance.
        name (str, optional): The name of the thread. Defaults to None.
    """
    def __init__(self, layer_id, num_layers, rabbit_conn_params, redis_conn_params, initial_params,
                 input_q, output_q, ack_trigger_q, stop_evt, logger, name=None):
        super().__init__(name=name or f"IOThread-L{layer_id}")
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.rabbit_conn_params = rabbit_conn_params
        self.redis_conn_params = redis_conn_params
        self.initial_params = initial_params
        self.input_q = input_q
        self.output_q = output_q
        self.ack_trigger_q = ack_trigger_q
        self.stop_evt = stop_evt
        self.logger = logger
        self.is_first_layer = layer_id == 1
        self.is_last_layer = layer_id == num_layers
        self.connection = None
        self.channel = None
        self.consumer_tag = None
        self.redis_manager = None
        self.data_transfer_handler = None
        self._initialize_params()

    def _initialize_params(self):
        """Initializes configuration parameters from the initial_params dictionary."""
        ...
        self.prefetch_val = self.initial_params.get("io_prefetch_count", 5)
        self.rabbit_retry_delay = self.initial_params.get("rabbit_retry_delay", 5)
        self.process_events_timeout = self.initial_params.get("io_process_events_timeout", 0.1)
        self.output_q_timeout = self.initial_params.get("io_output_q_timeout", 0.05)
        self.ack_queue_process_delay = self.initial_params.get("ack_queue_process_delay", 0.05)
        self.redis_tensor_ttl = self.initial_params.get("redis_tensor_ttl_seconds", 300)

    def _connect_rabbitmq(self):
        """Establishes a connection to the RabbitMQ server.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        ...
        try:
            credentials = pika.PlainCredentials(
                self.rabbit_conn_params["username"], self.rabbit_conn_params["password"])
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(
                host=self.rabbit_conn_params["address"],
                port=self.rabbit_conn_params["port"],
                virtual_host=self.rabbit_conn_params["virtual_host"],
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            ))
            self.channel = self.connection.channel()
            self.logger.log_info(f"[{self.name}] RabbitMQ connected.")
            return True
        except Exception as e:
            self.logger.log_error(f"[{self.name}] RabbitMQ connection failed: {e}")
            return False

    def _setup_redis(self):
        """Establishes connection to Redis and initializes HybridDataTransfer."""
        ...
        try:
            self.redis_manager = RedisManager(
                host=self.redis_conn_params["host"],
                port=self.redis_conn_params["port"],
                db=self.redis_conn_params["db"],
                password=self.redis_conn_params.get("password"),
                logger=self.logger
            )
            if self.redis_manager.is_connected():
                self.data_transfer_handler = HybridDataTransfer(
                    pika_channel=self.channel,
                    redis_manager=self.redis_manager,
                    logger=self.logger,
                    default_ttl_seconds=self.redis_tensor_ttl
                )
                self.logger.log_info(f"[{self.name}] Redis connected.")
        except Exception as e:
            self.logger.log_error(f"[{self.name}] Redis setup failed: {e}")
            self.redis_manager = None
            self.data_transfer_handler = None


class FirstLayerIOWorker(BaseIOWorker):
    """I/O thread for the first layer client.

    Its primary task is to get inference results from the `output_q` and
    send them to the next layer via RabbitMQ.
    """
    def run(self):
        """The main loop of the thread, which sends data."""
        ...

        if not self._connect_rabbitmq():
            self.logger.log_error(f"[{self.name}] RabbitMQ connection failed during _connect_rabbitmq. Exiting.")
            self.stop_evt.set()
            return
        
        self._setup_redis() 
        

        if not self.data_transfer_handler:
            self.logger.log_error(f"[{self.name}] Data Handler (HybridDataTransfer) failed to initialize after _setup_redis. Exiting.")
            self.stop_evt.set()
            return

        self.channel.queue_declare(queue=f"intermediate_queue_{self.layer_id + 1}", durable=False)

        while not self.stop_evt.is_set():
            try:
                item_tuple = self.output_q.get(timeout=self.output_q_timeout)
                
                # Trích xuất dữ liệu
                item_data, target_queue_name= item_tuple
                
                metrics = item_data.get("metrics", {})
                
                # Đo t2 và q1
                put_time = item_data.get('l1_inference_timestamp')
                if put_time:
                    metrics['t2'] = time.time() - put_time
                metrics['q1'] = self.output_q.qsize()

                actual_payload = item_data.get("payload")
                # Thêm timestamp để đo t3
                metrics["l1_sent_timestamp"] = time.time()

                # Gửi đi, metrics chính là additional_metadata
                success = self.data_transfer_handler.send_data(
                    actual_data_payload=actual_payload,
                    rabbitmq_target_queue=target_queue_name,
                    additional_metadata=metrics
                )
                if not success:
                    self.logger.log_error(f"[{self.name}] Failed to send data via Hybrid Transfer.")
                
                self.output_q.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error(f"[{self.name}] Publish error: {e}")
                self.stop_evt.set()


class MiddleLayerIOWorker(BaseIOWorker):
    """I/O thread for intermediate layer clients.

    Its tasks include receiving data from the previous layer and sending
    processed data to the next layer.
    """
    def run(self):
        self.logger.log_info(f"[{self.name}] Middle layer I/O - logic not implemented. Exiting thread.")
        self.stop_evt.set()

class LastLayerIOWorker(BaseIOWorker):
    """I/O thread for the last layer client.

    Its primary task is to receive data from the previous layer via RabbitMQ,
    fetch the payload using Redis (Claim Check pattern), and put it into the
    `input_q` for the Inference thread.
    """
    def run(self):
        """Starts the consumer to listen for messages from RabbitMQ, with detailed checkpoints."""
        self.logger.log_info(f"[{self.name}] Checkpoint 0: Run method started.")
        try:
            # ---- Giai đoạn Kết nối và Thiết lập ----
            if not self._connect_rabbitmq(): 
                self.logger.log_error(f"[{self.name}] Checkpoint FAILED at RabbitMQ connection. Exiting.")
                self.stop_evt.set()
                return
            self.logger.log_info(f"[{self.name}] Checkpoint 1: RabbitMQ connected successfully.")

            self._setup_redis()
            if not self.data_transfer_handler:
                self.logger.log_error(f"[{self.name}] Checkpoint FAILED at Data Handler initialization. Exiting.")
                self.stop_evt.set()
                return
            self.logger.log_info(f"[{self.name}] Checkpoint 2: Redis and Data Handler setup complete.")

            source_queue_name = f"intermediate_queue_{self.layer_id}"

            # ---- Giai đoạn Thiết lập Consumer RabbitMQ ----
            try:
                self.logger.log_info(f"[{self.name}] Checkpoint 3: Declaring queue '{source_queue_name}'.")
                # Khai báo queue với các thuộc tính mong muốn, durable=False nghĩa là queue sẽ mất nếu broker restart
                # exclusive=False (mặc định) nghĩa là nhiều connection có thể access
                # auto_delete=False (mặc định) nghĩa là queue không tự xóa khi consumer cuối cùng disconnect
                self.channel.queue_declare(queue=source_queue_name, durable=False) 
                
                self.logger.log_info(f"[{self.name}] Checkpoint 4: Setting QOS (prefetch_count={self.prefetch_val}).")
                self.channel.basic_qos(prefetch_count=self.prefetch_val)
            except pika.exceptions.ChannelClosedByBroker as e_ch_closed_broker:
                self.logger.log_error(f"[{self.name}] Checkpoint FAILED: Channel closed by broker during queue_declare/qos for '{source_queue_name}'. Error: {e_ch_closed_broker}")
                self.stop_evt.set()
                return
            except pika.exceptions.AMQPConnectionError as e_conn_amqp:
                self.logger.log_error(f"[{self.name}] Checkpoint FAILED: AMQP Connection Error during queue_declare/qos for '{source_queue_name}'. Error: {e_conn_amqp}")
                self.stop_evt.set()
                return
            except Exception as e_setup: # Bắt các lỗi khác
                self.logger.log_error(f"[{self.name}] Checkpoint FAILED: Declare queue or set QOS for '{source_queue_name}'. Error: {e_setup}")
                self.stop_evt.set()
                return
            self.logger.log_info(f"[{self.name}] Checkpoint 5: Queue '{source_queue_name}' declared and QOS set.")

            def _callback(ch, method, properties, body):
                """Callback function invoked when a new message is received from RabbitMQ."""
                self.logger.log_info(f"[{self.name}] Callback: Received message. Delivery Tag: {method.delivery_tag}, Size: {len(body)} bytes.")
                
                if self.stop_evt.is_set():
                    self.logger.log_info(f"[{self.name}] Callback: Stop event is set. Nacking message (tag: {method.delivery_tag}) to requeue.")
                    try:
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                    except Exception as e_nack:
                        self.logger.error(f"[{self.name}] Callback: Error nacking message during stop: {e_nack}")
                    return
                
                try:
                    metrics = json.loads(body.decode()) # Đây là metadata từ layer trước

                    # Logic xử lý STOP message (nếu còn) nên được loại bỏ ở đây,
                    # vì việc dừng giờ được quản lý bởi self.stop_evt chung.

                    l1_sent_time = metrics.get('l1_sent_timestamp') # Hoặc key tương ứng
                    if l1_sent_time:
                        metrics['t3'] = time.time() - l1_sent_time

                    redis_key = metrics.get("redis_key")
                    if not redis_key:
                        self.logger.log_warning(f"[{self.name}] Callback: No redis_key in message (tag: {method.delivery_tag}). Nacking, no requeue. Body: {body[:100]}")
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                        return

                    if not self.data_transfer_handler:
                        self.logger.log_error(f"[{self.name}] Callback: Data_transfer_handler is None. Cannot process message (tag: {method.delivery_tag}). Nacking, requeue.")
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                        return

                    actual_payload = self.data_transfer_handler.receive_data_from_metadata(
                        metrics, delete_after_retrieval=True
                    )

                    if actual_payload is not None: # Kiểm tra actual_payload có giá trị
                        item_for_inference = {
                            "payload": actual_payload,
                            "delivery_tag": method.delivery_tag,
                            "metrics": metrics, # Truyền metrics đã được cập nhật t3
                            "put_to_l2_input_q_timestamp": time.time()
                        }
                        if not self.stop_evt.is_set():
                            try:
                                self.input_q.put(item_for_inference, timeout=0.1) # Thêm timeout nhỏ
                                ch.basic_ack(delivery_tag=method.delivery_tag) # ACK sau khi put thành công
                                self.logger.log_info(f"[{self.name}] Callback: Message (tag: {method.delivery_tag}) processed, put to input_q, and acked.")
                            except queue.Full:
                                self.logger.log_warning(f"[{self.name}] Callback: Input queue full. Nacking message (tag: {method.delivery_tag}) to requeue.")
                                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                        else:
                            self.logger.info(f"[{self.name}] Callback: Stop event set during put to input_q. Nacking message (tag: {method.delivery_tag}).")
                            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                    else:
                        self.logger.log_error(f"[{self.name}] Callback: Failed to retrieve data from Redis for key {redis_key} (tag: {method.delivery_tag}). Nacking, no requeue.")
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

                except json.JSONDecodeError as e_json:
                    self.logger.error(f"[{self.name}] Callback: JSONDecodeError for message (tag: {method.delivery_tag}). Body: {body[:200]}. Error: {e_json}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False) # Không requeue message JSON lỗi
                except Exception as e_cb:
                    self.logger.log_error(f"[{self.name}] Callback: Unhandled error processing message (tag: {method.delivery_tag}). Error: {e_cb}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False) # Requeue cho các lỗi khác

            try:
                self.logger.log_info(f"[{self.name}] Checkpoint 6: Setting up consumer for queue '{source_queue_name}'.")
                # auto_ack=False là mặc định và được khuyến khích để xử lý ack/nack thủ công
                self.consumer_tag = self.channel.basic_consume(
                    queue=source_queue_name,
                    on_message_callback=_callback,
                    auto_ack=False # Quan trọng: Xử lý ACK/NACK thủ công trong callback
                )
            except Exception as e_consume:
                self.logger.log_error(f"[{self.name}] Checkpoint FAILED: basic_consume for queue '{source_queue_name}'. Error: {e_consume}")
                self.stop_evt.set()
                return
            
            self.logger.log_info(f"[{self.name}] Checkpoint 7: Consumer started (tag: {self.consumer_tag}). Waiting for messages on '{source_queue_name}'.")

            # ---- Vòng lặp Xử lý Sự kiện Chính ----
            self.logger.log_info(f"[{self.name}] Checkpoint 8: Entering main event loop.")
            while not self.stop_evt.is_set():
                try:
                    # self.logger.log_debug(f"[{self.name}] Checkpoint 8.1: Calling process_data_events.")
                    self.connection.process_data_events(time_limit=self.process_events_timeout)
                    # self.logger.log_debug(f"[{self.name}] Checkpoint 8.2: Returned from process_data_events.")
                except pika.exceptions.StreamLostError as e_stream:
                    self.logger.error(f"[{self.name}] RabbitMQ StreamLostError in event loop. Setting stop_evt. Error: {e_stream}")
                    self.stop_evt.set() # Kích hoạt dừng
                except pika.exceptions.AMQPConnectionError as e_conn_amqp: # Bắt lỗi kết nối cụ thể hơn
                    self.logger.error(f"[{self.name}] RabbitMQ AMQPConnectionError in event loop. Setting stop_evt. Error: {e_conn_amqp}")
                    self.stop_evt.set() # Kích hoạt dừng
                except Exception as e_loop: # Bắt các lỗi không mong muốn khác
                    if not self.stop_evt.is_set(): # Chỉ log nếu chưa bị dừng bởi nguyên nhân khác
                        self.logger.log_error(f"[{self.name}] Unexpected error in event loop. Setting stop_evt. Error: {e_loop}")
                    self.stop_evt.set() # Kích hoạt dừng
            self.logger.log_info(f"[{self.name}] Checkpoint 9: Exited main event loop (stop_evt is {self.stop_evt.is_set()}).")

        finally:
            # ---- Giai đoạn Dọn dẹp ----
            self.logger.log_info(f"[{self.name}] Checkpoint 10: Entering finally block for cleanup.")
            if hasattr(self, 'consumer_tag') and self.consumer_tag and self.channel and self.channel.is_open:
                try:
                    self.logger.log_info(f"[{self.name}] Attempting to cancel consumer (tag: {self.consumer_tag}).")
                    self.channel.basic_cancel(self.consumer_tag)
                    self.logger.log_info(f"[{self.name}] Consumer cancelled.")
                except Exception as e_cancel:
                    self.logger.error(f"[{self.name}] Error cancelling consumer: {e_cancel}")
            
            # Việc đóng connection thường được quản lý bởi luồng chính của client sau khi join tất cả các thread.
            # Hoặc nếu luồng này sở hữu connection một cách độc quyền và cần tự đóng.
            # Hiện tại, giả định connection được quản lý/đóng bởi luồng chính.
            
            self.logger.log_info(f"[{self.name}] Checkpoint 11: Run method finished execution.")