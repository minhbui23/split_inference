
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
                 input_q, output_q, stop_evt, logger, 
                 name=None):
        super().__init__(name=name or f"IOThread-L{layer_id}")
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.rabbit_conn_params = rabbit_conn_params
        self.redis_conn_params = redis_conn_params
        self.initial_params = initial_params
        self.input_q = input_q
        self.output_q = output_q
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
    """
    I/O thread for the final layer client. It consumes data from RabbitMQ,
    retrieves payloads from Redis, and puts them into an internal queue for the
    InferenceWorker. It implements a robust backpressure mechanism to handle
    high loads gracefully.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the worker and backpressure state variables."""
        super().__init__(*args, **kwargs)
        self.is_consuming = False
        self.backoff_attempt = 0
        self.source_queue_name = f"intermediate_queue_{self.layer_id}"
        
        # Backpressure configuration constants
        self.INITIAL_BACKOFF_SECONDS = 1.0
        self.MAX_BACKOFF_SECONDS = 16.0
        self.RESUME_CONSUMPTION_THRESHOLD = 0.25 # Resume when queue is 25% full or less

    def _setup_consumer(self):
        """Declares the queue, sets QOS, and starts the consumer."""
        try:
            self.logger.log_info(f"[{self.name}] Declaring queue '{self.source_queue_name}' and setting QOS.")
            self.channel.queue_declare(queue=self.source_queue_name, durable=False)
            self.channel.basic_qos(prefetch_count=self.prefetch_val)
            self.consumer_tag = self.channel.basic_consume(
                queue=self.source_queue_name,
                on_message_callback=self._on_message_callback,
                auto_ack=False
            )
            self.is_consuming = True
            self.logger.log_info(f"[{self.name}] Consumer started (tag: {self.consumer_tag}). Waiting for messages.")
            return True
        except Exception as e:
            self.logger.log_error(f"[{self.name}] Failed to set up consumer: {e}")
            self.stop_evt.set()
            return False

    def _pause_consumer_due_to_backpressure(self, channel, delivery_tag):
        """
        Nacks the current message and pauses the consumer by cancelling it.
        Schedules a check to resume consumption later.
        """
        # 1. Nack the message to return it to the queue
        channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
        
        # 2. Pause consumption if it's currently active
        if self.is_consuming:
            try:
                channel.basic_cancel(self.consumer_tag)
                self.is_consuming = False
                self.logger.log_info(f"[{self.name}] Consumer paused due to backpressure.")
                
                # 3. Schedule the first attempt to resume
                self.backoff_attempt = 1
                self.connection.call_later(self.INITIAL_BACKOFF_SECONDS, self._resume_consumption)
            except Exception as e:
                self.logger.log_error(f"[{self.name}] Failed to cancel consumer: {e}")
                self.stop_evt.set() # Critical failure, stop the thread

    def _resume_consumption(self):
        """
        Checks if the internal queue has capacity. If so, resumes consumption.
        If not, schedules another check with exponential backoff.
        """
        if self.stop_evt.is_set() or not (self.connection and self.connection.is_open):
            return

        # Check if the queue has drained below the threshold
        if self.input_q.qsize() < (self.input_q.maxsize * self.RESUME_CONSUMPTION_THRESHOLD):
            self.logger.log_info(f"[{self.name}] Input queue has capacity. Resuming consumption.")
            self.backoff_attempt = 0 # Reset backoff on success
            if not self._setup_consumer():
                self.logger.log_error(f"[{self.name}] Could not resume consumption. Stopping.")
                self.stop_evt.set()
        else:
            # If queue is still full, schedule the next check with backoff
            self.backoff_attempt += 1
            wait_time = min(self.INITIAL_BACKOFF_SECONDS * (2 ** (self.backoff_attempt - 1)), self.MAX_BACKOFF_SECONDS)
            
            self.logger.log_info(f"[{self.name}] Input queue still full ({self.input_q.qsize()}). Will check again in {wait_time:.2f}s.")
            self.connection.call_later(wait_time, self._resume_consumption)
            
    def _on_message_callback(self, channel, method, properties, body):
        """
        The main callback function for processing incoming messages from RabbitMQ.
        This version includes detailed checkpoints and robust error handling.
        ACK is performed immediately after successfully queueing the item for inference.
        """

        if self.stop_evt.is_set():
            try:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception as e_nack:
                self.logger.log_error(f"[{self.name}] Callback: Error nacking message during stop: {e_nack}")
            return

        try:
            metrics = json.loads(body.decode())
            
            l1_sent_time = metrics.get('l1_sent_timestamp')
            if l1_sent_time:
                metrics['t3'] = time.time() - l1_sent_time
            
            
            actual_payload = self.data_transfer_handler.receive_data_from_metadata(
                metrics, delete_after_retrieval=True
            )

            if actual_payload is None:
                self.logger.log_error(f"[{self.name}] Failed to retrieve payload from Redis for key {metrics.get('redis_key')}. Discarding message (tag: {method.delivery_tag}).")
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return

            item_for_inference = {
                "payload": actual_payload,
                "delivery_tag": method.delivery_tag,
                "metrics": metrics,
                "put_to_l2_input_q_timestamp": time.time()
            }
            
            try:
                # Measure queue depth *before* putting the item in.
                metrics['q2'] = self.input_q.qsize()

                self.input_q.put_nowait(item_for_inference)
                
                channel.basic_ack(delivery_tag=method.delivery_tag)

            except queue.Full:
                # Backpressure: Internal queue is full. Trigger the pause mechanism.
                self.logger.log_warning(f"[{self.name}] Input queue is full. Triggering backpressure mechanism for message (tag: {method.delivery_tag}).")
                # This will nack(requeue=True) and pause the consumer.
                self._pause_consumer_due_to_backpressure(channel, method.delivery_tag)
        
        except json.JSONDecodeError as e_json:
            self.logger.log_error(f"[{self.name}] JSONDecodeError. Discarding malformed message (tag: {method.delivery_tag}).")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e_unhandled:
            # This block catches any other unexpected errors during the process.
            # Adding exc_info=True will print the full traceback.
            self.logger.log_error(f"[{self.name}] Unhandled error in callback. Discarding message (tag: {method.delivery_tag}). Error: {e_unhandled}")
            try:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception as e_final_nack:
                self.logger.log_error(f"[{self.name}] Failed to even nack after unhandled error: {e_final_nack}")


    def run(self):
        """The main execution method for the thread."""
        self.logger.log_info(f"[{self.name}] Starting.")
        try:
            # --- 1. Setup Phase ---
            if not self._connect_rabbitmq(): return
            self._setup_redis()
            if not self.data_transfer_handler:
                self.logger.log_error(f"[{self.name}] Data Handler failed initialization. Stopping.")
                self.stop_evt.set()
                return
            
            # --- 2. Consumer Start ---
            if not self._setup_consumer(): return
            
            # --- 3. Main Event Loop ---
            self.logger.log_info(f"[{self.name}] Entering main event loop.")
            while not self.stop_evt.is_set():
                try:
                    # Process I/O events for a short duration. This allows Pika's
                    # internal mechanisms, including timers (`call_later`), to run.
                    self.connection.process_data_events(time_limit=1.0)
                except (pika.exceptions.StreamLostError, pika.exceptions.AMQPConnectionError) as e:
                    self.logger.log_error(f"[{self.name}] Connection lost in event loop: {e}. Stopping thread.")
                    self.stop_evt.set() # Trigger a clean shutdown
                except Exception as e:
                    self.logger.log_error(f"[{self.name}] Unexpected error in event loop: {e}. Stopping thread.")
                    self.stop_evt.set()
        
        finally:
            # --- 4. Cleanup Phase ---
            self.logger.log_info(f"[{self.name}] Exiting run loop. Cleaning up.")
            if self.is_consuming and self.channel and self.channel.is_open:
                try:
                    self.channel.basic_cancel(self.consumer_tag)
                    self.logger.log_info(f"[{self.name}] Consumer cancelled successfully during cleanup.")
                except Exception as e:
                    self.logger.error(f"[{self.name}] Error cancelling consumer during cleanup: {e}")
            self.logger.log_info(f"[{self.name}] Run method finished.")