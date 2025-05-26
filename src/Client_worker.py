import threading
import queue
import time
import pickle
import os
import cv2
import torch
import pika
import pika.exceptions
from src.Utils import FPSLogger

class InferenceWorker(threading.Thread):
    def __init__(self, layer_id, num_layers, device, model_obj, predictor_obj, initial_params,
                 input_q, output_q, ack_trigger_q, stop_evt, logger, name=None):
        """Initialize the InferenceWorker thread for processing model inference.

        Args:
            layer_id (int): Layer ID of the client.
            num_layers (int): Total number of layers in the pipeline.
            device: Device for model inference (e.g., CPU/GPU).
            model_obj: Instance of SplitDetectionModel.
            predictor_obj: Instance of Predictor (e.g., YOLO or DetectionPredictor).
            initial_params (dict): Initial parameters from server.
            input_q: Queue for input data.
            output_q: Queue for output data.
            ack_trigger_q: Queue for sending ACK/NACK triggers.
            stop_evt: Threading event to signal stop.
            logger: Logger instance for logging.
            name (str, optional): Thread name.
        """
        super().__init__(name=name or f"InferenceThread-L{layer_id}")
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.device = device
        self.model_obj = model_obj
        self.predictor_obj = predictor_obj
        self.initial_params = initial_params
        self.input_q = input_q
        self.output_q = output_q
        self.ack_trigger_q = ack_trigger_q
        self.stop_evt = stop_evt
        self.logger = logger
        self._initialize_params()

    def _initialize_params(self):
        """Extract and set parameters from initial_params."""
        self.batch_frame_size = self.initial_params.get("batch_frame", 1)
        imgsz = self.initial_params.get("imgsz", (640, 640))
        self.img_width, self.img_height = int(imgsz[0]), int(imgsz[1])
        log_prefix = self._get_log_prefix()
        self.fps_logger = FPSLogger(
            layer_id=self.layer_id,
            logger_obj=self.logger,
            log_interval_seconds=self.initial_params.get("fps_log_interval", 10),
            log_prefix=log_prefix
        )

    def _get_log_prefix(self):
        """Determine log prefix based on layer position."""
        if self.layer_id == 1:
            return "Batch (L1 Video)"
        if self.layer_id == self.num_layers and self.layer_id > 1:
            return "Batch of features (L-Last)"
        return f"Batch of features (L{self.layer_id} Middle)"

    def run(self):
        """Run the inference thread logic."""
        self.logger.log_info(f"[{self.name}] Starting.")
        try:
            if self.layer_id == 1:
                self._process_first_layer()
            elif self.layer_id > 1:
                self._process_subsequent_layer()
            else:
                self.logger.log_error(f"[{self.name}] Invalid layer_id: {self.layer_id}.")
                self.stop_evt.set()
        except Exception as e:
            self.logger.log_error(f"[{self.name}] Critical error: {e}")
            self.stop_evt.set()
        finally:
            self.logger.log_info(f"[{self.name}] Stopped.")

    def _process_first_layer(self):
        """Process video input for Layer 1."""
        video_path = self.initial_params.get("data_source")
        if not video_path or not os.path.exists(video_path):
            self.logger.log_error(f"[{self.name}] L1: Video file not found: {video_path}")
            self.stop_evt.set()
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.log_error(f"[{self.name}] L1: Cannot open video: {video_path}")
            self.stop_evt.set()
            return

        self.logger.log_info(f"[{self.name}] L1: Processing video: {video_path}")
        frames_batch = []
        frame_count = 0
        save_layers = self.initial_params.get("save_layers")

        while not self.stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                self._handle_end_of_video(frame_count, frames_batch)
                break

            frame_count += 1
            try:
                resized_frame = cv2.resize(frame, (self.img_width, self.img_height))
                tensor = torch.from_numpy(resized_frame).float().permute(2, 0, 1) / 255.0
                frames_batch.append(tensor)
            except Exception as e:
                self.logger.log_error(f"[{self.name}] L1: Error processing frame {frame_count}: {e}")
                continue

            if len(frames_batch) == self.batch_frame_size:
                self._process_batch(frames_batch, save_layers)
                frames_batch = []

        cap.release()
        self.fps_logger.log_overall_fps(process_description="L1: Video processing finished")

    def _handle_end_of_video(self, frame_count, frames_batch):
        """Handle end of video stream for Layer 1."""
        self.logger.log_info(f"[{self.name}] L1: End of video. Frames read: {frame_count}")
        if frames_batch:
            self.logger.log_info(f"[{self.name}] L1: {len(frames_batch)} frames remaining, not processed.")
        if self.layer_id < self.num_layers:
            self.output_q.put(("STOP_INFERENCE", f"intermediate_queue_{self.layer_id + 1}"))

    def _process_batch(self, frames_batch, save_layers):
        """Process a batch of frames for Layer 1."""
        
        try:

            batch = torch.stack(frames_batch).to(self.device)
            batch_size = batch.size(0)

            self.predictor_obj.setup_source(batch)
            input_tensor = self._prepare_input_tensor(batch)

            self.fps_logger.start_batch_timing()

            output = self.model_obj.forward_head(input_tensor, save_layers)
            output["l1_processed_timestamp"] = time.time()
            output["layers_output"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in output["layers_output"]]

            if self.layer_id < self.num_layers:
                self.output_q.put((output, f"intermediate_queue_{self.layer_id + 1}"))
                
            self.fps_logger.end_batch_and_log_fps(batch_size)

        except Exception as e:
            self.logger.log_error(f"[{self.name}] L1: Error processing batch: {e}")
            self.fps_logger._current_batch_start_time = None

    def _prepare_input_tensor(self, batch):
        """Prepare input tensor for model inference."""
        for data in self.predictor_obj.dataset:
            if isinstance(data, tuple) and len(data) > 1:
                return data[1].to(self.device) if isinstance(data[1], torch.Tensor) else data[1]
        return batch.to(self.device)

    def _process_subsequent_layer(self):
        """Process input data for Layer 2 or middle layers."""
        is_last_layer = self.layer_id == self.num_layers
        self.logger.log_info(
            f"[{self.name}] Waiting for data from input_q (Batch size: {self.batch_frame_size}). "
            f"Layer {self.layer_id} of {self.num_layers}."
        )

        while not self.stop_evt.is_set():
            try:
                item = self.input_q.get(timeout=0.5)
                if item == "STOP_FROM_PREVIOUS":
                    self._handle_stop_signal(is_last_layer)
                    break

                payload = item.get("payload")
                delivery_tag = item.get("delivery_tag")
                if payload is None or delivery_tag is None:
                    self.logger.log_warning(f"[{self.name}] Invalid item: {item}")
                    self.input_q.task_done()
                    continue

                self._process_payload(payload, delivery_tag, is_last_layer)
                self.input_q.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error(f"[{self.name}] Unexpected error: {e}")
                if self.stop_evt.is_set():
                    break
                time.sleep(0.1)

        self.fps_logger.log_overall_fps(process_description=f"L{self.layer_id}: Feature processing finished")

    def _handle_stop_signal(self, is_last_layer):
        """Handle STOP signal from the previous layer."""
        self.logger.log_info(f"[{self.name}] Received STOP from input_q.")
        if not is_last_layer:
            self.logger.log_info(f"[{self.name}] Middle layer, forwarding STOP_INFERENCE.")
            self.output_q.put(("STOP_INFERENCE", f"intermediate_queue_{self.layer_id + 1}"))

    def _process_payload(self, payload, delivery_tag, is_last_layer):
        """Process payload data for subsequent layers."""
        self.fps_logger.start_batch_timing()
        ack_status = "failure"
        requeue = False

        try:
            if not (isinstance(payload, dict) and "layers_output" in payload):
                raise ValueError(f"Payload for L{self.layer_id} invalid format.")

            arrival_ts = time.time()
            send_ts = payload.get("l1_processed_timestamp")
            if send_ts:
                self.logger.log_info(
                    f"[{self.name}] Packet propagation from prev layer: {arrival_ts - send_ts:.4f}s"
                )

            payload["layers_output"] = [
                t.to(self.device) if isinstance(t, torch.Tensor) else None for t in payload["layers_output"]
            ]

            if is_last_layer:
                self.model_obj.forward_tail(payload)
            # Middle layer logic can be added here if needed
            ack_status = "success"

        except Exception as e:
            self.logger.log_error(f"[{self.name}] Error processing payload for tag {delivery_tag}: {e}")
        finally:
            self.fps_logger.end_batch_and_log_fps(self.batch_frame_size)
            self.ack_trigger_q.put({"delivery_tag": delivery_tag, "status": ack_status, "requeue": requeue})

class IOWorker(threading.Thread):
    def __init__(self, layer_id, num_layers, rabbit_conn_params, initial_params,
                 input_q, output_q, ack_trigger_q, stop_evt, logger, name=None):
        """Initialize the IOWorker thread for RabbitMQ communication.

        Args:
            layer_id (int): Layer ID of the client.
            num_layers (int): Total number of layers in the pipeline.
            rabbit_conn_params (dict): RabbitMQ connection parameters.
            initial_params (dict): Initial parameters from server.
            input_q: Queue for input data.
            output_q: Queue for output data.
            ack_trigger_q: Queue for ACK/NACK triggers.
            stop_evt: Threading event to signal stop.
            logger: Logger instance for logging.
            name (str, optional): Thread name.
        """
        super().__init__(name=name or f"IOThread-L{layer_id}")
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.rabbit_conn_params = rabbit_conn_params
        self.initial_params = initial_params
        self.input_q = input_q
        self.output_q = output_q
        self.ack_trigger_q = ack_trigger_q
        self.stop_evt = stop_evt
        self.logger = logger
        self.is_first_layer = self.layer_id == 1
        self.is_last_layer = self.layer_id == num_layers
        self._initialize_params()
        self.connection = None
        self.channel = None
        self.consumer_tag = None

    def _initialize_params(self):
        """Extract and set parameters from initial_params."""
        self.prefetch_val = self.initial_params.get("io_prefetch_count", 5)
        self.rabbit_retry_delay = self.initial_params.get("rabbit_retry_delay", 5)
        self.process_events_timeout = self.initial_params.get("io_process_events_timeout", 0.1)
        self.output_q_timeout = self.initial_params.get("io_output_q_timeout", 0.05)
        self.ack_queue_process_delay = self.initial_params.get("ack_queue_process_delay", 0.05)

    def run(self):
        """Run the IO thread logic."""
        self.logger.log_info(f"[{self.name}] Starting.")
        if not self._connect_rabbitmq():
            self.logger.log_error(f"[{self.name}] Failed to connect to RabbitMQ.")
            self.stop_evt.set()
            return

        try:
            self._setup_queues()
            if self.layer_id > 1:
                self._start_consumer()
                self._process_ack_queue()
            self._main_loop()
        except Exception as e:
            self.logger.log_error(f"[{self.name}] Critical error: {e}")
            self.stop_evt.set()
        finally:
            self._cleanup()
            self.logger.log_info(f"[{self.name}] Stopped.")

    def _connect_rabbitmq(self):
        """Establish RabbitMQ connection."""
        while not self.stop_evt.is_set():
            try:
                credentials = pika.PlainCredentials(
                    self.rabbit_conn_params["username"], self.rabbit_conn_params["password"]
                )
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=self.rabbit_conn_params["address"],
                        port=self.rabbit_conn_params["port"],
                        virtual_host=self.rabbit_conn_params["virtual_host"],
                        credentials=credentials,
                        heartbeat=600,
                        blocked_connection_timeout=300
                    )
                )
                self.channel = self.connection.channel()
                self.logger.log_info(f"[{self.name}] RabbitMQ connection established.")
                return True
            except pika.exceptions.AMQPConnectionError as e:
                self.logger.log_error(
                    f"[{self.name}] Could not connect to RabbitMQ: {e}. Retrying in {self.rabbit_retry_delay}s."
                )
                time.sleep(self.rabbit_retry_delay)
        return False

    def _setup_queues(self):
        """Declare RabbitMQ queues for sending and receiving."""
        if self.layer_id < self.num_layers:
            self.channel.queue_declare(queue=f"intermediate_queue_{self.layer_id + 1}", durable=False)
            self.logger.log_info(f"[{self.name}] Declared send queue: intermediate_queue_{self.layer_id + 1}")

        if self.layer_id > 1:
            self.channel.queue_declare(queue=f"intermediate_queue_{self.layer_id}", durable=False)
            self.channel.basic_qos(prefetch_count=self.prefetch_val)
            self.logger.log_info(
                f"[{self.name}] Declared listen queue: intermediate_queue_{self.layer_id}, QOS: {self.prefetch_val}."
            )

    def _start_consumer(self):
        """Start consuming messages from the input queue."""
        self.consumer_tag = self.channel.basic_consume(
            queue=f"intermediate_queue_{self.layer_id}",
            on_message_callback=self._on_message,
            auto_ack=False
        )
        self.logger.log_info(
            f"[{self.name}] Consumer started on intermediate_queue_{self.layer_id} with tag: {self.consumer_tag}."
        )

    def _on_message(self, ch, method, properties, body):
        """Handle incoming RabbitMQ messages."""
        if self.stop_evt.is_set():
            self.logger.log_info(f"[{self.name} Callback] Stop event. Ignoring message.")
            return

        try:
            message = pickle.loads(body)
            if message == "STOP":
                self.logger.log_info(f"[{self.name} Callback] Received STOP.")
                self.input_q.put("STOP_FROM_PREVIOUS")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self.stop_evt.set()
            else:
                payload = message.get("data") if isinstance(message, dict) else message
                if payload is not None:
                    self.input_q.put({"payload": payload, "delivery_tag": method.delivery_tag})
                else:
                    self.logger.log_warning(f"[{self.name} Callback] Invalid message. Nacking.")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except pickle.UnpicklingError:
            self.logger.log_error(f"[{self.name} Callback] Unpickle error. Dropping.")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            self.logger.log_error(f"[{self.name} Callback] Error: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def _process_ack_queue(self):
        """Process ACK/NACK triggers from ack_trigger_q."""
        if self.stop_evt.is_set():
            return

        while not self.ack_trigger_q.empty():
            try:
                ack_info = self.ack_trigger_q.get_nowait()
                delivery_tag = ack_info.get("delivery_tag")
                status = ack_info.get("status")
                requeue = ack_info.get("requeue", False)

                if delivery_tag and self.channel and self.channel.is_open:
                    if status == "success":
                        self.channel.basic_ack(delivery_tag=delivery_tag)
                    elif status == "failure":
                        self.channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
                self.ack_trigger_q.task_done()
            except queue.Empty:
                break
            except pika.exceptions.AMQPError as e:
                self.logger.log_error(f"[{self.name}] AMQP error in ack processing: {e}")
                break
            except Exception as e:
                self.logger.log_error(f"[{self.name}] Error in ack processing: {e}")

        if self.connection and self.connection.is_open and not self.stop_evt.is_set():
            self.connection.call_later(self.ack_queue_process_delay, self._process_ack_queue)

    def _main_loop(self):
        """Main loop for sending data and processing Pika events."""
        while not self.stop_evt.is_set():
            if self.layer_id < self.num_layers:
                self._send_output_data()

            if self.connection and self.connection.is_open:
                try:
                    self.connection.process_data_events(time_limit=self.process_events_timeout)
                except (pika.exceptions.StreamLostError, pika.exceptions.AMQPConnectionError) as e:
                    self.logger.log_error(f"[{self.name}] Pika error: {e}")
                    self.stop_evt.set()
                    break
                except Exception as e:
                    self.logger.log_error(f"[{self.name}] Pika event error: {e}")
                    self.stop_evt.set()
                    break
            else:
                self.logger.log_warning(f"[{self.name}] Connection not open.")
                self.stop_evt.set()
                break

    def _send_output_data(self):
        """Send data from output queue to RabbitMQ."""
        try:
            item = self.output_q.get(block=True, timeout=self.output_q_timeout)
            data_payload, target_queue = item
            message_body = pickle.dumps("STOP") if data_payload == "STOP_INFERENCE" else pickle.dumps({"action": "OUTPUT", "data": data_payload})

            if self.channel and self.channel.is_open:
                self.channel.basic_publish(exchange='', routing_key=target_queue, body=message_body)
                self.logger.log_info(f"[{self.name}] Sent data to {target_queue}")
            else:
                self.logger.log_warning(f"[{self.name}] Channel closed. Re-queuing item.")
                self.output_q.put(item)
                time.sleep(0.1)

            self.output_q.task_done()
        except queue.Empty:
            pass
        except pika.exceptions.AMQPError as e:
            self.logger.log_error(f"[{self.name}] Publish error: {e}")
            self.stop_evt.set()
        except Exception as e:
            self.logger.log_error(f"[{self.name}] Publish error: {e}")

    def _cleanup(self):
        """Clean up RabbitMQ consumer and connection."""
        if self.layer_id > 1 and self.consumer_tag and self.channel and self.channel.is_open:
            try:
                self.channel.basic_cancel(self.consumer_tag)
                self.logger.log_info(f"[{self.name}] Consumer cancelled.")
            except Exception as e:
                self.logger.log_error(f"[{self.name}] Error cancelling consumer: {e}")

        if self.connection and self.connection.is_open:
            try:
                if self.channel and self.channel.is_open:
                    self.channel.close()
                self.connection.close()
                self.logger.log_info(f"[{self.name}] RabbitMQ connection closed.")
            except Exception as e:
                self.logger.log_error(f"[{self.name}] Error closing RabbitMQ: {e}")