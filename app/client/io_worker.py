
import threading
import torch
import time
import pickle
import os
import json
import queue
import cv2
import pika
from core.utils.fps_logger import FPSLogger
from core.utils.data_transfer import RedisManager, HybridDataTransfer


class BaseIOWorker(threading.Thread):
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
        self.prefetch_val = self.initial_params.get("io_prefetch_count", 5)
        self.rabbit_retry_delay = self.initial_params.get("rabbit_retry_delay", 5)
        self.process_events_timeout = self.initial_params.get("io_process_events_timeout", 0.1)
        self.output_q_timeout = self.initial_params.get("io_output_q_timeout", 0.05)
        self.ack_queue_process_delay = self.initial_params.get("ack_queue_process_delay", 0.05)
        self.redis_tensor_ttl = self.initial_params.get("redis_tensor_ttl_seconds", 300)

    def _connect(self):
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
    def run(self):
        if not self._connect():
            self.stop_evt.set()
            return
        self._setup_redis()
        self.channel.queue_declare(queue=f"intermediate_queue_{self.layer_id + 1}", durable=False)

        while not self.stop_evt.is_set():
            try:
                item = self.output_q.get(timeout=self.output_q_timeout)
                data_payload, target_queue = item
                message_body = pickle.dumps("STOP") if data_payload == "STOP_INFERENCE" else pickle.dumps({"action": "OUTPUT", "data": data_payload})

                self.channel.basic_publish(exchange='', routing_key=target_queue, body=message_body)
                self.logger.log_info(f"[{self.name}] Sent to {target_queue}")
                self.output_q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error(f"[{self.name}] Publish error: {e}")
                self.stop_evt.set()


class MiddleLayerIOWorker(BaseIOWorker):
    def run(self):
        self.logger.log_info(f"[{self.name}] Middle layer I/O - logic not implemented. Exiting thread.")
        self.stop_evt.set()


class LastLayerIOWorker(BaseIOWorker):
    def run(self):
        if not self._connect():
            self.stop_evt.set()
            return
        self._setup_redis()
        self.channel.queue_declare(queue=f"intermediate_queue_{self.layer_id}", durable=False)
        self.channel.basic_qos(prefetch_count=self.prefetch_val)

        def _callback(ch, method, properties, body):
            try:
                metadata_message = json.loads(body.decode())
                if metadata_message == "STOP":
                    self.logger.log_info(f"[{self.name} Callback] Received STOP.")
                    self.input_q.put("STOP_FROM_PREVIOUS")
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    self.stop_evt.set()
                elif isinstance(metadata_message, dict) and "redis_key" in metadata_message:
                    if not self.data_transfer_handler:
                        self.logger.log_error(f"[{self.name}] Redis unavailable. NACK.")
                        ch.basic_nack(delivery_tag=method.delivery_tag)
                        return

                    actual_payload = self.data_transfer_handler.receive_data_from_metadata(
                        metadata_message, delete_after_retrieval=True
                    )
                    if actual_payload:
                        item_for_inference = {
                            "payload": actual_payload,
                            "delivery_tag": method.delivery_tag
                        }
                        self.input_q.put(item_for_inference)
                    else:
                        self.logger.log_error(f"[{self.name}] Failed to retrieve data from Redis.")
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                else:
                    self.logger.log_warning(f"[{self.name}] Unexpected format. NACK.")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception as e:
                self.logger.log_error(f"[{self.name}] Callback error: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        self.channel.basic_consume(queue=f"intermediate_queue_{self.layer_id}", on_message_callback=_callback, auto_ack=False)
        self.logger.log_info(f"[{self.name}] Waiting for messages on intermediate_queue_{self.layer_id}")

        while not self.stop_evt.is_set():
            try:
                self.connection.process_data_events(time_limit=self.process_events_timeout)
            except Exception as e:
                self.logger.log_error(f"[{self.name}] RabbitMQ error: {e}")
                self.stop_evt.set()
                break
