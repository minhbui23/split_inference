
import time
import pika
from requests.auth import HTTPBasicAuth
import requests
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FPSLogger:
    def __init__(self, layer_id: int, logger_obj, log_interval_seconds: int = 10, log_prefix: str = "Batch"):

        self.layer_id = layer_id
        self.logger = logger_obj
        self.log_interval_seconds = log_interval_seconds
        self.log_prefix = log_prefix

        self.total_frames_processed = 0
        self.total_inference_time = 0.0  
        self.last_cumulative_log_time = time.time()
        
        self._current_batch_start_time = None # Time to track the start of the current batch processing

    def start_batch_timing(self):
        self._current_batch_start_time = time.time()

    def end_batch_and_log_fps(self, frames_in_current_batch: int):
        """
        Time to end batch processing and log FPS.
        Args:
            frames_in_current_batch (int): Number of frames processed in the current batch.
        """
        if self._current_batch_start_time is None:
            if self.logger: 
                self.logger.log_warning(f"[FPSLogger L{self.layer_id}] end_batch_and_log_fps() được gọi mà không có start_batch_timing(). Bỏ qua log FPS cho batch này.")
            return

        batch_processing_time = time.time() - self._current_batch_start_time
        self._current_batch_start_time = None  

        self.total_frames_processed += frames_in_current_batch
        self.total_inference_time += batch_processing_time

        if self.logger: 
            if batch_processing_time > 0:
                batch_fps = frames_in_current_batch / batch_processing_time
                self.logger.log_info(
                    f"[InferenceThread L{self.layer_id}] {self.log_prefix}: {frames_in_current_batch} frames processed in {batch_processing_time:.4f}s (FPS: {batch_fps:.2f})."
                )
            else:
                self.logger.log_info(
                    f"[InferenceThread L{self.layer_id}] {self.log_prefix}: {frames_in_current_batch} frames processed very quickly."
                )

        current_time = time.time()
        if self.total_frames_processed > 0 and \
           (current_time - self.last_cumulative_log_time >= self.log_interval_seconds) and \
           self.logger: 
            if self.total_inference_time > 0:
                cumulative_avg_fps = self.total_frames_processed / self.total_inference_time
                self.logger.log_info(
                    f"[InferenceThread L{self.layer_id}] CUMULATIVE: {self.total_frames_processed} frames in {self.total_inference_time:.4f}s (Avg FPS: {cumulative_avg_fps:.2f})."
                )
            self.last_cumulative_log_time = current_time


    
    def log_overall_fps(self, process_description: str = "Processing finished"):
        if not self.logger: 
            print(f"[FPSLogger L{self.layer_id}] Logger not available for final FPS log.")
            return

        if self.total_frames_processed == 0:
            self.logger.log_info(f"[InferenceThread L{self.layer_id}] {process_description}. No frames were processed or timed.")
            return
            
        if self.total_inference_time > 0:
            overall_avg_fps = self.total_frames_processed / self.total_inference_time
            self.logger.log_info(
                f"[InferenceThread L{self.layer_id}] {process_description}. OVERALL: {self.total_frames_processed} frames in {self.total_inference_time:.4f}s (Avg FPS: {overall_avg_fps:.2f})."
            )
        else:
            self.logger.log_info(
                f"[InferenceThread L{self.layer_id}] {process_description}. {self.total_frames_processed} frames processed, but total inference time was zero or not recorded."
            )


def delete_old_queues(address, username, password, virtual_host):
    encoded_vhost = requests.utils.quote(virtual_host, safe='')
    mgmt_url = f'http://{address}:15672/api/queues/{encoded_vhost}'
    logging.info(f"Cleanup: Get queue list from API: {mgmt_url}")

    try:
        response = requests.get(mgmt_url, auth=HTTPBasicAuth(username, password), timeout=10) 
        response.raise_for_status() 
        queues = response.json()
        logging.info(f"Cleanup: Successfully retrieved {len(queues)} queue(s) via API.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Cleanup: Error when connecting or getting queue list from Management API: {e}")
        return False 

    connection = None 
    deleted_count = 0
    try:
        credentials = pika.PlainCredentials(username, password)
        params = pika.ConnectionParameters(
            host=address,
            port=5672,
            virtual_host=virtual_host,
            credentials=credentials,
            heartbeat=30,
            blocked_connection_timeout=30
        )
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        logging.info("Cleanup: Connected AMQP for cleanup.")

        app_queue_prefixes = ("reply_", "intermediate_queue_", "result", "rpc_queue")

        for queue_info in queues:
            queue_name = queue_info.get('name')
            if not queue_name:
                continue

            if queue_name.startswith(app_queue_prefixes):
                logging.info(f"Cleanup: Prepare to delete application queue: {queue_name}")
                try:
                    channel.queue_delete(queue=queue_name)
                    deleted_count += 1
                    logging.info(f"Cleanup: Delete queue success: {queue_name}")
                except pika.exceptions.ChannelClosedByBroker as e:
                
                    logging.warning(f"Cleanup: Unable to delete queue {queue_name}. Broker closed channel: {e}. Queue may be in use or exclusive.")

                    if connection.is_open and (not channel or channel.is_closed):
                        try:
                            channel = connection.channel()
                            logging.info("Cleanup: Reopened channel after deletion error.")
                        except Exception as reopen_e:
                            logging.error(f"Cleanup: Unable to reopen channel: {reopen_e}. Stop cleaning.")
                            break # Thoát vòng lặp nếu không mở lại được channel
                except Exception as e:
                    logging.error(f"Cleanup: Unknown error while deleting queue {queue_name}: {e}")
                    if connection.is_open and (not channel or channel.is_closed):
                        try:
                           channel = connection.channel()
                           logging.info("Cleanup: Reopened channel after unknown error.")
                        except Exception as reopen_e:
                           logging.error(f"Cleanup: Unable to reopen channel: {reopen_e}. Stop cleaning.")
                           break

        logging.info(f"Cleanup: Finished cleaning. Number of queues deleted: {deleted_count}")
        return True

    except pika.exceptions.AMQPConnectionError as e:
        logging.error(f"Cleanup: AMQP connection error to clean up: {e}")
        return False
    except Exception as e:
        logging.error(f"Cleanup: Unknown error during AMQP cleanup: {e}")
        import traceback
        traceback.print_exc() 
        return False
    finally:

        if connection and connection.is_open:
            connection.close()
            logging.info("Cleanup: AMPQ connection closed.")

