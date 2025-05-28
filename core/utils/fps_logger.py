import time
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FPSLogger:
    def __init__(self, layer_id: int, logger_obj, log_interval_seconds: int = 10, log_prefix: str = "Batch"):
        """Initializes the FPSLogger.

        Args:
            layer_id (int): The ID of the layer being logged.
            logger_obj: The logger instance to be used for logging.
            log_interval_seconds (int): The interval for logging cumulative FPS.
            log_prefix (str): A prefix string for log messages.
        """
        ...

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

