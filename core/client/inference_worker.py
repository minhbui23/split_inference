import threading
import torch
import time
import queue
import cv2
import uuid
from core.utils.fps_logger import FPSLogger

class BaseInferenceWorker(threading.Thread):
    """Base class for inference processing threads.

    Manages the model, computing device (CPU/CUDA), and the main loop for
    processing data from queues.

    Args:
        layer_id (int): The ID of the layer this worker belongs to.
        num_layers (int): The total number of layers in the pipeline.
        device (str): The computing device ('cpu' or 'cuda').
        model_obj: The initialized model object.
        predictor_obj: The Ultralytics predictor object.
        initial_params (dict): Initialization parameters from the server.
        input_q (queue.Queue): Queue for receiving data for inference.
        output_q (queue.Queue): Queue for holding results after inference.
        ack_trigger_q (queue.Queue): Queue for sending ACK signals.
        stop_evt (threading.Event): Event to safely stop the thread.
        logger: The logger instance.
        name (str, optional): The name of the thread. Defaults to None.
    """
    def __init__(self, layer_id, num_layers, device, 
                 model_obj, predictor_obj, initial_params,
                 input_q, output_q, ack_trigger_q, 
                 stop_evt, logger, name=None, metrics_logger=None):
        
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
        self.metrics_logger = metrics_logger
        self._initialize_params()

    def _initialize_params(self):
        """Initializes configuration parameters from the initial_params dictionary."""
        ...
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
        """Creates a log prefix based on the layer's position.

        Returns:
            str: The prefix string for logs (e.g., "Batch (L1 Video)").
        """
        ...
        if self.layer_id == 1:
            return "Batch (L1 Video)"
        if self.layer_id == self.num_layers and self.layer_id > 1:
            return "Batch of features (L-Last)"
        return f"Batch of features (L{self.layer_id} Middle)"


class FirstLayerWorker(BaseInferenceWorker):
    """Inference thread for the first layer client.

    Its main task is to read data from a source (video), process frames,
    perform the first part of the inference (`forward_head`), and put the
    result into the `output_q`.
    """
    def run(self):
        """The main loop of the thread, reads video and performs inference."""
        ...
        self.logger.log_info(f"[{self.name}] Starting First Layer")
        video_path = self.initial_params.get("data_source")
        if not video_path or not cv2.VideoCapture(video_path).isOpened():
            self.logger.log_error(f"[{self.name}] Video not found or unreadable: {video_path}")
            self.stop_evt.set()
            return

        cap = cv2.VideoCapture(video_path)
        frames_batch = []
        frame_count = 0
        save_layers = self.initial_params.get("save_layers")

        while not self.stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                self.logger.log_info(f"[{self.name}] End of video stream or error reading frame.")
                break

            frame_count += 1
            try:
                resized_frame = cv2.resize(frame, (self.img_width, self.img_height))
                tensor = torch.from_numpy(resized_frame).float().permute(2, 0, 1) / 255.0
                frames_batch.append(tensor)
            except Exception as e:
                self.logger.log_error(f"[{self.name}] Error on frame {frame_count}: {e}")
                continue

            if len(frames_batch) == self.batch_frame_size:
                self._process_batch(frames_batch, save_layers)
                frames_batch = []

        cap.release()
        self.fps_logger.log_overall_fps("L1: Video processing finished")

    def _process_batch(self, frames_batch, save_layers):
        """Processes a single batch of frames.

        Args:
            frames_batch (list): A list of frame tensors.
            save_layers (list): A list of layer indices whose output should be saved.
        """
        ...
        self.fps_logger.start_batch_timing()
        t1_start = time.time()
        try:
            batch = torch.stack(frames_batch).to(self.device)
            batch_size = batch.size(0)

            self.predictor_obj.setup_source(batch)
            input_tensor = self._prepare_input_tensor(batch)

            output = self.model_obj.forward_head(input_tensor, save_layers)
            self.fps_logger.end_batch_and_log_fps(batch_size)
            
            t1 = time.time() - t1_start

            output["layers_output"] = [
                t.cpu() if isinstance(t, torch.Tensor) else None
                for t in output["layers_output"]
            ]
            item = {
                "payload": output,
                "metrics": {
                    "t1": t1,
                    "batch_id" : str(uuid.uuid4()),
                },
                "l1_inference_timestamp": time.time()
            }
            if self.layer_id < self.num_layers:
                self.output_q.put((item, f"intermediate_queue_{self.layer_id + 1}"))
        except Exception as e:
            self.logger.log_error(f"[{self.name}] Error processing batch: {e}")
            self.fps_logger._current_batch_start_time = None

    def _prepare_input_tensor(self, batch):
        """Prepares the input tensor for the model from the predictor object.

        Args:
            batch (torch.Tensor): The input data batch.

        Returns:
            torch.Tensor: The tensor ready to be fed into the model.
        """
        ...
        for data in self.predictor_obj.dataset:
            if isinstance(data, tuple) and len(data) > 1:
                return data[1].to(self.device) if isinstance(data[1], torch.Tensor) else data[1]
        return batch.to(self.device)


class MiddleLayerWorker(BaseInferenceWorker):
    def run(self):
        self.logger.log_info(f"[{self.name}] No logic needed for middle layer. Exiting.")
        self.stop_evt.set()


class LastLayerWorker(BaseInferenceWorker):
    """Inference thread for the last layer client.

    Its main task is to get data from the `input_q`, perform the final
    part of the inference (`forward_tail`), and trigger an ACK.
    """
    def run(self):
        """The main loop of the thread, receives and processes features."""
        ...
        self.logger.log_info(f"[{self.name}] Starting Last Layer")
        is_last_layer = True

        while not self.stop_evt.is_set():
            try:
                item = self.input_q.get(timeout=0.5)

                self._process_payload(item)
                self.input_q.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error(f"[{self.name}] Unexpected error: {e}")
                time.sleep(0.1)

        self.fps_logger.log_overall_fps("L-Last: Feature processing finished")

    def _process_payload(self, item): # Sửa tham số
        # Trích xuất dữ liệu từ item
        payload = item.get("payload")
        delivery_tag = item.get("delivery_tag")
        metrics = item.get("metrics", {})
        
        # Đo t4 và q2
        put_time = item.get('put_to_l2_input_q_timestamp')
        if put_time:
            metrics['t4'] = time.time() - put_time

        # Đo t5
        t5_start = time.time()
        ack_status = "failure"
        requeue = False
        self.fps_logger.start_batch_timing()
        try:
            payload["layers_output"] = [
                t.to(self.device) if isinstance(t, torch.Tensor) else None
                for t in payload["layers_output"]
            ]
            self.model_obj.forward_tail(payload)
            ack_status = "success"
        except Exception as e:
            self.logger.log_error(f"[{self.name}] Error with payload {delivery_tag}: {e}")
        finally:
            metrics['t5'] = time.time() - t5_start
            
            # GHI LOG CUỐI CÙNG
            if self.metrics_logger:
                self.metrics_logger.log(metrics)

            self.fps_logger.end_batch_and_log_fps(self.batch_frame_size)
            self.ack_trigger_q.put({
                "delivery_tag": delivery_tag,
                "status": ack_status,
                "requeue": requeue
            })
