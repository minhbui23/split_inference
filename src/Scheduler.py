import pickle
import time
import torch
import cv2
import threading
import traceback
import queue # Thêm queue
import functools # Thêm functools

from src.Model import SplitDetectionPredictor
import src.Log
import numpy as np

class Scheduler:
    # Thêm connection vào __init__
    def __init__(self, client_id, layer_id, connection, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        # Lưu connection để dùng add_callback_threadsafe
        self.connection = connection
        self.channel = channel
        self.device = device
        self.model = None
        self.predictor = None
        self.config = None # Thêm để lưu config
        self.num_layers = 0 # Thêm để lưu số layers

        # --- Biến cho mô hình 2 luồng ---
        self.is_consuming = False # Cờ đánh dấu consumer I/O đang chạy
        self.consumer_tag = None
        self.task_queue = None # Hàng đợi chuyển việc từ I/O -> Inference
        self.inference_thread = None # Thread xử lý inference
        self._stop_event = threading.Event() # Event để dừng các luồng

        # --- Biến cho Layer cuối (ghi video) ---
        self.video_writer = None
        self.video_width = None
        self.video_height = None
        self.frame_count = 0

        # --- Biến cho Layer 1 ---
        self.layer1_thread = None # Thread xử lý của Layer 1


    def send_next_layer(self, data, is_metadata=False):
        """Gửi dữ liệu đến layer tiếp theo."""
        # Hàm này có thể được gọi từ cả luồng Layer 1 hoặc Luồng Inference
        # Cần đảm bảo channel còn hoạt động
        if not self.channel or self.channel.is_closed:
            # Kiểm tra thêm connection vì channel có thể bị đóng trước connection
            if not self.connection or self.connection.is_closed:
                 src.Log.print_with_color(f"[Lyr {self.layer_id}] Cannot send, connection is closed.", "red")
                 return False # Không thể gửi
            else:
                 # Thử lấy lại channel nếu connection còn mở (ít khả năng thành công nếu channel đã đóng)
                 try:
                      self.channel = self.connection.channel()
                      src.Log.print_with_color(f"[Lyr {self.layer_id}] Warning: Channel was closed, attempting to reopen.", "yellow")
                 except Exception as e:
                      src.Log.print_with_color(f"[Lyr {self.layer_id}] Cannot send, failed to reopen channel: {e}", "red")
                      return False

        # Xác định queue đích
        next_layer_queue = ""
        if self.layer_id < self.num_layers: # Chỉ gửi nếu không phải layer cuối
            next_layer_queue = f"intermediate_queue_{self.layer_id}"
        else:
             src.Log.print_with_color(f"[Lyr {self.layer_id}] Is last layer, not sending forward.", "grey")
             return True # Không gửi nhưng coi như thành công

        try:
            # Khai báo queue (có thể không cần thiết nếu server/client trước đã khai báo)
            # self.channel.queue_declare(next_layer_queue, durable=False)

            action = "METADATA" if is_metadata else "OUTPUT"
            message = pickle.dumps({
                "action": action,
                "data": data,
                "source_layer": self.layer_id # Thêm thông tin layer nguồn
            })
            # Gửi message
            self.channel.basic_publish(
                exchange='',
                routing_key=next_layer_queue,
                body=message
            )
            # src.Log.print_with_color(f"[Lyr {self.layer_id} >>>] Sent {action} to {next_layer_queue}", "magenta")
            return True
        except Exception as e:
            src.Log.print_with_color(f"[Lyr {self.layer_id}] Error sending to {next_layer_queue}: {e}", "red")
            # In stack trace để debug lỗi gửi
            # traceback.print_exc()
            return False

    # --- Hàm Callback cho Pika I/O Thread (Layer > 1) ---
    def _on_message_received_io(self, ch, method, properties, body):
        """Callback chạy trong luồng I/O của Pika khi nhận message."""
        # src.Log.print_with_color(f"[Lyr {self.layer_id} IO <<<] Received message tag={method.delivery_tag}", "grey")
        try:
            # Chỉ deserialize và đưa vào hàng đợi xử lý
            message_data = pickle.loads(body)
            # Đưa cả dữ liệu và delivery_tag vào queue cho inference thread
            self.task_queue.put((message_data, method.delivery_tag))
        except pickle.UnpicklingError as e:
             src.Log.print_with_color(f"[Lyr {self.layer_id} IO] Error deserializing message: {e}", "red")
             # Không thể xử lý -> Nack message này
             self._schedule_nack(method.delivery_tag)
        except Exception as e:
             src.Log.print_with_color(f"[Lyr {self.layer_id} IO] Unexpected error in IO callback: {e}", "red")
             traceback.print_exc()
             # Cũng nên Nack nếu có lỗi bất ngờ
             self._schedule_nack(method.delivery_tag)

    # --- Hàm cho Inference Thread (Layer > 1) ---
    def _inference_worker(self):
        """Hàm chạy trong luồng inference riêng biệt."""
        thread_name = threading.current_thread().name
        src.Log.print_with_color(f"Thread {thread_name}: [Lyr {self.layer_id} Infer] Inference worker started.", "cyan")

        while not self._stop_event.is_set():
            task = None
            try:
                # Lấy task từ queue, timeout để kiểm tra cờ stop định kỳ
                task = self.task_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                continue # Không có task, tiếp tục vòng lặp để kiểm tra cờ stop

            if task is None: # Sentinel để dừng luồng
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr {self.layer_id} Infer] Received stop signal.", "yellow")
                break

            # Giải nén task
            message_data, delivery_tag = task
            action = message_data.get("action")
            data = message_data.get("data")
            source_layer = message_data.get("source_layer", "Unknown") # Lấy layer nguồn

            # src.Log.print_with_color(f"Thread {thread_name}: [Lyr {self.layer_id} Infer] Processing message {delivery_tag} from layer {source_layer}", "green")
            processing_success = False
            try:
                start_time = time.time()
                if action == "METADATA":
                    processing_success = self._handle_metadata(data)

                elif action == "OUTPUT":
                     # Chỉ xử lý nếu là layer cuối hoặc layer giữa
                     if self.layer_id == self.num_layers:
                         processing_success = self._process_final_output(data)
                     else:
                         processing_success = self._process_intermediate_output(data)
                else:
                     src.Log.print_with_color(f"Thread {thread_name}: [Lyr {self.layer_id} Infer] Unknown action '{action}' for tag {delivery_tag}", "yellow")
                     processing_success = True # Coi như thành công để Ack message không xác định

                stop_time = time.time()
                # src.Log.print_with_color(f"Thread {thread_name}: [Lyr {self.layer_id} Infer] Processing time for tag {delivery_tag}: {stop_time - start_time:.4f}s", "grey")

            except Exception as e:
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr {self.layer_id} Infer] Error processing message tag {delivery_tag}: {e}", "red")
                traceback.print_exc()
                processing_success = False # Đánh dấu xử lý lỗi

            # === Gửi Ack/Nack từ luồng Inference ===
            if action == "METADATA" and data.get("status") == "EOS":
                 src.Log.print_with_color(f"Thread {thread_name}: [Lyr {self.layer_id} Infer] EOS received, stopping consumer.", "yellow")
                 # Không cần ack/nack message EOS, chỉ cần dừng consumer
                 # Việc dừng sẽ được xử lý trong luồng chính hoặc stop_consuming
                 self._stop_event.set() # Set cờ dừng cho chính luồng này và luồng I/O
                 # Có thể cần cơ chế khác để báo cho luồng I/O dừng start_consuming()
                 # Hoặc dựa vào việc đóng connection từ client.py
            elif processing_success:
                # src.Log.print_with_color(f"Thread {thread_name}: [Lyr {self.layer_id} Infer] Scheduling Ack for tag {delivery_tag}", "grey")
                self._schedule_ack(delivery_tag)
            else:
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr {self.layer_id} Infer] Scheduling Nack for tag {delivery_tag}", "yellow")
                self._schedule_nack(delivery_tag)

            # Đánh dấu task đã hoàn thành trong queue (quan trọng nếu dùng queue.JoinableQueue)
            self.task_queue.task_done()

        src.Log.print_with_color(f"Thread {thread_name}: [Lyr {self.layer_id} Infer] Inference worker stopped.", "cyan")

    # --- Các hàm xử lý cụ thể (chạy trong luồng Inference) ---
    def _handle_metadata(self, metadata):
        """Xử lý metadata nhận được."""
        if "fps" in metadata: # Metadata khởi tạo video (cho layer cuối)
            if self.layer_id == self.num_layers:
                fps = metadata['fps']
                self.video_width = metadata['width']
                self.video_height = metadata['height']
                # Chỉ khởi tạo nếu save_output = True
                if self.config and self.config.get("save_output", False):
                    output_filename = f"output_client_{self.client_id}_layer_{self.layer_id}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (self.video_width, self.video_height))
                    src.Log.print_with_color(f"[Lyr {self.layer_id} Infer] Output video writer initialized: {output_filename}", "yellow")
                else:
                    src.Log.print_with_color(f"[Lyr {self.layer_id} Infer] Received video metadata but save_output=False, writer not created.", "grey")
            else:
                # Layer giữa chuyển tiếp metadata
                return self.send_next_layer(metadata, is_metadata=True)
        elif metadata.get("status") == "EOS":
             src.Log.print_with_color(f"[Lyr {self.layer_id} Infer] End Of Stream signal received.", "yellow")
             # Nếu không phải layer cuối, chuyển tiếp EOS
             if self.layer_id < self.num_layers:
                  return self.send_next_layer(metadata, is_metadata=True)
             # Layer cuối không cần làm gì thêm ở đây, luồng inference sẽ dừng sau khi xử lý xong task này
        else:
             src.Log.print_with_color(f"[Lyr {self.layer_id} Infer] Received unknown metadata: {metadata}", "grey")
             # Chuyển tiếp nếu là layer giữa
             if self.layer_id < self.num_layers:
                  return self.send_next_layer(metadata, is_metadata=True)

        return True # Xử lý metadata thành công

    def _process_intermediate_output(self, data):
        """Xử lý output trung gian (cho layer giữa)."""
        # Chuyển tensor về device nếu cần
        y_prev = self._move_data_to_device(data)

        # Xử lý bằng model middle/tail
        with torch.no_grad():
            # Giả sử vẫn dùng forward_tail cho phần còn lại
            y_next = self.model.forward_tail(y_prev)

        # Chuyển output về CPU trước khi gửi (hoặc giữ trên GPU nếu layer sau cùng device?)
        # Tạm thời chuyển về CPU để an toàn
        y_next_cpu = self._move_data_to_cpu(y_next)

        # Gửi kết quả cho layer tiếp theo
        return self.send_next_layer(y_next_cpu, is_metadata=False)

    def _process_final_output(self, data):
        """Xử lý output ở layer cuối."""
        # Chuyển input tensor về đúng device
        y_prev = self._move_data_to_device(data)

        # === Xử lý bằng model tail ===
        with torch.no_grad():
            predictions = self.model.forward_tail(y_prev)

        # === Hậu xử lý (nếu cần ghi video) ===
        if self.predictor and self.video_writer and self.config.get("save_output", False):
            try:
                # Kiểm tra xem data có chứa đủ thông tin cho postprocess không
                # Code gốc yêu cầu y["img"] và y["orig_imgs"] được gửi từ layer 1 nếu save_output=True
                # Nếu cấu trúc data thay đổi, cần điều chỉnh ở đây
                img_tensor = y_prev.get("img") # Lấy tensor ảnh đã pre-process
                orig_imgs_data = y_prev.get("orig_imgs") # Lấy ảnh gốc (có thể là list numpy hoặc tensor)

                if img_tensor is not None and orig_imgs_data is not None:
                     # Chuyển prediction và ảnh về CPU để xử lý NMS và vẽ
                     # Đảm bảo orig_imgs là list numpy nếu predictor yêu cầu
                     if isinstance(orig_imgs_data, torch.Tensor):
                          orig_imgs_np = ops.convert_torch2numpy_batch(orig_imgs_data.cpu())
                     else: # Giả sử là list numpy sẵn
                          orig_imgs_np = orig_imgs_data

                     # Đường dẫn giả lập vì không có trong dữ liệu truyền qua
                     dummy_paths = [f"frame_{self.frame_count + i}" for i in range(len(orig_imgs_np))]

                     processed_preds = self.predictor.postprocess(predictions.cpu(), img_tensor.cpu(), orig_imgs_np, path=dummy_paths)

                     for i, res in enumerate(processed_preds):
                          annotated_frame = res.plot() # Vẽ lên ảnh gốc
                          # Resize về kích thước video output và ghi
                          self.video_writer.write(cv2.resize(annotated_frame, (self.video_width, self.video_height)))
                          self.frame_count += 1 # Tăng frame count ở đây
                else:
                     src.Log.print_with_color(f"[Lyr {self.layer_id} Infer] Missing 'img' or 'orig_imgs' in data for visualization. Keys: {list(y_prev.keys())}", "yellow")

            except Exception as viz_e:
                src.Log.print_with_color(f"[Lyr {self.layer_id} Infer] Error during visualization/saving: {viz_e}", "red")
                # traceback.print_exc() # Bật nếu cần debug sâu
                # Không nên trả về False ở đây nếu chỉ lỗi vẽ, vẫn ack message đã xử lý inference

        return True # Inference và xử lý (nếu có) hoàn tất

    def _move_data_to_device(self, data):
        """Chuyển dữ liệu (dict tensor) sang device của client."""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self.device)
                elif isinstance(v, list):
                    # Giả định list chứa tensor
                    data[k] = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in v]
        elif isinstance(data, torch.Tensor): # Trường hợp dữ liệu chỉ là tensor
             data = data.to(self.device)
        return data

    def _move_data_to_cpu(self, data):
         """Chuyển dữ liệu (dict tensor) về CPU."""
         if isinstance(data, dict):
              cpu_data = {}
              for k, v in data.items():
                   if isinstance(v, torch.Tensor):
                        cpu_data[k] = v.cpu()
                   elif isinstance(v, list):
                        # Giả định list chứa tensor
                        cpu_data[k] = [t.cpu() if isinstance(t, torch.Tensor) else t for t in v]
                   else:
                        cpu_data[k] = v # Giữ nguyên các loại dữ liệu khác
              return cpu_data
         elif isinstance(data, torch.Tensor):
              return data.cpu()
         return data # Giữ nguyên nếu không phải dict hoặc tensor

    # --- Hàm tương tác với Pika từ Thread khác (Thread-safe) ---
    def _schedule_ack(self, delivery_tag):
        """Lên lịch gửi Ack từ luồng Pika I/O."""
        if self.connection and self.connection.is_open:
            cb = functools.partial(self._ack_message, delivery_tag=delivery_tag)
            self.connection.add_callback_threadsafe(cb)

    def _ack_message(self, delivery_tag):
        """Hàm thực sự gửi Ack (chạy trong luồng Pika I/O)."""
        if self.channel and self.channel.is_open:
            try:
                self.channel.basic_ack(delivery_tag=delivery_tag)
                # src.Log.print_with_color(f"[Lyr {self.layer_id} IO] Ack sent for tag {delivery_tag}", "grey")
            except Exception as e:
                src.Log.print_with_color(f"[Lyr {self.layer_id} IO] Error sending Ack for tag {delivery_tag}: {e}", "red")
        else:
             src.Log.print_with_color(f"[Lyr {self.layer_id} IO] Cannot Ack tag {delivery_tag}, channel/connection closed.", "yellow")


    def _schedule_nack(self, delivery_tag):
        """Lên lịch gửi Nack từ luồng Pika I/O."""
        if self.connection and self.connection.is_open:
            cb = functools.partial(self._nack_message, delivery_tag=delivery_tag)
            self.connection.add_callback_threadsafe(cb)

    def _nack_message(self, delivery_tag):
        """Hàm thực sự gửi Nack (chạy trong luồng Pika I/O)."""
        if self.channel and self.channel.is_open:
            try:
                self.channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
                src.Log.print_with_color(f"[Lyr {self.layer_id} IO] Nack sent for tag {delivery_tag}", "yellow")
            except Exception as e:
                src.Log.print_with_color(f"[Lyr {self.layer_id} IO] Error sending Nack for tag {delivery_tag}: {e}", "red")
        else:
            src.Log.print_with_color(f"[Lyr {self.layer_id} IO] Cannot Nack tag {delivery_tag}, channel/connection closed.", "yellow")


    # --- Hàm xử lý Layer 1 (Vẫn giữ nguyên chạy trong thread riêng) ---
    def _handle_first_layer(self):
        """Logic xử lý Layer 1 (Dựa trên code gốc) + Thêm Try/Catch."""
        thread_name = threading.current_thread().name
        src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Starting video processing...", "green")

        if not self.model or not self.config:
            src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Error: Model or Config not set before starting processing.", "red")
            return

        # Lưu cấu hình để dùng trong send_next_layer
        self.num_layers = self.config.get("num_layers", 1)

        input_images = []
        orig_image_batch_np = [] # Lưu batch ảnh gốc dạng numpy để gửi đi nếu cần
        batch_size = self.config.get("batch_size", 1)
        save_layers_flags = self.config.get("save_layers", []) # Các layer trung gian cần lưu output
        save_output_global = self.config.get("save_output", False) # Cờ lưu output cuối cùng/vẽ vời
        video_path = self.config.get("data", "video.mp4") # Lấy video path từ config

        predictor = None
        cap = None
        frame_read_count = 0
        total_inference_time = 0

        try:
            # --- Block 1: Khởi tạo Predictor và Model Setup ---
            src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Initializing predictor...", "green")
            try:
                # Sử dụng model đã được gán cho self.model
                # Predictor chỉ cần thiết nếu save_output=True (để preprocess/postprocess)
                # Tuy nhiên, preprocess vẫn cần dùng
                predictor = SplitDetectionPredictor(model=self.model, overrides={"imgsz": 640, "device": self.device})
                # Không gán self.model cho predictor.model vì self.model là SplitDetectionModel
                # predictor.model = self.model # Bỏ dòng này
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Predictor initialized.", "green")
            except Exception as e:
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] FAILED at Predictor Initialization: {e}", "red")
                traceback.print_exc()
                return

            src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Setting model to eval mode and device...", "green")
            try:
                self.model.eval()
                self.model.to(self.device)
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Model ready on device {self.device}.", "green")
            except Exception as e:
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] FAILED at Model eval/to_device: {e}", "red")
                traceback.print_exc()
                return

            # --- Block 2: Mở Video ---
            src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Opening video capture: {video_path}", "green")
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] FAILED: Could not open video {video_path}", "red")
                    return
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Video capture opened.", "green")
            except Exception as e:
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] FAILED at cv2.VideoCapture: {e}", "red")
                traceback.print_exc()
                return

            # --- Block 3: Lấy và Gửi Metadata ---
            src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Getting video metadata...", "green")
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Video Info: {width}x{height} @ {fps:.2f} FPS", "green")
                metadata = {"fps": fps, "width": width, "height": height}
                if not self.send_next_layer(metadata, is_metadata=True):
                     src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] FAILED to send initial metadata. Stopping.", "red")
                     return # Dừng nếu không gửi được metadata ban đầu
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Metadata sent.", "green")
            except Exception as e:
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] FAILED at Getting/Sending Metadata: {e}", "red")
                traceback.print_exc()
                return # Dừng nếu lỗi metadata

            # --- Block 4: Vòng lặp xử lý Frame ---
            src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Starting frame processing loop...", "green")

            while not self._stop_event.is_set(): # Kiểm tra cờ dừng
                frame_start_time = time.time()
                current_frame_num = frame_read_count + 1

                # Đọc frame
                ret, frame = False, None
                try:
                    ret, frame = cap.read()
                except Exception as e:
                    src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] FAILED at cap.read() frame {current_frame_num}: {e}", "red")
                    traceback.print_exc()
                    break # Thoát vòng lặp nếu lỗi đọc frame

                if not ret:
                    src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] End of video detected.", "yellow")
                    # Gửi tín hiệu kết thúc
                    if not self.send_next_layer({"status": "EOS"}, is_metadata=True):
                         src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Warning: Failed to send EOS signal.", "yellow")
                    break # Kết thúc video

                frame_read_count += 1
                orig_image_batch_np.append(frame.copy()) # Lưu ảnh gốc dạng numpy

                # Tiền xử lý cơ bản (có thể dùng predictor.preprocess sau)
                try:
                    frame_resized = cv2.resize(frame, (640, 640)) # Giả sử imgsz=640
                    # Chuyển sang tensor float, permute, normalize
                    tensor = torch.from_numpy(frame_resized).to(self.device).float()
                    tensor = tensor.permute(2, 0, 1).contiguous() / 255.0
                    input_images.append(tensor)
                except Exception as e:
                    src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] FAILED at basic preprocessing frame {frame_read_count}: {e}", "red")
                    traceback.print_exc()
                    orig_image_batch_np.pop() # Xóa ảnh gốc tương ứng nếu tiền xử lý lỗi
                    continue # Bỏ qua frame lỗi

                # Xử lý khi đủ batch
                if len(input_images) == batch_size:
                    batch_start_time = time.time()
                    batch_idx = (frame_read_count // batch_size) if batch_size > 0 else frame_read_count

                    try:
                        # Stack batch tensor ảnh đã tiền xử lý
                        batch_tensor_preprocessed = torch.stack(input_images).to(self.device)
                        input_images = [] # Reset ngay

                        # === Chạy Model Head ===
                        with torch.no_grad():
                             y = self.model.forward_head(batch_tensor_preprocessed, output_from=save_layers_flags)
                             # y là dict {"layers_output": [...], "last_layer_idx": ...}

                        # === Chuẩn bị dữ liệu gửi đi ===
                        # Chuyển output tensor về CPU
                        y_cpu = self._move_data_to_cpu(y) # Hàm này xử lý dict

                        # Thêm ảnh gốc và ảnh đã pre-process vào data nếu cần cho layer cuối vẽ vời
                        if save_output_global:
                            y_cpu["img"] = batch_tensor_preprocessed.cpu() # Ảnh đã tiền xử lý (tensor)
                            # Ảnh gốc (list numpy)
                            # Cần đảm bảo rằng orig_image_batch_np có cùng số lượng phần tử
                            if len(orig_image_batch_np) == batch_size:
                                y_cpu["orig_imgs"] = orig_image_batch_np
                            else:
                                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Mismatch between preprocessed ({batch_size}) and original images ({len(orig_image_batch_np)}) count.", "yellow")
                                # Có thể bỏ qua việc gửi ảnh gốc nếu lỗi
                        orig_image_batch_np = [] # Reset batch ảnh gốc

                        # === Gửi kết quả ===
                        if not self.send_next_layer(y_cpu, is_metadata=False):
                             src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] FAILED to send batch {batch_idx}. Stopping.", "red")
                             # Có thể break hoặc return tùy vào mức độ nghiêm trọng
                             break # Thoát vòng lặp nếu không gửi được batch

                        batch_end_time = time.time()
                        total_inference_time += (batch_end_time - batch_start_time)

                    except Exception as e:
                        src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] FAILED during processing batch ending with frame {frame_read_count}: {e}", "red")
                        traceback.print_exc()
                        input_images = [] # Đảm bảo reset batch nếu lỗi
                        orig_image_batch_np = [] # Reset cả batch ảnh gốc

        # --- Block 5: Kết thúc và Dọn dẹp ---
        except Exception as e:
            src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] UNEXPECTED Error during frame loop: {e}", "red")
            traceback.print_exc()
        finally:
            if cap and cap.isOpened():
                cap.release()
                src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Video capture released.", "green")
            src.Log.print_with_color(f"Thread {thread_name}: [Lyr 1] Finished processing. Total frames read: {frame_read_count}. Approx total inference time: {total_inference_time:.2f}s", "green")
            # Không cần gửi EOS lần nữa vì đã gửi trong loop hoặc khi break

    # --- Thiết lập và Quản lý luồng ---
    def setup_consumer(self, model, config):
        """Thiết lập consumer hoặc producer dựa trên layer_id."""
        if not self.channel or self.channel.is_closed:
             src.Log.print_with_color(f"[Lyr {self.layer_id}] Cannot setup consumer, channel is closed.", "red")
             return False

        self.model = model
        self.config = config # Lưu config
        self.num_layers = config.get("num_layers", 1) # Lưu số layers
        save_output_global = config.get("save_output", False) # Lưu cờ save output

        src.Log.print_with_color(f"[Lyr {self.layer_id}] Setting up role...", "cyan")

        if self.layer_id == 1:
            # --- Layer 1: Producer ---
            src.Log.print_with_color(f"[Lyr 1] Setup complete (Producer role). Starting processing thread...", "cyan")
            self._stop_event.clear() # Đảm bảo cờ stop được reset
            # Chạy _handle_first_layer trong thread riêng
            self.layer1_thread = threading.Thread(target=self._handle_first_layer, daemon=True)
            self.layer1_thread.start()
            return True # Trả về True để client.py biết là thành công

        else:
            # --- Layer > 1: Consumer (I/O + Inference Threads) ---
            src.Log.print_with_color(f"[Lyr {self.layer_id}] Setup complete (Consumer role).", "cyan")
            input_queue_name = f"intermediate_queue_{self.layer_id - 1}"

            # Thiết lập model eval/device
            try:
                self.model.eval()
                self.model.to(self.device)
                src.Log.print_with_color(f"[Lyr {self.layer_id}] Model ready on device {self.device}.", "green")
            except Exception as e:
                src.Log.print_with_color(f"[Lyr {self.layer_id}] FAILED at Model eval/to_device: {e}", "red")
                return False

            # Khởi tạo predictor cho layer cuối nếu cần vẽ vời
            if self.layer_id == self.num_layers and save_output_global:
                try:
                    # Predictor cần model gốc để lấy tên class, etc.
                    # Lấy model gốc từ SplitDetectionModel nếu có thể, hoặc tạo mới
                    base_model_name = self.config.get("model_name", "yolov8n") # Lấy tên model gốc
                    base_model = YOLO(f"{base_model_name}.pt").model # Tải lại model gốc
                    self.predictor = SplitDetectionPredictor(model=base_model, overrides={"imgsz": 640, "device": self.device})
                    # Không gán self.model (split model) cho predictor
                    src.Log.print_with_color(f"[Lyr {self.layer_id}] Predictor for postprocessing initialized.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"[Lyr {self.layer_id}] FAILED at Predictor Initialization: {e}", "red")
                    # Có thể vẫn tiếp tục mà không có predictor nếu chỉ lỗi này?
                    # return False # Hoặc dừng nếu predictor là bắt buộc

            try:
                # Khởi tạo hàng đợi task giữa 2 luồng
                self.task_queue = queue.Queue()
                self._stop_event.clear() # Reset cờ dừng

                # Khởi động luồng Inference Worker
                self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True, name=f"InferWorker-L{self.layer_id}")
                self.inference_thread.start()
                src.Log.print_with_color(f"[Lyr {self.layer_id}] Inference worker thread started.", "cyan")

                # Khai báo queue input và QoS cho luồng I/O
                self.channel.queue_declare(input_queue_name, durable=False)
                prefetch_count = self.config.get("prefetch_count", 10) # Lấy từ config hoặc mặc định
                self.channel.basic_qos(prefetch_count=prefetch_count)
                src.Log.print_with_color(f"[Lyr {self.layer_id}] Input queue '{input_queue_name}' declared, QoS prefetch={prefetch_count}.", "cyan")

                # Đăng ký consumer (luồng I/O sẽ chạy callback _on_message_received_io)
                self.consumer_tag = self.channel.basic_consume(
                    queue=input_queue_name,
                    on_message_callback=self._on_message_received_io,
                    auto_ack=False # Quan trọng: Tắt auto_ack
                )
                self.is_consuming = True # Đánh dấu consumer I/O sẵn sàng
                src.Log.print_with_color(f"[Lyr {self.layer_id}] Consumer started on queue {input_queue_name} with tag {self.consumer_tag}. Waiting for messages...", "cyan")
                return True # Setup thành công
            except Exception as e:
                src.Log.print_with_color(f"[Lyr {self.layer_id}] Error setting up consumer/inference worker: {e}", "red")
                traceback.print_exc()
                # Dọn dẹp nếu lỗi giữa chừng
                self._stop_event.set()
                if self.inference_thread and self.inference_thread.is_alive():
                    self.task_queue.put(None) # Báo dừng thread
                    self.inference_thread.join(timeout=1.0)
                return False # Setup lỗi


    def start_processing_messages(self):
         """Bắt đầu vòng lặp tiêu thụ message (cho luồng I/O nếu là consumer)."""
         if self.layer_id == 1:
              # Layer 1 đã chạy thread xử lý riêng, trả về handle thread đó
              src.Log.print_with_color(f"[Lyr 1] Producer role, processing runs in background thread.", "yellow")
              return self.layer1_thread # Trả về thread để client.py có thể join
         elif self.is_consuming:
              # Layer > 1: Bắt đầu vòng lặp blocking của Pika trong luồng hiện tại
              # Luồng này sẽ trở thành luồng I/O
              src.Log.print_with_color(f"[Lyr {self.layer_id}] Starting Pika I/O consumption loop (blocking)...", "yellow")
              try:
                   # start_consuming sẽ block luồng này cho đến khi bị dừng
                   self.channel.start_consuming()
                   # Code sau dòng này chỉ chạy khi start_consuming bị dừng (bởi stop_consuming hoặc lỗi)
                   src.Log.print_with_color(f"[Lyr {self.layer_id}] Pika I/O consumption loop finished.", "yellow")
              except BaseException as e: # Bắt cả KeyboardInterrupt và lỗi khác
                   src.Log.print_with_color(f"[Lyr {self.layer_id}] Pika I/O consumption loop stopped due to: {type(e).__name__}", "yellow")
                   # Không cần gọi stop_consuming ở đây vì nó sẽ được gọi từ client.py hoặc signal handler
              # Không trả về gì vì hàm này block
         else:
              src.Log.print_with_color(f"[Lyr {self.layer_id}] Cannot start processing, consumer not set up.", "red")
              return None # Trả về None nếu không phải layer 1 và consumer chưa sẵn sàng

    def stop_consuming(self):
        """Dừng việc xử lý và các luồng liên quan."""
        src.Log.print_with_color(f"[Lyr {self.layer_id}] Initiating stop sequence...", "yellow")
        self._stop_event.set() # Set cờ dừng cho tất cả các luồng con

        # --- Dừng luồng Layer 1 ---
        if self.layer1_thread and self.layer1_thread.is_alive():
            src.Log.print_with_color(f"[Lyr 1] Waiting for processing thread to finish...", "yellow")
            self.layer1_thread.join(timeout=5.0) # Chờ tối đa 5s
            if self.layer1_thread.is_alive():
                 src.Log.print_with_color(f"[Lyr 1] Warning: Processing thread did not stop gracefully.", "red")

        # --- Dừng luồng Inference (Layer > 1) ---
        if self.inference_thread and self.inference_thread.is_alive():
             src.Log.print_with_color(f"[Lyr {self.layer_id}] Stopping inference worker thread...", "yellow")
             if self.task_queue:
                  self.task_queue.put(None) # Gửi tín hiệu dừng
             self.inference_thread.join(timeout=5.0) # Chờ tối đa 5s
             if self.inference_thread.is_alive():
                  src.Log.print_with_color(f"[Lyr {self.layer_id}] Warning: Inference thread did not stop gracefully.", "red")

        # --- Dừng luồng Pika I/O (Layer > 1) ---
        # Cách tốt nhất để dừng start_consuming là đóng connection từ luồng khác
        # Hoặc gọi channel.stop_consuming() nếu có thể truy cập channel từ luồng khác một cách an toàn
        # Trong trường hợp này, client.py sẽ gọi rpc_client.close_connection()
        # nên start_consuming sẽ tự thoát ra do connection bị đóng.
        if self.is_consuming and self.channel and self.channel.is_open:
             try:
                  # Không cần gọi basic_cancel nếu connection sắp bị đóng
                  # src.Log.print_with_color(f"[Lyr {self.layer_id}] Cancelling consumer {self.consumer_tag}...", "yellow")
                  # self.channel.basic_cancel(self.consumer_tag)
                  # Chỉ cần đánh dấu là không consume nữa
                  self.is_consuming = False
                  src.Log.print_with_color(f"[Lyr {self.layer_id}] Marked Pika consumer as stopped.", "yellow")
             except Exception as e:
                  src.Log.print_with_color(f"[Lyr {self.layer_id}] Error during consumer cleanup (potential cancel): {e}", "red")


        # --- Dọn dẹp tài nguyên khác ---
        if self.video_writer is not None:
             src.Log.print_with_color(f"[Lyr {self.layer_id}] Releasing video writer.", "yellow")
             self.video_writer.release()
             self.video_writer = None
             # Có thể không cần destroyAllWindows nếu không dùng imshow
             # cv2.destroyAllWindows()

        src.Log.print_with_color(f"[Lyr {self.layer_id}] Stop sequence completed.", "yellow")