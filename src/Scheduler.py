import pickle
import time
import torch
import src.Log
import cv2
import threading
# from PIL import Image # Không dùng trực tiếp
# import torchvision.transforms as transforms # Không dùng trực tiếp
from src.Model import SplitDetectionPredictor # Cần dùng SplitDetectionPredictor
import numpy as np # Cần dùng numpy


class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.model = None # Model sẽ được truyền vào sau
        self.predictor = None # Predictor sẽ được tạo sau
        self.is_consuming = False # Cờ đánh dấu đang consume hay không
        self.consumer_tag = None # Lưu consumer tag
        self.video_writer = None # Lưu đối tượng ghi video
        self.video_width = None
        self.video_height = None
        self.frame_count = 0 # Đếm frame để debug/log

    def send_next_layer(self, data, is_metadata=False):
        """Gửi dữ liệu đến layer tiếp theo."""
        if not self.channel or self.channel.is_closed:
            src.Log.print_with_color("Cannot send to next layer, channel is closed.", "red")
            return

        next_layer_queue = f"intermediate_queue_{self.layer_id}"
        try:
            # Khai báo queue cho chắc chắn
            self.channel.queue_declare(next_layer_queue, durable=False)

            action = "METADATA" if is_metadata else "OUTPUT"
            message = pickle.dumps({
                "action": action,
                "data": data
            })
            # Gửi message
            self.channel.basic_publish(
                exchange='',
                routing_key=next_layer_queue,
                body=message
            )
            # src.Log.print_with_color(f"[Lyr {self.layer_id} >>>] Sent {action} to {next_layer_queue}", "magenta")
        except Exception as e:
            src.Log.print_with_color(f"Error sending to {next_layer_queue}: {e}", "red")

    # --- Hàm Callback cho các layer ---
    def _handle_first_layer(self):
        """Logic xử lý cho Layer 1 (Đọc video và chạy head)."""
        # Hàm này không phải callback của consume mà là hàm chạy logic chính của layer 1
        # Nó sẽ đọc video và publish message, không consume gì cả.
        src.Log.print_with_color(f"[Lyr 1] Starting video processing...", "cyan")
        input_images = [] # Đổi tên biến để rõ ràng hơn
        # batch_size nên được lấy từ config response
        batch_size = self.config.get("batch_size", 1)
        save_output_flags = self.config.get("save_layers", []) # Layer cần lưu output trung gian

        # Tạo predictor một lần
        self.predictor = SplitDetectionPredictor(overrides={"imgsz": 640, "device": self.device})
        self.predictor.model = self.model # Gán model đã load cho predictor
        self.model.eval()
        self.model.to(self.device)

        video_path = "video.mp4" # Nên lấy từ config hoặc làm tham số
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            src.Log.print_with_color(f"Error: Could not open video {video_path}", "red")
            return False # Nên có cơ chế dừng chương trình

        # Gửi metadata một lần
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src.Log.print_with_color(f"Video Info: {width}x{height} @ {fps:.2f} FPS", "yellow")
        self.send_next_layer({"fps": fps, "width": width, "height": height}, is_metadata=True)


        frame_read_count = 0
        try:
            while True:
                start_frame_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    src.Log.print_with_color(f"End of video or cannot read frame.", "yellow")
                    break # Kết thúc video

                frame_read_count += 1
                # Resize và chuẩn hóa frame
                # Kích thước resize nên lấy từ config/model
                frame = cv2.resize(frame, (640, 640))
                # Chuyển sang tensor, đổi chiều, chuẩn hóa về [0, 1]
                # Đảm bảo frame là contiguous sau khi permute
                tensor = torch.from_numpy(frame).to(self.device).float().permute(2, 0, 1).contiguous() / 255.0
                input_images.append(tensor)

                # Xử lý khi đủ batch size
                if len(input_images) == batch_size:
                    # Stack các tensor thành batch
                    batch_tensor = torch.stack(input_images).to(self.device)
                    input_images = [] # Reset batch

                    # === Xử lý bằng model head ===
                    with torch.no_grad(): # Tắt tính gradient để tiết kiệm bộ nhớ và tăng tốc
                         # Chạy phần head của model
                         # Lưu ý: cần truyền đúng các layer cần lưu output từ config
                         y = self.model.forward_head(batch_tensor, output_from=save_output_flags)

                    # Chuyển output về CPU để gửi đi (nếu cần)
                    # Việc này quan trọng để tránh lỗi pickle/truyền tensor GPU
                    # Cần kiểm tra cấu trúc của y và chuyển từng tensor nếu cần
                    # Ví dụ đơn giản hóa:
                    if isinstance(y, dict):
                        for k, v in y.items():
                            if isinstance(v, torch.Tensor):
                                y[k] = v.cpu()
                            elif isinstance(v, list): # Xử lý trường hợp output là list các tensor
                                y[k] = [t.cpu() if isinstance(t, torch.Tensor) else t for t in v]

                    # Gửi kết quả trung gian đi
                    self.send_next_layer(y, is_metadata=False)
                    # Log tốc độ
                    # src.Log.print_with_color(f"[Lyr 1] Batch {frame_read_count // batch_size} sent.", "cyan")


                stop_frame_time = time.time()
                processing_time = stop_frame_time - start_frame_time
                # print(f"Frame {frame_read_count} processing time: {processing_time:.4f}s")
                # Có thể thêm sleep nhỏ ở đây nếu xử lý quá nhanh gây tốn CPU vô ích
                # time.sleep(0.001)

        except Exception as e:
            src.Log.print_with_color(f"Error during Layer 1 processing: {e}", "red", )
            import traceback
            traceback.print_exc() # In chi tiết lỗi
        finally:
            src.Log.print_with_color(f"Layer 1 finished processing. Total frames read: {frame_read_count}", "yellow")
            cap.release()
            # Có thể gửi một message đặc biệt để báo hiệu kết thúc? (ví dụ: action="EOS")
            self.send_next_layer({"status": "EOS"}, is_metadata=True)


    def _handle_last_layer_message(self, ch, method, properties, body):
        """Hàm callback xử lý message cho Layer cuối."""
        start_time = time.time()
        try:
            self.frame_count += 1
            # src.Log.print_with_color(f"[Lyr {self.layer_id} <<<] Received message {self.frame_count}, tag={method.delivery_tag}", "green")
            received_message = pickle.loads(body)
            action = received_message["action"]
            data = received_message["data"]

            if action == "METADATA":
                # Xử lý metadata (ví dụ: setup video writer)
                if "fps" in data: # Giả sử đây là metadata khởi tạo video
                    fps = data['fps']
                    self.video_width = data['width']
                    self.video_height = data['height']
                    output_filename = "output_push.mp4" # Đổi tên file output
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (self.video_width, self.video_height))
                    src.Log.print_with_color(f"Output video writer initialized: {output_filename}", "yellow")
                elif data.get("status") == "EOS":
                     src.Log.print_with_color("End Of Stream signal received.", "yellow")
                     # Có thể dừng consume ở đây nếu muốn
                     self.stop_consuming() # Gọi hàm dừng consume
                     # Không cần ack message EOS này nếu không muốn xử lý gì thêm


            elif action == "OUTPUT":
                 # Xử lý dữ liệu trung gian
                 y = data # Dữ liệu từ layer trước

                 # Chuyển input tensor về đúng device nếu cần (ví dụ nếu gửi từ CPU)
                 if isinstance(y, dict):
                      for k, v in y.items():
                           if isinstance(v, torch.Tensor):
                                y[k] = v.to(self.device)
                           elif isinstance(v, list):
                                y[k] = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in v]

                 # === Xử lý bằng model tail ===
                 with torch.no_grad():
                      predictions = self.model.forward_tail(y)

                 # === Hậu xử lý ===
                 # Cần lấy lại các thông tin gốc từ y nếu predictor cần
                 # Giả sử predictor đã được setup và có model
                 if self.predictor:
                    # Postprocess yêu cầu ảnh gốc và ảnh đã tiền xử lý nếu có lưu
                    # Trong code gốc, y["img"] và y["orig_imgs"] chỉ có khi save_output=True ở layer 1
                    # Cần điều chỉnh logic này nếu không dùng save_output=True
                    # Giả sử chúng ta không cần ảnh gốc cho postprocess ở đây để đơn giản hóa
                    # Hoặc cần sửa đổi cách truyền dữ liệu để luôn có ảnh
                    # results = self.predictor.postprocess(predictions, img=None, orig_imgs=None) # Ví dụ đơn giản

                    # Logic postprocess từ code gốc (cần đảm bảo y chứa đủ thông tin)
                    # Kiểm tra xem có cần save output không
                    save_viz = self.config.get("save_output", False)
                    if save_viz and self.video_writer:
                        # Cần đảm bảo 'y' chứa 'img', 'orig_imgs', 'path' khi save_output=True
                        # Nếu không, cần lấy thông tin này từ nguồn khác hoặc bỏ qua vẽ vời
                        try:
                            # Chuyển prediction về CPU để xử lý NMS và vẽ
                            processed_preds = self.predictor.postprocess(predictions.cpu(), y["img"].cpu(), y["orig_imgs"].cpu())
                            # processed_preds = self.predictor.postprocess(predictions.cpu(), y["img"].cpu(), y["orig_imgs"]) # Nếu orig_imgs là numpy

                            for res in processed_preds:
                                annotated_frame = res.plot() # Vẽ lên ảnh gốc
                                # Resize về kích thước video output và ghi
                                self.video_writer.write(cv2.resize(annotated_frame, (self.video_width, self.video_height)))
                        except KeyError as ke:
                             src.Log.print_with_color(f"Missing key for visualization: {ke}. Cannot save annotated frame.", "yellow")
                        except Exception as viz_e:
                             src.Log.print_with_color(f"Error during visualization/saving: {viz_e}", "red")

                 # Log thời gian xử lý
                 stop_time = time.time()
                 processing_time = stop_time - start_time
                 # print(f"Frame {self.frame_count} processing time (Lyr {self.layer_id}): {processing_time:.4f}s")


            # === Gửi Ack ===
            # Chỉ ack nếu không phải message EOS đã xử lý dừng
            if action != "METADATA" or data.get("status") != "EOS":
                 ch.basic_ack(delivery_tag=method.delivery_tag)
                 # src.Log.print_with_color(f"[Lyr {self.layer_id}] Ack sent for tag {method.delivery_tag}", "grey")

        except Exception as e:
            src.Log.print_with_color(f"Error processing message in Layer {self.layer_id}: {e}", "red")
            import traceback
            traceback.print_exc()
            # Cân nhắc gửi Nack để message không bị mất hoàn toàn nếu lỗi xử lý
            try:
                 ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False) # Không requeue để tránh lặp lỗi
                 src.Log.print_with_color(f"Nack sent for tag {method.delivery_tag}", "yellow")
            except Exception as nack_e:
                 src.Log.print_with_color(f"Error sending Nack: {nack_e}", "red")


    def _handle_middle_layer_message(self, ch, method, properties, body):
        """Hàm callback xử lý message cho Layer giữa (NẾU CÓ)."""
        # Tương tự last_layer, nhưng sau khi xử lý xong sẽ gọi send_next_layer
        src.Log.print_with_color(f"[Lyr {self.layer_id} <<<] Received message tag={method.delivery_tag}", "green")
        try:
            # Deserialize, xử lý bằng model middle part
            received_message = pickle.loads(body)
            action = received_message["action"]
            data = received_message["data"]

            if action == "OUTPUT":
                 y_prev = data
                 # Chuyển tensor về device nếu cần
                 if isinstance(y_prev, dict):
                      for k, v in y_prev.items():
                           if isinstance(v, torch.Tensor):
                                y_prev[k] = v.to(self.device)
                           elif isinstance(v, list):
                                y_prev[k] = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in v]

                 # Xử lý bằng model middle (cần định nghĩa trong Model.py)
                 # Giả sử có hàm forward_middle
                 with torch.no_grad():
                      # y_next = self.model.forward_middle(y_prev, output_from=...) # Ví dụ
                      # Hoặc nếu chỉ là forward_tail cho phần còn lại
                      y_next = self.model.forward_tail(y_prev) # Tạm dùng tail

                 # Chuyển output về CPU trước khi gửi
                 if isinstance(y_next, dict):
                      for k, v in y_next.items():
                           if isinstance(v, torch.Tensor):
                                y_next[k] = v.cpu()
                           elif isinstance(v, list):
                                y_next[k] = [t.cpu() if isinstance(t, torch.Tensor) else t for t in v]

                 # Gửi kết quả cho layer tiếp theo
                 self.send_next_layer(y_next, is_metadata=False)

            elif action == "METADATA":
                 # Chuyển tiếp metadata nếu cần
                 self.send_next_layer(data, is_metadata=True)


            # Gửi Ack
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            src.Log.print_with_color(f"Error processing message in Middle Layer {self.layer_id}: {e}", "red")
            # Cân nhắc gửi Nack
            try:
                 ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception as nack_e:
                 src.Log.print_with_color(f"Error sending Nack: {nack_e}", "red")


    def setup_consumer(self, model, config):
        """Thiết lập consumer dựa trên layer_id."""
        if not self.channel or self.channel.is_closed:
             src.Log.print_with_color("Cannot setup consumer, channel is closed.", "red")
             return False

        self.model = model # Nhận model đã load từ RpcClient
        self.config = config # Nhận config đã nhận từ RpcClient
        num_layers = config.get("num_layers", 1)

        src.Log.print_with_color(f"[Lyr {self.layer_id}] Setting up consumer...", "cyan")

        if self.layer_id == 1:
             # Layer 1 là producer, không cần consume dữ liệu trung gian
             # Nó sẽ chạy logic đọc video và publish
             # Chạy trong một thread riêng để không block luồng chính nghe ack/reply?
             # Hoặc nếu client.py dùng process_data_events thì có thể chạy tuần tự?
             # Tạm thời: Giả sử logic layer 1 chạy riêng biệt.
             src.Log.print_with_color(f"[Lyr 1] Setup complete (Producer role).", "cyan")
             # Gọi hàm xử lý chính của layer 1
             # Chạy trong thread để start_consuming ở client.py không bị block hoàn toàn việc xử lý L1
             # Hoặc nếu client.py dùng process_data_events thì không cần thread
             # self._handle_first_layer() # Chạy trực tiếp nếu client.py xử lý event loop
             processing_thread = threading.Thread(target=self._handle_first_layer, daemon=True)
             processing_thread.start()
             return True

        else:
             # Các layer khác là consumer
             input_queue = f"intermediate_queue_{self.layer_id - 1}"
             callback_func = None
             if self.layer_id == num_layers:
                 # Layer cuối
                 callback_func = self._handle_last_layer_message
                 # Setup predictor cho layer cuối nếu cần vẽ vời/lưu output
                 if self.config.get("save_output", False):
                      self.predictor = SplitDetectionPredictor(overrides={"imgsz": 640, "device": self.device})
                      self.predictor.model = self.model
                      self.model.eval()
                      self.model.to(self.device)

             else:
                 # Layer giữa (nếu có)
                 callback_func = self._handle_middle_layer_message
                 # Cũng cần setup model eval/device cho layer giữa
                 self.model.eval()
                 self.model.to(self.device)

             try:
                 # Khai báo queue input
                 self.channel.queue_declare(input_queue, durable=False)
                 # Đặt QoS
                 # Prefetch count nên lấy từ config hoặc để mặc định phù hợp
                 prefetch_count = self.config.get("prefetch_count", 10) # Ví dụ
                 self.channel.basic_qos(prefetch_count=prefetch_count)

                 # Đăng ký consumer
                 self.consumer_tag = self.channel.basic_consume(
                     queue=input_queue,
                     on_message_callback=callback_func,
                     auto_ack=False # Luôn đặt False để xử lý ack thủ công
                 )
                 self.is_consuming = True
                 src.Log.print_with_color(f"[Lyr {self.layer_id}] Consumer started on queue {input_queue} with tag {self.consumer_tag}", "cyan")
                 return True # Setup thành công
             except Exception as e:
                 src.Log.print_with_color(f"Error setting up consumer for layer {self.layer_id}: {e}", "red")
                 return False # Setup lỗi

    def start_processing_messages(self):
         """Bắt đầu vòng lặp tiêu thụ message (nếu là consumer)."""
         # Hàm này sẽ được gọi từ client.py SAU KHI RpcClient nhận xong START
         if self.is_consuming:
              src.Log.print_with_color(f"[Lyr {self.layer_id}] Starting blocking consumption loop...", "yellow")
              try:
                   # start_consuming sẽ block luồng này cho đến khi bị dừng
                   self.channel.start_consuming()
              except KeyboardInterrupt:
                   src.Log.print_with_color("Consumption interrupted by user.", "yellow")
                   self.stop_consuming()
              except Exception as e:
                   src.Log.print_with_color(f"Error during consumption: {e}", "red")
                   self.stop_consuming()
              finally:
                   # Dọn dẹp video writer nếu có
                   if self.video_writer:
                       src.Log.print_with_color("Releasing video writer.", "yellow")
                       self.video_writer.release()
                       cv2.destroyAllWindows() # Đóng cửa sổ nếu có imshow
         elif self.layer_id == 1:
              # Layer 1 không consume, nó đã chạy xử lý trong thread riêng (hoặc trực tiếp)
              src.Log.print_with_color(f"[Lyr 1] Producer role, no blocking consumption loop needed here.", "yellow")
              # Có thể thêm logic chờ thread xử lý của layer 1 hoàn thành nếu cần
              pass
         else:
              src.Log.print_with_color(f"[Lyr {self.layer_id}] Consumer not set up, cannot start processing.", "red")


    def stop_consuming(self):
        """Dừng việc tiêu thụ message."""
        if self.is_consuming and self.channel and self.channel.is_open and self.consumer_tag:
            try:
                src.Log.print_with_color(f"Stopping consumer {self.consumer_tag}...", "yellow")
                # self.channel.basic_cancel(self.consumer_tag) # Hủy consumer
                # Thay vì cancel, gọi stop_consuming trên channel/connection thường an toàn hơn
                self.channel.stop_consuming(consumer_tag=self.consumer_tag)
                self.is_consuming = False
            except Exception as e:
                src.Log.print_with_color(f"Error stopping consumer: {e}", "red")
        # Đóng video writer khi dừng
        if self.video_writer is not None:
             src.Log.print_with_color("Releasing video writer on stop.", "yellow")
             self.video_writer.release()
             self.video_writer = None
             cv2.destroyAllWindows()