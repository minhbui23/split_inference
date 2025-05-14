# # src/Scheduler.py
# import cv2
# import torch


# class Scheduler:
#     def __init__(self, client_id, layer_id, total_layers, rpc_client, logger, is_last_layer, device,
#                  imgsz_wh_for_nms=(640,640)):
#         self.client_id = client_id
#         self.layer_id = layer_id
#         self.total_layers = total_layers
#         self.rpc_client = rpc_client # Instance của RabbitMQClient hoặc tương tự để gửi message
#         self.logger = logger
#         self.is_last_layer = is_last_layer
#         self.device = device
#         self.imgsz_wh_for_nms = imgsz_wh_for_nms

#         self.batch_processing_info = {} 

#     def _request_io_publish(self, target_queue_name, message_body_dict):
#         """Helper để đưa yêu cầu publish vào IO queue (nếu dùng kiến trúc đó)
#            Hoặc gọi trực tiếp rpc_client nếu Scheduler có quyền."""
#         # Trong kiến trúc client.py hiện tại, Scheduler không trực tiếp dùng data_to_io_queue
#         # mà sẽ gọi phương thức của rpc_client được truyền vào.
#         try:
#             # Giả sử rpc_client có phương thức send_message_to_queue(routing_key, body_dict)
#             # hoặc một phương thức cụ thể như send_intermediate_result
#             # Đây là một placeholder, bạn cần điều chỉnh cho phù hợp với cách RpcClient/IOThread thực sự publish
#             if hasattr(self.rpc_client, 'send_message_direct'): # Ví dụ một hàm publish trực tiếp (cần thread-safe)
#                 self.rpc_client.send_message_direct(target_queue_name, message_body_dict)
#                 self.logger.log_info(f"[Scheduler] Gửi message trực tiếp tới {target_queue_name}")
#             # Nếu dùng data_to_io_queue (cách cũ hơn):
#             # elif self.data_to_io_queue:
#             #     self.data_to_io_queue.put({
#             #         "type": "PUBLISH_TO_QUEUE",
#             #         "payload": {
#             #             "routing_key": target_queue_name,
#             #             "message_body_dict": message_body_dict
#             #         }
#             #     })
#             #     self.logger.log_info(f"[Scheduler] Đã yêu cầu IO thread publish tới {target_queue_name}")
#             else: # Fallback nếu rpc_client không có hàm send trực tiếp và không có data_to_io_queue
#                   # Điều này chỉ ra một vấn đề thiết kế cần sửa.
#                   # Trong code client.py mới, rpc_client là IOThreadWorker, có _perform_publish
#                   # Scheduler nên gọi một phương thức của IOThreadWorker để publish an toàn.
#                   # Ví dụ: self.rpc_client.schedule_publish(target_queue_name, message_body_dict)
#                   # Hiện tại, giả sử rpc_client có hàm send_intermediate_result (như đã dùng ở dưới)
#                 self.logger.log_warning(f"[Scheduler] Không có cơ chế publish rõ ràng từ rpc_client. Cần kiểm tra lại.")
#                 return False # Không thể publish
#             return True
#         except Exception as e:
#             self.logger.log_error(f"[Scheduler] Lỗi khi yêu cầu publish tới {target_queue_name}: {e}")
#             return False


#     def _apply_post_processing(self, model_output_from_last_part, batch_idx):
#         self.logger.log_info(f"[Scheduler L{self.layer_id}] Áp dụng hậu xử lý cho batch_idx {batch_idx}.")
#         if batch_idx not in self.batch_processing_info:
#             self.logger.log_error(f"Thiếu thông tin frame (original_shapes_hw, letterbox_params) cho batch_idx {batch_idx}.")
#             return None

#         batch_info = self.batch_processing_info[batch_idx]
#         original_shapes_hw_batch = batch_info['original_shapes_hw']
#         letterbox_params_batch = batch_info['letterbox_params']
        
#         # Đảm bảo model_output_from_last_part là tensor trước khi vào NMS
#         # (YOLOv8 Detect() layer thường trả về list 3 tensor, cần concat thành [batch, num_preds, nc+5])
#         # Hoặc nếu SplitDetectionModel đã xử lý việc này, thì model_output sẽ là tensor phù hợp.
#         # Giả sử model_output đã là tensor [batch, num_preds, nc+5]
#         if isinstance(model_output_from_last_part, list): # Ví dụ output từ YOLOv8 Detect
#             # Code để concat (nếu cần) ví dụ:
#             # model_output_from_last_part = torch.cat([o.view(o.shape[0], -1, o.shape[-1]) for o in model_output_from_last_part], dim=1)
#             self.logger.log_warning(f"Output model cho NMS là list, cần kiểm tra logic concat. Hiện tại giả sử nó vẫn chạy được với non_max_suppression.")

#         preds = non_max_suppression(model_output_from_last_part,
#                                     conf_thres=self.conf_thres,
#                                     iou_thres=self.iou_thres)
#         self.logger.log_info(f"Sau NMS, số detections mỗi ảnh trong batch {batch_idx}: {[len(p) for p in preds]}")

#         final_detections_for_batch = []
#         for i, det_per_image in enumerate(preds):
#             if i < len(original_shapes_hw_batch) and i < len(letterbox_params_batch):
#                 original_shape_hw = original_shapes_hw_batch[i]
#                 lb_params = letterbox_params_batch[i]
#                 if det_per_image is not None and len(det_per_image):
#                     scaled_det = det_per_image.clone()
#                     scaled_det[:, :4] = scale_boxes(self.imgsz_wh_for_nms, scaled_det[:, :4], original_shape_hw, lb_params)
#                     final_detections_for_batch.append(scaled_det.cpu().numpy().tolist())
#                 else:
#                     final_detections_for_batch.append([])
#             else:
#                 self.logger.log_warning(f"Thiếu thông tin original_shape hoặc letterbox_params cho ảnh {i} batch {batch_idx}.")
#                 final_detections_for_batch.append(det_per_image.cpu().numpy().tolist() if det_per_image is not None and len(det_per_image) else [])
        
#         if batch_idx in self.batch_processing_info: # Dọn dẹp
#             del self.batch_processing_info[batch_idx]

#         return {
#             "results": final_detections_for_batch,
#             "client_id": self.client_id,
#             "batch_idx": batch_idx,
#             "message": "Hậu xử lý hoàn tất"
#         }

#     def process_video_layer1(self, model, video_path, batch_size, imgsz_wh_preprocess, device, half=False):
#         self.logger.log_info(f"[Scheduler L{self.layer_id}] Bắt đầu xử lý TOÀN BỘ video: {video_path} với batch_size={batch_size}")
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             self.logger.log_error(f"Không thể mở video: {video_path}")
#             return False

#         frames_buffer = []
#         current_batch_idx = 0
#         all_final_results_if_l1_is_last = []
#         total_frames_read = 0

#         while cap.isOpened() and not self.rpc_client.stop_event.is_set(): # Kiểm tra stop_event từ RabbitMQClient (IOThreadWorker)
#             ret, frame = cap.read()
#             if not ret:
#                 self.logger.log_info(f"Kết thúc video hoặc không thể đọc frame. Tổng số frame đã đọc: {total_frames_read}")
#                 break
#             total_frames_read += 1
#             frames_buffer.append(frame)

#             if len(frames_buffer) == batch_size:
#                 self.logger.log_debug(f"Đã thu thập batch {current_batch_idx} gồm {len(frames_buffer)} frames.")
                
#                 try:
#                     input_tensor, original_shapes_hw, letterbox_params = preprocess_batch(
#                         frames_buffer, imgsz_wh=imgsz_wh_preprocess, device=device, half=half
#                     )
#                     self.logger.log_debug(f"Tiền xử lý batch {current_batch_idx} OK. Shape: {input_tensor.shape}")
#                     self.batch_processing_info[current_batch_idx] = {
#                         'original_shapes_hw': original_shapes_hw,
#                         'letterbox_params': letterbox_params
#                     }
#                 except Exception as e:
#                     self.logger.log_error(f"Lỗi tiền xử lý batch {current_batch_idx}: {e}")
#                     frames_buffer = []
#                     current_batch_idx +=1
#                     continue

#                 try:
#                     with torch.no_grad():
#                         model_output = model(input_tensor)
#                     self.logger.log_debug(f"Suy luận L1 cho batch {current_batch_idx} OK.")

#                     if self.is_last_layer:
#                         final_batch_results_dict = self._apply_post_processing(model_output, current_batch_idx)
#                         if final_batch_results_dict:
#                             all_final_results_if_l1_is_last.append(final_batch_results_dict)
#                     else:
#                         next_layer_id = self.layer_id + 1
#                         payload_to_next = {
#                             "data": model.serialize_tensor(model_output),
#                             "source_client_id": self.client_id,
#                             "source_layer_id": self.layer_id,
#                             "batch_idx": current_batch_idx,
#                             "batch_processing_info": self.batch_processing_info[current_batch_idx]
#                         }
#                         # Sử dụng phương thức của rpc_client (IOThreadWorker) để publish an toàn
#                         self.rpc_client.schedule_publish(
#                             routing_key=f"intermediate_queue_{self.layer_id}", # L1 publish vào output queue của nó
#                             message_body_dict=payload_to_next
#                         )
#                         self.logger.log_info(f"L1 đã yêu cầu gửi output batch {current_batch_idx} tới L{next_layer_id} qua queue intermediate_queue_{self.layer_id}")
#                         if current_batch_idx in self.batch_processing_info:
#                             del self.batch_processing_info[current_batch_idx]
#                 except Exception as e:
#                     self.logger.log_error(f"Lỗi suy luận L1 hoặc gửi dữ liệu batch {current_batch_idx}: {e}")
#                     if current_batch_idx in self.batch_processing_info:
#                          del self.batch_processing_info[current_batch_idx]

#                 frames_buffer = []
#                 current_batch_idx += 1
        
#         # Xử lý frames còn lại nếu có (batch cuối không đủ size)
#         if frames_buffer and not self.rpc_client.stop_event.is_set():
#             self.logger.log_info(f"Xử lý {len(frames_buffer)} frame còn lại trong batch cuối {current_batch_idx}...")
#             try:
#                 input_tensor, original_shapes_hw, letterbox_params = preprocess_batch(
#                     frames_buffer, imgsz_wh=imgsz_wh_preprocess, device=device, half=half
#                 )
#                 self.batch_processing_info[current_batch_idx] = {
#                     'original_shapes_hw': original_shapes_hw,
#                     'letterbox_params': letterbox_params
#                 }
#                 with torch.no_grad():
#                     model_output = model(input_tensor)
#                 if self.is_last_layer:
#                     final_batch_results_dict = self._apply_post_processing(model_output, current_batch_idx)
#                     if final_batch_results_dict:
#                         all_final_results_if_l1_is_last.append(final_batch_results_dict)
#                 else:
#                     payload_to_next = {
#                         "data": model.serialize_tensor(model_output),
#                         "source_client_id": self.client_id,
#                         "source_layer_id": self.layer_id,
#                         "batch_idx": current_batch_idx,
#                         "batch_processing_info": self.batch_processing_info[current_batch_idx]
#                     }
#                     self.rpc_client.schedule_publish(
#                         routing_key=f"intermediate_queue_{self.layer_id}",
#                         message_body_dict=payload_to_next
#                     )
#                     if current_batch_idx in self.batch_processing_info:
#                         del self.batch_processing_info[current_batch_idx]
#                 self.logger.log_info(f"Đã xử lý batch cuối {current_batch_idx} với {len(frames_buffer)} frames.")
#             except Exception as e:
#                 self.logger.log_error(f"Lỗi khi xử lý batch cuối {current_batch_idx}: {e}")
#                 if current_batch_idx in self.batch_processing_info:
#                      del self.batch_processing_info[current_batch_idx]
#             current_batch_idx +=1


#         cap.release()
#         # cv2.destroyAllWindows() # Không cần thiết trong headless

#         # Gửi tín hiệu kết thúc stream nếu không phải layer cuối cùng
#         if not self.is_last_layer:
#             end_stream_payload = {
#                 "action": "STREAM_END",
#                 "source_client_id": self.client_id,
#                 "source_layer_id": self.layer_id
#             }
#             self.rpc_client.schedule_publish(
#                 routing_key=f"intermediate_queue_{self.layer_id}", # Gửi vào output queue của L1
#                 message_body_dict=end_stream_payload
#             )
#             self.logger.log_info(f"L1 đã gửi tín hiệu STREAM_END tới queue intermediate_queue_{self.layer_id}")

#         self.logger.log_info(f"Hoàn tất xử lý video Layer 1. Tổng số batch đã được xử lý hoặc cố gắng xử lý: {current_batch_idx}.")
        
#         if self.is_last_layer:
#             return all_final_results_if_l1_is_last
#         return True

#     def process_intermediate_data(self, model, received_payload, device, half=False):
#         # (Giữ nguyên logic của process_intermediate_data như đã cung cấp ở các bước trước,
#         #  nó đã được thiết kế để xử lý từng batch nhận được.
#         #  Chỉ cần đảm bảo nó nhận đúng các tham số và gọi _apply_post_processing nếu là layer cuối)
#         self.logger.log_info(f"[Scheduler L{self.layer_id}] Nhận dữ liệu trung gian từ L{received_payload['source_layer_id']} cho batch {received_payload.get('batch_idx')}.")
        
#         current_batch_idx = received_payload.get("batch_idx", -1)
#         if current_batch_idx == -1:
#             self.logger.log_error("batch_idx không tồn tại trong payload nhận được.")
#             return False 

#         try:
#             intermediate_tensor_serialized = received_payload["data"]
#             received_batch_info = received_payload.get("batch_processing_info")
#             if received_batch_info:
#                  self.batch_processing_info[current_batch_idx] = received_batch_info
#             else:
#                 self.logger.log_warning(f"batch_processing_info không có trong payload cho batch {current_batch_idx} từ L{received_payload['source_layer_id']}.")

#             intermediate_tensor = model.deserialize_tensor(intermediate_tensor_serialized, device=device)
            
#             with torch.no_grad():
#                 model_output = model(intermediate_tensor)
#             self.logger.log_info(f"Suy luận Layer {self.layer_id} cho batch {current_batch_idx} hoàn tất.")

#             if self.is_last_layer:
#                 self.logger.log_info(f"Đây là layer cuối (L{self.layer_id}). Áp dụng hậu xử lý cho batch {current_batch_idx}.")
#                 final_batch_results_dict = self._apply_post_processing(model_output, current_batch_idx)
#                 return final_batch_results_dict # Trả về dict kết quả
#             else: # Layer trung gian (nếu có nhiều hơn 2 layer)
#                 next_layer_id = self.layer_id + 1
#                 payload_to_next = {
#                     "data": model.serialize_tensor(model_output),
#                     "source_client_id": self.client_id,
#                     "source_layer_id": self.layer_id,
#                     "batch_idx": current_batch_idx,
#                     "batch_processing_info": self.batch_processing_info.get(current_batch_idx)
#                 }
#                 # Sử dụng phương thức của rpc_client (IOThreadWorker) để publish an toàn
#                 self.rpc_client.schedule_publish(
#                     routing_key=f"intermediate_queue_{self.layer_id}", # Publish vào output queue của layer hiện tại
#                     message_body_dict=payload_to_next
#                 )
#                 self.logger.log_info(f"L{self.layer_id} đã yêu cầu gửi output batch {current_batch_idx} tới L{next_layer_id} qua queue intermediate_queue_{self.layer_id}")
#                 if current_batch_idx in self.batch_processing_info: # Dọn dẹp
#                     del self.batch_processing_info[current_batch_idx]
#                 return True # Báo hiệu xử lý và gửi thành công
#         except Exception as e:
#             self.logger.log_error(f"Lỗi L{self.layer_id} xử lý batch {current_batch_idx}: {e}")
#             if current_batch_idx in self.batch_processing_info:
#                  del self.batch_processing_info[current_batch_idx]
#             return False