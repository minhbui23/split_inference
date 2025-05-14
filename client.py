# client.py
import pika
import uuid
import argparse
import yaml
import torch
import threading
import queue
import time
import pickle
import os
import cv2 # Để đọc video

import src.Log
from src.RpcClient import RpcClient
from src.Model import SplitDetectionModel, SplitDetectionPredictor
from ultralytics import YOLO # Để load model gốc
from src.Utils import FPSLogger

# ----- INFERENCE WORKER FUNCTION -----
def inference_worker_function(
    layer_id, num_layers, device, model_obj, predictor_obj,
    initial_params, # Chứa video_path, batch_frame, save_layers
    input_q, output_q, stop_evt, logger
):
    logger.log_info(f"[InferenceThread L{layer_id}] Starting.")
    is_first_layer = (layer_id == 1)
    is_last_layer = (layer_id == num_layers) # num_layers lấy từ initial_params

    batch_frame_size = initial_params["batch_frame"]

    # Khởi tạo FPSLogger
    log_prefix_for_fps = "Batch" if is_first_layer else "Batch of features"
    # Cẩn thận: logger có thể là None nếu có lỗi khởi tạo logger ở main.
    # FPSLogger nên có khả năng xử lý nếu logger_obj là None (ví dụ: không log gì cả).
    # Hoặc đảm bảo logger luôn được truyền vào là một đối tượng hợp lệ.
    fps_logger = FPSLogger(layer_id=layer_id, 
                           logger_obj=logger, 
                           log_interval_seconds=initial_params.get("fps_log_interval", 10), # Lấy từ config nếu có
                           log_prefix=log_prefix_for_fps)


    if is_first_layer:
        video_path = initial_params["data_source"]
        save_layers_config = initial_params["save_layers"]

        logger.log_info(f"[InferenceThread L{layer_id}] Processing video: {video_path}")
        if not os.path.exists(video_path):
            logger.log_error(f"[InferenceThread L{layer_id}] Video file not found: {video_path}")
            stop_evt.set()
            return
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.log_error(f"[InferenceThread L{layer_id}] Cannot open video: {video_path}")
            stop_evt.set()
            return

        frames_batch = []
        frame_count = 0
        while not stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.log_info(f"[InferenceThread L{layer_id}] End of video. Total frames processed: {frame_count}")
                if not is_last_layer: # Chỉ gửi STOP nếu không phải layer cuối (trường hợp num_layers = 1)
                    output_q.put(("STOP_INFERENCE", f"intermediate_queue_{layer_id + 1}"))
                break

            frame = cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            frames_batch.append(tensor)
            frame_count +=1

            if len(frames_batch) == batch_frame_size:
                fps_logger.start_batch_timing()
                stacked_batch = torch.stack(frames_batch).to(device)
                frames_in_this_actual_batch = stacked_batch.size(0) 
                frames_batch = []

                predictor_obj.setup_source(stacked_batch) # setup_source với batch tensor
                for predictor_obj.batch in predictor_obj.dataset: _, current_batch_images, _ = predictor_obj.batch
                preprocess_image = predictor_obj.preprocess(current_batch_images)
                
                y = model_obj.forward_head(preprocess_image, save_layers_config)

                fps_logger.end_batch_and_log_fps(frames_in_this_actual_batch)

                y["layers_output"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in y["layers_output"]]
                
                if not is_last_layer: # Chỉ gửi nếu không phải layer cuối
                    output_q.put((y, f"intermediate_queue_{layer_id + 1}"))

        cap.release()

        fps_logger.log_overall_fps(process_description="Video processing finished")
    # Bỏ qua logic middle layer theo yêu cầu (layer_id > 1 and not is_last_layer)

    elif is_last_layer and not is_first_layer: # Chỉ chạy nếu là last layer và không phải first layer
        logger.log_info(f"[InferenceThread L{layer_id}] Waiting for data from previous layer.")
        while not stop_evt.is_set():
            received_item = input_q.get() # Blocking get
            if received_item == "STOP_FROM_PREVIOUS":
                logger.log_info(f"[InferenceThread L{layer_id}] Received STOP from previous layer.")
                break 
            
            fps_logger.start_batch_timing()

            # Giả định received_item là dict có key "data"
            y_from_prev = received_item["data"]
            y_from_prev["layers_output"] = [t.to(device) if t is not None else None for t in y_from_prev["layers_output"]]
            
            final_predictions = model_obj.forward_tail(y_from_prev)

            fps_logger.end_batch_and_log_fps(batch_frame_size) 
            # results = predictor_obj.postprocess(final_predictions, y_from_prev.get("img"), y_from_prev.get("orig_imgs"), y_from_prev.get("path"))
            # logger.log_info(f"[InferenceThread L{layer_id}] Final prediction processed. Result example: {results[0] if results else 'No results'}")
            logger.log_info(f"[InferenceThread L{layer_id}] Last layer processed a batch.")
            input_q.task_done()

        fps_logger.log_overall_fps(process_description="Feature processing finished")    

    logger.log_info(f"[InferenceThread L{layer_id}] Stopped.")

# ----- IO WORKER FUNCTION -----
def io_worker_function(
    layer_id, num_layers, # num_layers từ initial_params
    rabbit_conn_params, 
    input_q, output_q, stop_evt, logger
):
    logger.log_info(f"[IOThread L{layer_id}] Starting.")
    is_first_layer = (layer_id == 1)
    is_last_layer = (layer_id == num_layers)

    credentials = pika.PlainCredentials(rabbit_conn_params["username"], rabbit_conn_params["password"])
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=rabbit_conn_params["address"], port=rabbit_conn_params["port"],
            virtual_host=rabbit_conn_params["virtual_host"], credentials=credentials,
            heartbeat=600, blocked_connection_timeout=300
        )
    )
    channel = connection.channel()
    logger.log_info(f"[IOThread L{layer_id}] RabbitMQ connection established for IO.")

    next_q_name = f"intermediate_queue_{layer_id + 1}"
    prev_q_name = f"intermediate_queue_{layer_id}"

    if not is_last_layer: # Layer 1 (và Middle nếu có) sẽ gửi đi
        channel.queue_declare(queue=next_q_name, durable=False)
        logger.log_info(f"[IOThread L{layer_id}] Declared send queue: {next_q_name}")

    if not is_first_layer: # Layer Last (và Middle nếu có) sẽ nhận
        channel.queue_declare(queue=prev_q_name, durable=False)
        channel.basic_qos(prefetch_count=5) 
        logger.log_info(f"[IOThread L{layer_id}] Declared listen queue: {prev_q_name}, QOS set.")

    while not stop_evt.is_set():
        # 1. Gửi dữ liệu (nếu là layer 1 và có gì đó trong output_q)
        if not is_last_layer: # Chỉ layer 1 gửi (trong mô hình 2 layer)
            item_to_send = None
            try:
                item_to_send = output_q.get(block=False, timeout=0.01) 
            except queue.Empty:
                pass 

            if item_to_send:
                data_payload, target_rabbit_q = item_to_send
                message_body = None
                if data_payload == "STOP_INFERENCE":
                    logger.log_info(f"[IOThread L{layer_id}] Sending STOP to {target_rabbit_q}")
                    message_body = pickle.dumps("STOP")
                else:
                    message_body = pickle.dumps({"action": "OUTPUT", "data": data_payload})
                
                channel.basic_publish(exchange='', routing_key=target_rabbit_q, body=message_body)
                output_q.task_done()
                if data_payload == "STOP_INFERENCE":
                    # Nếu layer 1 gửi STOP, nó có thể dừng phần gửi của mình
                    if is_first_layer and is_last_layer: # Trường hợp chỉ có 1 layer duy nhất
                         stop_evt.set() # Dừng hẳn nếu chỉ có 1 layer
                    elif is_first_layer and not is_last_layer:
                         pass # Vẫn chờ tín hiệu dừng chung từ stop_evt
                    # break # Không break ở đây, để vòng lặp chính check stop_evt

        # 2. Nhận dữ liệu (nếu là layer cuối và không phải layer đầu)
        if is_last_layer and not is_first_layer:
            method_frame, properties, body = channel.basic_get(queue=prev_q_name, auto_ack=False) # Manual ack
            if method_frame:
                channel.basic_ack(delivery_tag=method_frame.delivery_tag) # Ack ngay
                received_message = pickle.loads(body)
                if received_message == "STOP":
                    logger.log_info(f"[IOThread L{layer_id}] Received STOP from RabbitMQ on {prev_q_name}.")
                    input_q.put("STOP_FROM_PREVIOUS")
                    break # Thoát vòng lặp IOThread này vì đã nhận STOP cuối cùng
                else:
                    input_q.put(received_message)
            # else: time.sleep(0.01) # Bỏ sleep nếu dùng blocking_connection.process_data_events

        # Cho phép connection xử lý các sự kiện nền và check stop_evt
        # connection.process_data_events(time_limit=0.01) # Quan trọng để BlockingConnection không bị block hoàn toàn
        # Nếu không dùng process_data_events, vòng lặp sẽ quay nhanh nếu không có message, cần sleep
        if not (method_frame if 'method_frame' in locals() else None) and not (item_to_send if 'item_to_send' in locals() else None):
            time.sleep(0.05) # Ngủ nhẹ nếu không có gì xảy ra

        # Điều kiện dừng cho IO Thread:
        # Layer 1: Khi đã gửi STOP và output_q trống (hoặc stop_evt)
        # Layer cuối: Khi đã nhận STOP từ RabbitMQ (đã break ở trên) hoặc stop_evt
        if is_first_layer and item_to_send and data_payload == "STOP_INFERENCE" and output_q.empty():
             logger.log_info(f"[IOThread L{layer_id}] First layer sent STOP and output queue is empty. Signaling stop.")
             # stop_evt.set() # Để stop_event ở main kiểm soát chung
             pass # Chờ stop_event từ main

    if connection.is_open:
        channel.close()
        connection.close()
    logger.log_info(f"[IOThread L{layer_id}] Stopped and RabbitMQ connection closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split learning client")
    parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
    parser.add_argument('--device', type=str, required=False, help='Device of client (cpu or cuda)')
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    client_uuid = str(uuid.uuid4()) # Đổi tên biến để tránh nhầm lẫn
    layer_id_arg = args.layer_id

    # --- Logger ---
    main_logger = src.Log.Logger(f"client_{client_uuid}_L{layer_id_arg}.log") 
    main_logger.log_info(f"Client {client_uuid} (Layer {layer_id_arg}) starting...")

    # --- Device ---
    device_arg = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    main_logger.log_info(f"Using device: {device_arg}")

    # --- RabbitMQ Connection (Cho RpcClient) ---
    rabbit_config = config["rabbit"]
    
    # --- RPC Client ---
    rpc_client = RpcClient(client_uuid, layer_id_arg, rabbit_config["address"], 
                           rabbit_config["username"], rabbit_config["password"],
                           rabbit_config["virtual-host"], main_logger)

    main_logger.log_info("Sending registration message to server...")
    registration_data = {"action": "REGISTER", "client_id": client_uuid, "layer_id": layer_id_arg}
    rpc_client.send_to_server(registration_data)

    if rpc_client.wait_response(): # Chờ xác nhận đăng ký và sau đó là START
        initial_params_from_server = rpc_client.initial_params
        if initial_params_from_server and "model_name" in initial_params_from_server:
            main_logger.log_info("Received START from server. Initializing worker threads...")
            
            # --- Load Model (Sau khi có initial_params) ---
            # Đường dẫn tới file model đã được RpcClient lưu
            model_file_path = initial_params_from_server["model_save_path"]
            if not os.path.exists(model_file_path):
                main_logger.log_error(f"Model file {model_file_path} not found after RpcClient download. Exiting.")
                rpc_client.close()
                exit()

            # Load pretrain model từ file .pt đã được RpcClient tải về
            # Cấu trúc của YOLO().model có thể cần xem lại, tùy thuộc vào cách SplitDetectionModel sử dụng nó
            # Giả sử pretrain_model_yolo là một object model đã load, không phải chỉ là state_dict
            try:
                # Đối với YOLOv8, việc load model .pt có thể trực tiếp hơn:
                # pretrain_model_yolo = YOLO(model_file_path).model # Dòng này có thể gây lỗi nếu model không đúng định dạng mong muốn
                # Hoặc nếu SplitDetectionModel mong đợi một state_dict hoặc một kiến trúc cụ thể:
                # model_checkpoint = torch.load(model_file_path, map_location=device_arg)
                # pretrain_model_yolo = YOLO(initial_params_from_server["model_name"]+".yaml") # Load kiến trúc từ yaml
                # pretrain_model_yolo.load_state_dict(model_checkpoint['model'].state_dict()) # Load weights
                # pretrain_model_yolo = pretrain_model_yolo.model
                
                # Đơn giản nhất là load trực tiếp nếu file .pt là model hoàn chỉnh
                # Cần đảm bảo server gửi file .pt đúng cách
                # Tạm thời giả định SplitDetectionModel có thể xử lý model_file_path
                pretrain_yolo_model_for_split = YOLO(model_file_path).model # Đây là cách gốc
            except Exception as e:
                main_logger.log_error(f"Error loading YOLO model from {model_file_path}: {e}")
                rpc_client.close()
                exit()


            split_detection_model = SplitDetectionModel(pretrain_yolo_model_for_split, 
                                                        split_layer=initial_params_from_server["splits"])
            split_detection_model.to(device_arg)
            split_detection_model.eval()
            
            # Khởi tạo Predictor (dùng trong InferenceThread)
            # Predictor cần model đã load, không phải model_obj
            predictor = SplitDetectionPredictor(split_detection_model,overrides={"imgsz": 640}) # Bỏ model_obj, vì nó sẽ dùng batch trực tiếp

            # --- Queues & Event ---
            input_data_queue = queue.Queue(maxsize=20) 
            output_data_queue = queue.Queue(maxsize=20)
            stop_event = threading.Event()

            num_total_layers = initial_params_from_server["num_layers"]

            # --- RabbitMQ Connection Params cho IOThread ---
            rabbit_conn_params_for_io = {
                "address": rabbit_config["address"], "username": rabbit_config["username"],
                "password": rabbit_config["password"], "virtual_host": rabbit_config["virtual-host"],
                "port": rabbit_config.get("port", 5672) # Lấy port từ config hoặc mặc định
            }

            # --- Khởi tạo và chạy Threads ---
            inference_thread = threading.Thread(target=inference_worker_function,
                                                args=(layer_id_arg, num_total_layers, device_arg,
                                                      split_detection_model, predictor,
                                                      initial_params_from_server,
                                                      input_data_queue, output_data_queue, 
                                                      stop_event, main_logger))

            io_thread = threading.Thread(target=io_worker_function,
                                         args=(layer_id_arg, num_total_layers,
                                               rabbit_conn_params_for_io,
                                               input_data_queue, output_data_queue, 
                                               stop_event, main_logger))
            
            start_overall_time = time.time()
            inference_thread.start()
            io_thread.start()

            # Chờ các luồng hoàn thành hoặc có KeyboardInterrupt
            try:
                while not stop_event.is_set() : 
                    if not inference_thread.is_alive() and not io_thread.is_alive():
                        main_logger.log_info("Both threads have completed their work.")
                        stop_event.set() # Đảm bảo vòng lặp dừng
                        break 
                    # Kiểm tra nếu một trong hai luồng chết bất thường
                    if layer_id_arg == 1 and not inference_thread.is_alive() and io_thread.is_alive() and output_data_queue.empty():
                        main_logger.log_warning("InferenceThread L1 finished but IOThread still alive and output_q empty. Signaling stop.")
                        stop_event.set() # Có thể inference xong nhưng IO chưa nhận được tín hiệu cuối
                    if layer_id_arg == num_total_layers and not io_thread.is_alive() and inference_thread.is_alive() and input_data_queue.empty():
                        main_logger.log_warning("IOThread L_last finished but InferenceThread still alive and input_q empty. Signaling stop.")
                        stop_event.set()


                    time.sleep(0.5)
            except KeyboardInterrupt:
                main_logger.log_info("Ctrl+C pressed. Signaling threads to stop...")
                stop_event.set()

            inference_thread.join(timeout=10) 
            io_thread.join(timeout=10)

            if inference_thread.is_alive():
                main_logger.log_warning("Inference thread did not stop in time.")
            if io_thread.is_alive():
                main_logger.log_warning("IO thread did not stop in time.")

            end_overall_time = time.time()
            main_logger.log_info(f"All threads joined. Overall execution time: {end_overall_time - start_overall_time:.4f}s")
        
        elif initial_params_from_server and "error" in initial_params_from_server:
             main_logger.log_error(f"Could not start processing due to server error: {initial_params_from_server['error']}")
        else:
            main_logger.log_error("Did not receive valid START command or parameters from server.")
    else:
        main_logger.log_error("Failed to get response from server after registration or timeout.")

    rpc_client.close() # Đảm bảo đóng kết nối RPC
    main_logger.log_info(f"Client {client_uuid} (Layer {layer_id_arg}) finished.")