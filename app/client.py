# client.py (Phiên bản chính, gọn hơn)
import pika # Vẫn cần cho exception handling ở một số chỗ nếu có
import uuid
import argparse
import yaml
import torch
import threading
import queue
import time
# import pickle # Không cần trực tiếp ở đây nữa nếu RpcClient và worker tự xử lý
import os
# import cv2 # Không cần trực tiếp ở đây nữa

from core.utils.logger import Logger
from core.rpc.rpc_client import RpcClient
# Các lớp worker giờ sẽ được import từ file mới
from core.client.inference_worker import FirstLayerWorker, LastLayerWorker, MiddleLayerWorker
from core.client.io_worker import FirstLayerIOWorker, LastLayerIOWorker, MiddleLayerIOWorker

from core.utils.metric_logger import MetricsLogger
from core.model.model_utils import setup_inference_components

stop_event = threading.Event()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split learning client")
    parser.add_argument('--layer_id', type=int, required=True, help='ID of layer')
    parser.add_argument('--device', type=str, required=False, help='Device (cpu or cuda)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"FATAL: Config file not found at '{args.config}'.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing config file: {e}")
        exit(1)


    client_uuid = str(uuid.uuid4())
    layer_id_arg = args.layer_id

    # --- Logger ---
    log_dir_from_config = config.get("log-path", "logs")
    client_log_file_name = f"client_{client_uuid}_L{layer_id_arg}.log"
    client_log_full_path = os.path.join(log_dir_from_config, client_log_file_name)
    main_logger = Logger(client_log_full_path)
    main_logger.log_info(f"Client {client_uuid} (L{layer_id_arg}) starting. Logging to: {client_log_full_path}")

    metrics_logger = None

    # --- Device ---
    device_arg = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    main_logger.log_info(f"Using device: {device_arg}")

    # --- RabbitMQ and Redis Connection Params (Cho RpcClient và IOThread) ---
    rabbit_config = config.get("rabbit")
    if not rabbit_config:
        main_logger.log_error("FATAL: RabbitMQ configuration missing in config.yaml.")
        exit(1)
    
    rabbit_conn_params_for_io = {
        "address": rabbit_config.get("address", "127.0.0.1"), 
        "username": rabbit_config.get("username", "guest"),
        "password": rabbit_config.get("password", "guest"), 
        "virtual_host": rabbit_config.get("virtual-host", "/"),
        "port": rabbit_config.get("port", 5672)
    }
    redis_conn_params = config.get("redis")

    if not redis_conn_params:
        main_logger.log_error("FATAL: Redis configuration missing in config.yaml.")
        exit(1)


    # --- RPC Client ---
    rpc_client = None # Khởi tạo để có thể close trong finally
    try:
        rpc_client = RpcClient(client_uuid, layer_id_arg, 
                               rabbit_conn_params_for_io["address"], 
                               rabbit_conn_params_for_io["username"], 
                               rabbit_conn_params_for_io["password"],
                               rabbit_conn_params_for_io["virtual_host"], 
                               main_logger, stop_event)

        main_logger.log_info("Sending registration to server...")
        registration_data = {"action": "REGISTER", "client_id": client_uuid, "layer_id": layer_id_arg}
        rpc_client.send_to_server(registration_data)

        if rpc_client.wait_response():
            initial_params_from_server = rpc_client.initial_params

            if initial_params_from_server and initial_params_from_server.get("model_name") and not initial_params_from_server.get("error"):
                main_logger.log_info("Received START from server. Initializing components and worker threads...")
                
                # Load Config from server response and config file 
                params_for_workers = initial_params_from_server.copy()

                server_config = config.get("server", {})
                client_config = config.get("client", {})

                params_for_workers["data_source"] = config.get("data")
                params_for_workers["debug_mode"] = config.get("debug-mode", False)
                params_for_workers["batch_frame"] = server_config.get("batch-frame")
                
                params_for_workers.update(client_config)

                params_for_workers["device"] = device_arg
                params_for_workers["client_layer_id"] = layer_id_arg

                num_total_layers = params_for_workers.get("num_layers")

                # --- Load Model và Predictor sử dụng hàm tiện ích ---
                model_obj_for_worker, predictor_obj_for_worker = setup_inference_components(
                    params_for_workers, device_arg, main_logger
                )

                if model_obj_for_worker is None or predictor_obj_for_worker is None:
                    main_logger.log_error("Failed to initialize model or predictor. Exiting.")
                    if rpc_client: rpc_client.close(); exit(1)


                # --- Queues & Event ---
                q_maxsize = params_for_workers.get("internal_queue_maxsize", 20)
                input_data_queue = queue.Queue(maxsize=q_maxsize)
                output_data_queue = queue.Queue(maxsize=q_maxsize)
                ack_trigger_queue = queue.Queue(maxsize=q_maxsize * 2) # ack_q có thể cần lớn hơn một chút
                

                io_worker_class = None
                inference_worker_class = None

                if layer_id_arg == 1:
                    io_worker_class = FirstLayerIOWorker
                    inference_worker_class = FirstLayerWorker
                elif layer_id_arg == num_total_layers:
                    io_worker_class = LastLayerIOWorker
                    inference_worker_class = LastLayerWorker
                    # Khởi tạo metrics logger chỉ cho layer cuối
                    metrics_fields = ['batch_id', 't1', 't2', 'q1', 't3', 'q2', 't4', 't5']
                    metrics_log_name = f'metrics_L{layer_id_arg}.csv'
                    metrics_log_full_path = os.path.join(log_dir_from_config, metrics_log_name)
                    metrics_logger = MetricsLogger(metrics_log_full_path, metrics_fields)

                else:
                    io_worker_class = MiddleLayerIOWorker
                    inference_worker_class = MiddleLayerWorker


                # --- Khởi tạo và chạy Threads ---
                io_thread = io_worker_class(
                    layer_id=layer_id_arg, num_layers=num_total_layers,
                    rabbit_conn_params=rabbit_conn_params_for_io,
                    redis_conn_params=redis_conn_params,
                    initial_params=params_for_workers, # Truyền full initial_params
                    input_q=input_data_queue, output_q=output_data_queue,
                    ack_trigger_q=ack_trigger_queue,
                    stop_evt=stop_event, logger=main_logger
                )

                inference_thread = inference_worker_class(
                    layer_id=layer_id_arg, num_layers=num_total_layers, device=device_arg,
                    model_obj=model_obj_for_worker, predictor_obj=predictor_obj_for_worker,
                    initial_params=params_for_workers,
                    input_q=input_data_queue, output_q=output_data_queue,
                    ack_trigger_q=ack_trigger_queue,
                    stop_evt=stop_event, 
                    logger=main_logger, metrics_logger=metrics_logger
                )
                
                main_logger.log_info("Starting IO and Inference threads...")
                start_overall_time = time.time()
                io_thread.start()
                inference_thread.start()
                
                try:
                    while not stop_event.is_set():
                        try:
                            if rpc_client and rpc_client.connection and rpc_client.connection.is_open:
                                rpc_client.connection.process_data_events(time_limit=0.1) 
                        except pika.exceptions.AMQPConnectionError:
                            main_logger.error("RPC connection lost while polling. Stopping client.")
                            stop_event.set()
                        except Exception as e: 
                            main_logger.error(f"Error polling RPC events: {e}")
                            stop_event.set()
                        if not inference_thread.is_alive() and not io_thread.is_alive():
                            main_logger.log_info("Both worker threads have finished.")
                            stop_event.set(); break
                        if not inference_thread.is_alive() and io_thread.is_alive():
                            main_logger.log_warning(f"InferenceThread L{layer_id_arg} exited prematurely. Signaling stop.")
                            stop_event.set()
                        elif not io_thread.is_alive() and inference_thread.is_alive():
                            main_logger.log_warning(f"IOThread L{layer_id_arg} exited prematurely. Signaling stop to InferenceThread.")
                            stop_event.set()
                        time.sleep(0.4)
                except KeyboardInterrupt:
                    main_logger.log_info("Ctrl+C pressed. Signaling threads to stop...")
                    stop_event.set()

                main_logger.log_info("Waiting for threads to join...")
                io_thread.join(timeout=10) # Join IOThread trước có thể tốt hơn
                inference_thread.join(timeout=10)

                if inference_thread.is_alive(): main_logger.log_warning("Inference thread did not join in time.")
                if io_thread.is_alive(): main_logger.log_warning("IO thread did not join in time.")

                end_overall_time = time.time()
                main_logger.log_info(f"All threads joined. Overall active time: {end_overall_time - start_overall_time:.4f}s")
            
            elif initial_params_from_server and "error" in initial_params_from_server:
                 main_logger.log_error(f"Could not start: Server error: {initial_params_from_server['error']}")
            else:
                main_logger.log_error("Did not receive valid START or parameters from server.")
        else:
            main_logger.log_error("Failed to get successful response from server after registration or timeout.")

    except Exception as e: # Bắt lỗi chung ở main
        main_logger.log_error(f"FATAL error in main client execution: {e}")
    finally:
        if rpc_client: rpc_client.close()
        main_logger.log_info(f"Client {client_uuid} (Layer {layer_id_arg}) finished execution sequence.")