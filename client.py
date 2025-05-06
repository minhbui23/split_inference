import pika
import uuid
import argparse
import yaml
import signal
import sys
import threading # Thêm threading

import torch

import src.Log
from src.RpcClient import RpcClient
from src.Scheduler import Scheduler

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False, help='Device of client')
args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]

device = None
if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")


# Khởi tạo RpcClient trước
rpc_client = RpcClient(client_id, args.layer_id, address, username, password, virtual_host, None, device)

# Kiểm tra kết nối ban đầu
if not rpc_client.connection or not rpc_client.channel:
     src.Log.print_with_color("Failed to establish initial connection. Exiting.", "red")
     sys.exit(1)

# Tạo Scheduler, truyền connection và channel từ RpcClient
# QUAN TRỌNG: Truyền cả connection và channel
scheduler = Scheduler(client_id, args.layer_id, rpc_client.connection, rpc_client.channel, device)

# --- Biến toàn cục để lưu thread Layer 1 ---
layer1_processing_thread = None

# --- Hàm xử lý tín hiệu dừng (Ctrl+C) ---
def signal_handler(sig, frame):
    print("\nCtrl+C detected. Cleaning up...")
    # Dừng scheduler (sẽ dừng các luồng con và Pika consumer)
    scheduler.stop_consuming()
    # Đóng kết nối RabbitMQ (sẽ làm start_consuming thoát ra nếu đang chạy)
    rpc_client.close_connection()
    # Chờ luồng layer 1 nếu nó đang chạy (an toàn hơn)
    if layer1_processing_thread and layer1_processing_thread.is_alive():
         print("Waiting for Layer 1 thread to finish after Ctrl+C...")
         layer1_processing_thread.join(timeout=2.0)
    print("Exiting.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


# --- Luồng chính ---
if __name__ == "__main__":
    global layer1_processing_thread # Để signal handler có thể truy cập

    try:
        # 1. Gửi đăng ký đến server
        src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
        register_data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id, "message": "Hello from Client!"}
        if not rpc_client.send_to_server(register_data):
             raise Exception("Failed to send registration message.")

        # 2. Chờ nhận lệnh START và cấu hình từ server (blocking)
        src.Log.print_with_color("Waiting for START command from server...", "yellow")
        loaded_model, start_config = rpc_client.wait_for_start_command(timeout=300)

        # 3. Kiểm tra kết quả chờ
        if loaded_model and start_config:
            src.Log.print_with_color("START command received and model loaded. Setting up scheduler...", "green")
            # 4. Yêu cầu Scheduler thiết lập consumer/producer
            if scheduler.setup_consumer(loaded_model, start_config):
                 src.Log.print_with_color("Scheduler setup complete.", "green")
                 # 5. Bắt đầu xử lý
                 # start_processing_messages sẽ block nếu là consumer (layer > 1)
                 # hoặc trả về thread handle nếu là producer (layer 1)
                 thread_handle = scheduler.start_processing_messages()

                 if args.layer_id == 1:
                      if thread_handle and isinstance(thread_handle, threading.Thread):
                           layer1_processing_thread = thread_handle # Lưu lại handle
                           src.Log.print_with_color("Layer 1 processing started in background. Main thread waiting...", "yellow")
                           # Chờ luồng xử lý của Layer 1 hoàn thành
                           layer1_processing_thread.join() # Block luồng chính ở đây
                           src.Log.print_with_color("Layer 1 processing thread finished.", "green")
                      else:
                           src.Log.print_with_color("Error: Layer 1 setup did not return a valid thread handle.", "red")
                 else:
                      # Đối với layer > 1, start_processing_messages đã block và chỉ quay lại khi Pika loop dừng
                      src.Log.print_with_color(f"Layer {args.layer_id} Pika I/O loop finished.", "yellow")

                 src.Log.print_with_color("Processing finished or stopped gracefully.", "yellow")
            else:
                 src.Log.print_with_color("Failed to setup scheduler.", "red")
        else:
            src.Log.print_with_color("Did not receive valid START command or failed to load model.", "red")

    except Exception as e:
        src.Log.print_with_color(f"An error occurred in main client loop: {e}", "red")
        traceback.print_exc()
    finally:
        # Dọn dẹp khi kết thúc (đảm bảo kết nối được đóng và các luồng dừng)
        src.Log.print_with_color("Client shutting down...", "yellow")
        # Gọi stop_consuming để dừng các luồng con trước khi đóng connection
        scheduler.stop_consuming()
        rpc_client.close_connection()
        # Chờ thread layer 1 lần nữa nếu nó vẫn còn sống (phòng trường hợp thoát bất thường)
        if layer1_processing_thread and layer1_processing_thread.is_alive():
            print("Final check: Waiting for Layer 1 thread...")
            layer1_processing_thread.join(timeout=1.0)
        print("Client shutdown complete.")