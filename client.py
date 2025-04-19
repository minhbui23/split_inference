import pika
import uuid
import argparse
import yaml
import signal # Thêm signal để xử lý Ctrl+C
import sys # Thêm sys

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



rpc_client = RpcClient(client_id, args.layer_id, address, username, password, virtual_host, None, device) 

# Kiểm tra kết nối ban đầu
if not rpc_client.channel:
     src.Log.print_with_color("Failed to establish initial connection. Exiting.", "red")
     sys.exit(1)

# Tạo Scheduler, truyền channel từ RpcClient
scheduler = Scheduler(client_id, args.layer_id, rpc_client.channel, device)

# --- Hàm xử lý tín hiệu dừng (Ctrl+C) ---
def signal_handler(sig, frame):
    print("\nCtrl+C detected. Cleaning up...")
    # Dừng consumer của scheduler nếu đang chạy
    scheduler.stop_consuming()
    # Đóng kết nối RabbitMQ
    rpc_client.close_connection()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


# --- Luồng chính mới ---
if __name__ == "__main__":
    try:
        # 1. Gửi đăng ký đến server
        src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
        register_data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id, "message": "Hello from Client!"}
        if not rpc_client.send_to_server(register_data):
             raise Exception("Failed to send registration message.") # Ném lỗi nếu gửi thất bại

        # 2. Chờ nhận lệnh START và cấu hình từ server (blocking)
        src.Log.print_with_color("Waiting for START command from server...", "yellow")
        # Hàm này sẽ block cho đến khi nhận được reply hoặc timeout
        loaded_model, start_config = rpc_client.wait_for_start_command(timeout=300) # Chờ tối đa 5 phút

        # 3. Kiểm tra kết quả chờ
        if loaded_model and start_config:
            src.Log.print_with_color("START command received and model loaded. Setting up scheduler consumer...", "green")
            # 4. Yêu cầu Scheduler thiết lập consumer tương ứng (nếu là layer > 1)
            # hoặc bắt đầu xử lý (nếu là layer 1)
            if scheduler.setup_consumer(loaded_model, start_config):
                 # 5. Bắt đầu vòng lặp xử lý message của Pika (nếu là consumer)
                 # hoặc để thread của layer 1 chạy
                 scheduler.start_processing_messages() # Hàm này sẽ block nếu là consumer
                 src.Log.print_with_color("Processing finished or stopped.", "yellow")
            else:
                 src.Log.print_with_color("Failed to setup scheduler consumer.", "red")
        else:
            src.Log.print_with_color("Did not receive valid START command or failed to load model.", "red")

    except Exception as e:
        src.Log.print_with_color(f"An error occurred in main client loop: {e}", "red")
        import traceback
        traceback.print_exc()
    finally:
        # Dọn dẹp khi kết thúc (đảm bảo kết nối được đóng)
        src.Log.print_with_color("Client shutting down...", "yellow")
        rpc_client.close_connection()