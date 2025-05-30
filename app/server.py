import argparse
import sys
import signal
import yaml
import threading
from core.server.server_logic import Server
from core.utils.common import delete_old_queues
from core.utils.logger import Logger

def main():

    parser = argparse.ArgumentParser(description="Split learning framework with controller.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"FATAL: Config file not found at '{args.config}'.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing config file: {e}")
        sys.exit(1)


    rabbit_config = config.get("rabbit", {})
    address = rabbit_config.get("address")
    username = rabbit_config.get("username")
    password = rabbit_config.get("password")
    virtual_host = rabbit_config.get("virtual-host")

    if not all([address, username, password, virtual_host]):
        print("FATAL: RabbitMQ connection details are missing in the config file.")
        sys.exit(1)


    def signal_handler(sig, frame):

        print("\nCatch stop signal Ctrl+C. Stop the program.")
        delete_old_queues(address, username, password, virtual_host)
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)


    print("Server is starting...")
    

    print("Cleaning up old queues...")
    delete_old_queues(address, username, password, virtual_host)


    server = Server(config)

        
    run_duration = config.get("app", {}).get("run_duration_seconds", 0)
    
    if run_duration > 0:
        print(f"Pipeline will automatically be signaled to stop in {run_duration} seconds.")
        # Timer sẽ gọi hàm _notify_all_clients_to_stop của đối tượng server
        shutdown_timer = threading.Timer(run_duration, server._notify_all_clients_to_stop)
        shutdown_timer.daemon = True 
        shutdown_timer.start()

    server.start()

    Logger.print_with_color("Ok, ready!", "green")

if __name__ == "__main__":
    main()