import argparse
import sys
import signal
from core.server.server_logic import Server
from core.utils.common import delete_old_queues
from core.utils.logger import Logger
import yaml

parser = argparse.ArgumentParser(description="Split learning framework with controller.")
args = parser.parse_args()

with open('config.yaml') as file:
    config = yaml.safe_load(file)

address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]


def signal_handler(sig, frame):
    """Handles the stop signal (Ctrl+C) to clean up queues and exit.

    Args:
        sig: The signal received.
        frame: The current stack frame.
    """
    ...
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    delete_old_queues(address, username, password, virtual_host)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    delete_old_queues(address, username, password, virtual_host)
    server = Server(config)
    server.start()
    Logger.print_with_color("Ok, ready!", "green")
