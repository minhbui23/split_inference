import os
import pika
import pickle
import torch
import torch.nn as nn

import src.Model
import src.Log


class Server:
    def __init__(self, config):
        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug-mode"]
        log_path = config["log-path"]
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]

        if action == "REGISTER":
            if (str(client_id), layer_id) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[layer_id-1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.notify_clients()

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_response(self, client_id, message):
        reply_queue_name = f"reply_{client_id}"
        self.reply_channel.queue_declare(reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def start(self):
        self.channel.start_consuming()

    def notify_clients(self):
        klass = getattr(src.Model, self.model_name)
        full_model = klass()
        full_model = nn.Sequential(*nn.ModuleList(full_model.children()))
        for (client_id, layer_id) in self.list_clients:
            filepath = f"{self.model_name}.pth"
            state_dict = None
            if layer_id == 1:
                layers = [0, self.cut_layer[0]]
            elif layer_id == len(self.total_clients):
                layers = [self.cut_layer[-1], -1]
            else:
                layers = [self.cut_layer[layer_id - 2], self.cut_layer[layer_id - 1]]

            if os.path.exists(filepath):
                full_state_dict = torch.load(filepath, weights_only=True)
                full_model.load_state_dict(full_state_dict)

                if layer_id == 1:
                    model_part = nn.Sequential(*nn.ModuleList(full_model.children())[:layers[1]])
                elif layer_id == len(self.total_clients):
                    model_part = nn.Sequential(*nn.ModuleList(full_model.children())[layers[0]:])
                else:
                    model_part = nn.Sequential(*nn.ModuleList(full_model.children())[layers[0]:layers[1]])

                state_dict = model_part.state_dict()
                src.Log.print_with_color("Model loaded successfully.", "green")
            else:
                src.Log.print_with_color(f"File {filepath} does not exist.", "yellow")

            response = {"action": "START",
                        "message": "Server accept the connection",
                        "parameters": state_dict,
                        "layers": layers,
                        "num_layers": len(self.total_clients),
                        "model_name": self.model_name}

            self.send_to_response(client_id, pickle.dumps(response))
