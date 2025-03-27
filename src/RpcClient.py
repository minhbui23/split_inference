import pickle
import time

import pika
import torch
import torch.nn as nn

import src.Log
import src.Model


class RpcClient:
    def __init__(self, client_id, layer_id, address, username, password, virtual_host, inference_func, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.inference_func = inference_func
        self.device = device

        self.channel = None
        self.connection = None
        self.response = None
        self.model = None
        self.connect()

    def wait_response(self):
        status = True
        reply_queue_name = f"reply_{self.client_id}"
        self.channel.queue_declare(reply_queue_name, durable=False)
        while status:
            method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
            if body:
                status = self.response_message(body)
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        src.Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]
        state_dict = self.response["parameters"]

        if action == "START":
            model_name = self.response["model_name"]
            cut_layers = self.response['layers']
            num_layers = self.response["num_layers"]

            klass = getattr(src.Model, model_name)
            full_model = klass()
            src.Log.print_with_color(f"Cut layer: {cut_layers}", "green")
            from_layer = cut_layers[0]
            to_layer = cut_layers[1]
            if to_layer == -1:
                self.model = nn.Sequential(*nn.ModuleList(full_model.children())[from_layer:])
            else:
                self.model = nn.Sequential(*nn.ModuleList(full_model.children())[from_layer:to_layer])
            self.model.to(self.device)
            if state_dict:
                self.model.load_state_dict(state_dict)
            status = True
            self.inference_func(self.model, num_layers)
            # Stop or Error
            return False
        else:

            return False

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, self.virtual_host, credentials))
        self.channel = self.connection.channel()

    def send_to_server(self, message):
        self.connect()
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

