import pickle
import torch
import src.Log
from PIL import Image
import torchvision.transforms as transforms


class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device

    def send_next_layer(self, trace, output):
        intermediate_queue = f"intermediate_queue_{self.layer_id}"
        self.channel.queue_declare(intermediate_queue, durable=False)
        if trace:
            trace.append(self.client_id)
            message = pickle.dumps({
                "data": output.detach().cpu().numpy(),
                "trace": trace
            })
        else:
            message = pickle.dumps({
                "data": output.detach().cpu().numpy(),
                "trace": [self.client_id]
            })
        self.channel.basic_publish(
            exchange='',
            routing_key=intermediate_queue,
            body=message
        )

    def send_result(self, output, trace):
        result_queue = f"result_{trace[0]}"
        self.channel.queue_declare(queue=result_queue, durable=False)
        message = pickle.dumps({
            "data": output
        })
        self.channel.basic_publish(
            exchange='',
            routing_key=result_queue,
            body=message
        )

    def first_layer(self, model):
        image_path = "so.jpg"
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(image_tensor)

        self.send_next_layer(trace=None, output=output)
        result_queue = f"result_{self.client_id}"
        self.channel.queue_declare(queue=result_queue, durable=False)
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=result_queue, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                output = received_data["data"]
                _, predicted_class = output.max(1)
                print(f"{predicted_class}")
                break

    def last_layer(self, model):
        last_queue = f"intermediate_queue_{self.layer_id - 1}"
        self.channel.queue_declare(queue=last_queue, durable=False)
        self.channel.basic_qos(prefetch_count=10)
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=last_queue, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                output = received_data["data"]
                output = torch.tensor(output).to(self.device)
                trace = received_data["trace"]
                with torch.no_grad():
                    output = model(output)

                self.send_result(output, trace)
                break

    def middle_layer(self, model):
        last_queue = f"intermediate_queue_{self.layer_id - 1}"
        next_queue = f"intermediate_queue_{self.layer_id}"
        self.channel.queue_declare(queue=last_queue, durable=False)
        self.channel.queue_declare(queue=next_queue, durable=False)
        self.channel.basic_qos(prefetch_count=10)
        src.Log.print_with_color("Waiting for intermediate output. To exit press CTRL+C", "red")
        model.to(self.device)
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=last_queue, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                output = received_data["data"]
                trace = received_data["trace"]
                output = torch.tensor(output).to(self.device)
                output = model(output)

                self.send_next_layer(trace, output)

    def inference_func(self, model, num_layers):
        if self.layer_id == 1:
            self.first_layer(model)
        elif self.layer_id == num_layers:
            self.last_layer(model)
        else:
            self.middle_layer(model)
