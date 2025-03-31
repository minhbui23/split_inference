# split_inference

## Configuration
Application configuration is in the `config.yaml` file:
```yaml
name: Split Inference
server: # server configuration
  cut-layer:
    - 6
  clients: # all devices in system
    - 1
    - 1
  model: VGG16 # model name
rabbit: # RabbitMQ connection configuration
  address: 127.0.0.1
  username: admin
  password: admin
  virtual-host: /

log-path: .
debug-mode: False
```
This configuration is use for server.

## How to Run
Alter your configuration, you need to run the server to listen and control the request from clients.
### Server
```commandline
python server.py
```
### Client
Now, when server is ready, run clients simultaneously with total number of client that you defined.

**Layer 1**

```commandline
python client.py --layer_id 1 
```
Where:
- `--layer_id` is the ID index of client's layer, start from 1.

If you want to use a specific device configuration for the training process, declare it with the `--device` argument when running the command line:
```commandline
python client.py --layer_id 1 --device cpu
```