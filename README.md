# split_inference

## Configuration
Application configuration is in the `config.yaml` file:
```yaml
# ===================================================================
# General & Data Source Configuration
# ===================================================================
# A general name for the YOLO model setup.
name: YOLO
# Path to the source video file for the pipeline.
data: video.mp4
# Directory where log files will be saved.
log-path: logs
debug-mode: False
control-count: 10

app:
  run_duration_seconds: 200 # Duration to run the pipeline in seconds.

# ===================================================================
# Server-Side Configuration
# ===================================================================
server:
  # Defines the model splitting point. Corresponds to pre-configured
  # splits in the server logic ('a', 'b', or 'c').
  cut-layer: a
  # Defines the number of client instances per layer.
  # e.g., [1, 1] means 1 client for layer 1, and 1 client for layer 2.
  clients:
    - 1
    - 1
  # Model file path
  model: yolov8n
  # Number of frames to process in a single batch.
  batch-frame: 20
  save-output: False

# ===================================================================
# Client-Side Configuration
# ===================================================================
client:
  # Target image size [width, height] for model input.
  imgsz: [640, 640]
  # Max items in the client's internal thread queues (input, output, ack = 2x).
  internal_queue_maxsize: 50
  # RabbitMQ consumer prefetch count for the IOThread.
  io_prefetch_count: 5
  # Seconds the IOThread waits before retrying a failed RabbitMQ connection.
  rabbit_retry_delay: 5
  # Timeout for Pika's process_data_events() in the IOThread (seconds).
  io_process_events_timeout: 0.1
  # Timeout for the IOThread when getting items from its output queue (seconds).
  io_output_q_timeout: 0.05


# ===================================================================
# Infrastructure Connection Details
# ===================================================================
# RabbitMQ message broker connection settings.
rabbit:
  address: rabbitmq
  username: "admin"
  password: "admin"
  virtual-host: "/"

# Redis server connection settings for the Claim Check pattern.
redis:
  host: redis
  port: 6379
  db: 0
```
This configuration is use for server and client.

### For Container Stack Deploy
```commandline
make start
make stop
```

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



