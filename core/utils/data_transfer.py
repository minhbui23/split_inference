# src/data_transfer_utils.py
import redis
import pickle
import uuid
import time
import torch
import pika
import json
from typing import Any, Dict, Optional
import zstandard

# You'll need to add 'redis' to your requirements.txt
# pip install redis

class RedisManager:
    """
    Handles connection and basic data operations (set/get/delete) with a Redis server.
    """
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, logger=None, password: Optional[str] = None):
        """
        Initializes the RedisManager.
        Args:
            host: Redis server host.
            port: Redis server port.
            db: Redis database number.
            logger: Logger instance.
            password: Password for Redis connection (if any).
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.logger = logger
        self.redis_client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self):
        """Establishes connection to the Redis server."""
        try:
            self.redis_client = redis.Redis(host=self.host, port=self.port, db=self.db, password=self.password, health_check_interval=30)
            self.redis_client.ping() # Verify connection
            self.logger.log_info(f"[RedisManager] Connected to Redis at {self.host}:{self.port}/{self.db}")
        except redis.exceptions.ConnectionError as e:
            self.logger.log_error(f"[RedisManager] Failed to connect to Redis: {e}")
            self.redis_client = None # Ensure client is None if connection failed
        except Exception as e: # Catch other potential errors like config issues
            self.logger.log_error(f"[RedisManager] An unexpected error occurred during Redis connection: {e}")
            self.redis_client = None

    def is_connected(self) -> bool:
        """Checks if the Redis client is connected."""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.ping()
            self.logger.log_info("[RedisManager] Redis client is initialized." )
        except redis.exceptions.ConnectionError:
            return False

    # For compress data before storing in Redis
    def store_raw_bytes(self, data_bytes: bytes, prefix: str = "data_chunk", ttl_seconds: Optional[int] = 300) -> Optional[str]:
        """
        Stores raw bytes directly in Redis.
        Args:
            data_bytes: The raw bytes to store.
            prefix: A prefix for the generated Redis key.
            ttl_seconds: Time-to-live for the key in seconds. If None, no TTL.
        Returns:
            The Redis key if successful, None otherwise.
        """
        if not self.is_connected():
            if self.logger: self.logger.log_error("[RedisManager] Cannot store raw bytes: Not connected to Redis.")
            return None
        
        redis_key = f"{prefix}:{uuid.uuid4()}"
        try:
            self.redis_client.set(redis_key, data_bytes)
            if ttl_seconds:
                self.redis_client.expire(redis_key, ttl_seconds)
            if self.logger:
                self.logger.log_info(f"[RedisManager] Stored raw bytes in Redis with key: {redis_key}, TTL: {ttl_seconds}s, Size: {len(data_bytes)} bytes")
            return redis_key
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"[RedisManager] Error storing raw bytes in Redis (key: {redis_key}): {e}")
            return None

    def retrieve_raw_bytes(self, redis_key: str) -> Optional[bytes]:
        """
        Retrieves raw bytes from Redis using the given key.
        Args:
            redis_key: The key to retrieve data from Redis.
        Returns:
            The raw bytes if successful, None otherwise.
        """
        if not self.is_connected():
            self.logger.log_error("[RedisManager] Cannot retrieve raw bytes: Not connected to Redis.")
            return None
        
        try:
            serialized_data = self.redis_client.get(redis_key)
            if serialized_data is None:
                self.logger.log_warning(f"[RedisManager] No raw bytes found in Redis for key: {redis_key}")
                return None
            self.logger.log_info(f"[RedisManager] Retrieved raw bytes from Redis for key: {redis_key}, Size: {len(serialized_data)} bytes")
            return serialized_data
        except Exception as e:
            self.logger.log_error(f"[RedisManager] Error retrieving raw bytes from Redis (key: {redis_key}): {e}")
            return None


    def delete_data(self, redis_key: str) -> bool:
        """Deletes data from Redis for a given key."""
        if not self.is_connected():
            self.logger.log_error("[RedisManager] Cannot delete data: Not connected to Redis.")
            return False
        try:
            result = self.redis_client.delete(redis_key)
            self.logger.log_info(f"[RedisManager] Deleted data from Redis for key: {redis_key}. Result: {result}")
            return result > 0
        except Exception as e:
            self.logger.log_error(f"[RedisManager] Error deleting data from Redis (key: {redis_key}): {e}")
            return False

    def close(self):
        """Closes the Redis connection."""
        if self.redis_client:
            try:
                self.redis_client.close()
                self.logger.log_info("[RedisManager] Redis connection closed.")
            except Exception as e:
                self.logger.log_error(f"[RedisManager] Error closing Redis connection: {e}")
        self.redis_client = None


class HybridDataTransfer:
    """
    High-level class to manage sending/receiving data using RabbitMQ for metadata
    and Redis for large data payloads (Claim Check pattern).
    """
    def __init__(self, pika_channel, redis_manager: RedisManager, logger,
                 default_ttl_seconds: int = 300,
                 metadata_content_type: str = 'application/json'):
        """
        Initializes the HybridDataTransfer.
        Args:
            pika_channel: An active Pika channel object for sending/receiving RabbitMQ messages.
            redis_manager: An initialized instance of RedisManager.
            logger: Logger instance.
            default_ttl_seconds: Default TTL for data stored in Redis.
            metadata_content_type: Content type for RabbitMQ metadata messages.
        """
        self.channel = pika_channel
        self.redis_manager = redis_manager
        self.logger = logger
        self.default_ttl_seconds = default_ttl_seconds
        self.metadata_content_type = metadata_content_type

        if not isinstance(self.redis_manager, RedisManager) or not self.redis_manager.is_connected():
            self.logger.log_error(f"[HybridDataTransfer] RedisManager is not provided or not connected. Data plane will fail.")           
            # Optionally raise an exception here if Redis is critical
            # raise ValueError("RedisManager must be provided and connected.")

    def _serialize_payload(self, payload: Any) -> bytes:
        """Serializes the payload to bytes (pickle or torch.save)."""
        if hasattr(torch, 'is_tensor') and \
           (torch.is_tensor(payload) or \
            (isinstance(payload, dict) and any(torch.is_tensor(v) for v in payload.values()))):
            import io
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            return buffer.getvalue()
        else:
            return pickle.dumps(payload)

    def _deserialize_payload(self, data_bytes: bytes) -> Any:
        """Deserializes bytes back to the original payload."""
        try:
            import io
            buffer = io.BytesIO(data_bytes)
            return torch.load(buffer)
        except Exception: # Fallback to pickle
            try:
                return pickle.loads(data_bytes)
            except Exception as e_deserialize:
                self.logger.log_error(f"[HybridDataTransfer] Failed to deserialize final payload: {e_deserialize}")
                return None

    def send_data(self, actual_data_payload: Any, rabbitmq_target_queue: str,
                  additional_metadata: Optional[Dict] = None,
                  redis_key_prefix: str = "data_chunk",
                  data_ttl_seconds: Optional[int] = None,
                  compress: bool = True,
                  compression_level: int = 3) -> bool:
        """
        Stores the actual_data_payload in Redis and sends metadata (including Redis key)
        to the specified RabbitMQ queue.
        Args:
            actual_data_payload: The large data (e.g., tensor dict from L1) to store in Redis.
            rabbitmq_target_queue: The RabbitMQ queue to send metadata to.
            additional_metadata: Optional dictionary of additional metadata to include.
            redis_key_prefix: Prefix for the generated Redis key.
            data_ttl_seconds: TTL for this specific data in Redis. Uses default if None.
        Returns:
            True if metadata was successfully published to RabbitMQ, False otherwise.
        """
        if not self.redis_manager or not self.redis_manager.is_connected():
            self.logger.log_error("[HybridDataTransfer] Cannot send data: RedisManager not connected.")
            return False
        if not self.channel or not self.channel.is_open:
            self.logger.log_error("[HybridDataTransfer] Cannot send data: Pika channel not open.")
            return False

        try:
            serialized_payload_bytes = self._serialize_payload(actual_data_payload)
        except Exception as e_serialize:
            self.logger.log_error(f"[HybridDataTransfer] Failed to serialize payload: {e_serialize}")
            return False
        
        original_size = len(serialized_payload_bytes)
        data_to_store_in_redis = serialized_payload_bytes
        is_compressed_flag = False

        if compress and serialized_payload_bytes:
            try:
                cctx = zstandard.ZstdCompressor(level=compression_level)
                compressed_bytes = cctx.compress(serialized_payload_bytes)
                data_to_store_in_redis = compressed_bytes
                is_compressed_flag = True
                compressed_size = len(compressed_bytes)
                self.logger.log_info(f"[HybridDataTransfer] Payload compressed with Zstd (L{compression_level}): "
                                     f"Original: {original_size} bytes, Compressed: {compressed_size} bytes. "
                                     f"Ratio: {original_size / compressed_size if compressed_size > 0 else float('inf'):.2f}x")
            except Exception as e_compress:
                self.logger.log_error(f"[HybridDataTransfer] Failed to compress payload, sending uncompressed. Error: {e_compress}")
             

        ttl_to_use = data_ttl_seconds if data_ttl_seconds is not None else self.default_ttl_seconds
        
        # 1. Store large data payload in Redis
        redis_key = self.redis_manager.store_raw_bytes(data_to_store_in_redis,
                                                         prefix=redis_key_prefix,
                                                         ttl_seconds=ttl_to_use)
        if not redis_key:
            self.logger.log_error(f"[HybridDataTransfer] Failed to store data in Redis for queue {rabbitmq_target_queue}.")
            return False

        # 2. Prepare metadata message for RabbitMQ
        metadata_message = {
            "redis_key": redis_key,
            "is_compressed": is_compressed_flag
        }
        if additional_metadata: # additional_metadata cũng phải chứa các kiểu tương thích JSON
            metadata_message.update(additional_metadata)
        
        # 3. Send metadata message to RabbitMQ
        try:
            # Serialize metadata sang chuỗi JSON, rồi encode thành bytes
            json_metadata_str = json.dumps(metadata_message)
            message_body_bytes = json_metadata_str.encode('utf-8') # Encode sang UTF-8 bytes
            
            self.channel.basic_publish(
                exchange='',
                routing_key=rabbitmq_target_queue,
                body=message_body_bytes, # Gửi bytes
                properties=pika.BasicProperties(
                    content_type=self.metadata_content_type
                )
            )
            log_compression_info = f"Compressed: {is_compressed_flag}"
            if is_compressed_flag:
                 log_compression_info += f" (Original: {original_size}B -> Stored: {len(data_to_store_in_redis)}B)"

            self.logger.log_info(f"[HybridDataTransfer] Sent metadata (Redis key: {redis_key}, {log_compression_info}) to RabbitMQ queue '{rabbitmq_target_queue}'.")
            
            return True
        except Exception as e:
            self.logger.log_error(f"[HybridDataTransfer] Error sending metadata to RabbitMQ queue '{rabbitmq_target_queue}': {e}")
            # Delete the already stored Redis data if RabbitMQ publish fails, to avoid orphaned data
            self.redis_manager.delete_data(redis_key) 
            return False

    def receive_data_from_metadata(self, metadata_message: Dict, delete_after_retrieval: bool = True) -> Any:
        """
        Retrieves the large data payload from Redis based on metadata received from RabbitMQ.
        Args:
            metadata_message: The deserialized metadata message (dict) from RabbitMQ,
                              containing at least "redis_key".
            delete_after_retrieval: If True, deletes the data from Redis after successful retrieval.
        Returns:
            The deserialized large data payload if successful, None otherwise.
        """
        if not self.redis_manager or not self.redis_manager.is_connected():
            self.logger.log_error("[HybridDataTransfer] Cannot retrieve data: RedisManager not connected.")
            return None
        if not isinstance(metadata_message, dict):
            self.logger.log_error(f"[HybridDataTransfer] Invalid metadata_message format: expected dict, got {type(metadata_message)}")
            return None

        redis_key = metadata_message.get("redis_key")
        if not redis_key:
            self.logger.log_error("[HybridDataTransfer] 'redis_key' not found in metadata_message.")
            return None

        retrieved_bytes = self.redis_manager.retrieve_raw_bytes(redis_key) # Sử dụng hàm mới

        if retrieved_bytes is None:
            self.logger.log_warning(f"[HybridDataTransfer] No data found in Redis for key {redis_key}.")
            return None

        payload_to_deserialize = retrieved_bytes
        was_compressed = metadata_message.get("is_compressed", False)

        if was_compressed:
            try:
                dctx = zstandard.ZstdDecompressor()
                decompressed_bytes = dctx.decompress(retrieved_bytes)
                payload_to_deserialize = decompressed_bytes
                self.logger.log_info(f"[HybridDataTransfer] Payload decompressed with Zstd for key {redis_key}: "
                                     f"Compressed: {len(retrieved_bytes)} bytes, Original: {len(decompressed_bytes)} bytes.")
            except Exception as e_decompress:
                self.logger.log_error(f"[HybridDataTransfer] Failed to decompress payload for key {redis_key}. "
                                     "Attempting to deserialize raw (possibly compressed) data. Error: {e_decompress}" )

        final_payload = self._deserialize_payload(payload_to_deserialize)

        if final_payload is not None and delete_after_retrieval:
            if not self.redis_manager.delete_data(redis_key):
                 self.logger.log_warning(f"[HybridDataTransfer] Failed to delete data from Redis for key {redis_key} after retrieval.")
        
        return final_payload