import time
import pika
from requests.auth import HTTPBasicAuth
import requests
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def delete_old_queues(address, username, password, virtual_host):
    encoded_vhost = requests.utils.quote(virtual_host, safe='')
    mgmt_url = f'http://{address}:15672/api/queues/{encoded_vhost}'
    logging.info(f"Cleanup: Get queue list from API: {mgmt_url}")

    try:
        response = requests.get(mgmt_url, auth=HTTPBasicAuth(username, password), timeout=10) 
        response.raise_for_status() 
        queues = response.json()
        logging.info(f"Cleanup: Successfully retrieved {len(queues)} queue(s) via API.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Cleanup: Error when connecting or getting queue list from Management API: {e}")
        return False 

    connection = None 
    deleted_count = 0
    try:
        credentials = pika.PlainCredentials(username, password)
        params = pika.ConnectionParameters(
            host=address,
            port=5672,
            virtual_host=virtual_host,
            credentials=credentials,
            heartbeat=30,
            blocked_connection_timeout=30
        )
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        logging.info("Cleanup: Connected AMQP for cleanup.")

        app_queue_prefixes = ("reply_", "intermediate_queue_", "result", "rpc_queue")

        for queue_info in queues:
            queue_name = queue_info.get('name')
            if not queue_name:
                continue

            if queue_name.startswith(app_queue_prefixes):
                logging.info(f"Cleanup: Prepare to delete application queue: {queue_name}")
                try:
                    channel.queue_delete(queue=queue_name)
                    deleted_count += 1
                    logging.info(f"Cleanup: Delete queue success: {queue_name}")
                except pika.exceptions.ChannelClosedByBroker as e:
                
                    logging.warning(f"Cleanup: Unable to delete queue {queue_name}. Broker closed channel: {e}. Queue may be in use or exclusive.")

                    if connection.is_open and (not channel or channel.is_closed):
                        try:
                            channel = connection.channel()
                            logging.info("Cleanup: Reopened channel after deletion error.")
                        except Exception as reopen_e:
                            logging.error(f"Cleanup: Unable to reopen channel: {reopen_e}. Stop cleaning.")
                            break # Thoát vòng lặp nếu không mở lại được channel
                except Exception as e:
                    logging.error(f"Cleanup: Unknown error while deleting queue {queue_name}: {e}")
                    if connection.is_open and (not channel or channel.is_closed):
                        try:
                           channel = connection.channel()
                           logging.info("Cleanup: Reopened channel after unknown error.")
                        except Exception as reopen_e:
                           logging.error(f"Cleanup: Unable to reopen channel: {reopen_e}. Stop cleaning.")
                           break

        logging.info(f"Cleanup: Finished cleaning. Number of queues deleted: {deleted_count}")
        return True

    except pika.exceptions.AMQPConnectionError as e:
        logging.error(f"Cleanup: AMQP connection error to clean up: {e}")
        return False
    except Exception as e:
        logging.error(f"Cleanup: Unknown error during AMQP cleanup: {e}")
        import traceback
        traceback.print_exc() 
        return False
    finally:

        if connection and connection.is_open:
            connection.close()
            logging.info("Cleanup: AMPQ connection closed.")

