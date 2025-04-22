import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
video_path = "./video.mp4" # Dùng cùng file video

logging.info(f"Attempting to open: {video_path}")
cap = None # Khởi tạo là None
try:
    cap = cv2.VideoCapture(video_path)
    logging.info(f"VideoCapture object created: {cap}") # Log đối tượng tạo ra
    if cap is None or not cap.isOpened():
        logging.error("Failed to open video capture.")
    else:
        logging.info("Video capture opened successfully!")
        # Thử đọc một frame để chắc chắn
        ret, frame = cap.read()
        if ret:
            logging.info(f"Successfully read one frame with shape: {frame.shape}")
        else:
            logging.warning("Could not read first frame.")
except Exception as e:
    logging.error(f"An exception occurred during OpenCV test: {e}")
    import traceback
    traceback.print_exc()
finally:
    if cap is not None and cap.isOpened():
        cap.release()
        logging.info("Video capture released.")
    logging.info("OpenCV test finished.")