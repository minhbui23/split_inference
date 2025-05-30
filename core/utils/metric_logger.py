# core/utils/metrics_logger.py
import logging
import os
from collections import OrderedDict

class MetricsLogger:
    """A simple logger to write structured metrics to a dedicated file."""

    def __init__(self, file_path, fieldnames):
        """Initializes the MetricsLogger.

        Args:
            file_path (str): Path to the metrics log file (e.g., 'metrics.csv').
            fieldnames (list): A list of strings for the CSV header fields.
        """
        self.fieldnames = fieldnames
        self.logger = logging.getLogger('metrics_logger')
        self.logger.setLevel(logging.INFO)
        
        self.logger.propagate = False

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)

        # Write header if the file is new/empty
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            self.logger.info(','.join(self.fieldnames))

    def log(self, metrics_dict):
        """Logs a dictionary of metrics as a CSV row.

        Args:
            metrics_dict (dict): A dictionary containing metric values.
        """
        ordered_metrics = OrderedDict.fromkeys(self.fieldnames, 'N/A')
        ordered_metrics.update(metrics_dict)
        
        log_values = []
        for key in self.fieldnames:
            value = ordered_metrics[key]
            if isinstance(value, float):
                log_values.append(f"{value:.6f}")
            else:
                log_values.append(str(value))
                
        self.logger.info(','.join(log_values))