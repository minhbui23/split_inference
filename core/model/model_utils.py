# src/model_utils.py
import os
import torch
from ultralytics import YOLO
from core.model.model import SplitDetectionModel, SplitDetectionPredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor 

def setup_inference_components(initial_params, device, logger):
    """
    Load the model, initialize SplitDetectionModel and predictor.
    Return (model_obj_for_worker, predictor_obj_for_worker) or (None, None) on error.
    """
    model_file_path = initial_params.get("model_save_path")
    splits_config = initial_params.get("splits") # This is the split_layer index for this client
    imgsz_tuple = initial_params.get("imgsz")

    if not (model_file_path and os.path.exists(model_file_path)):
        logger.log_error(f"[ModelUtils] Model file '{model_file_path}' not found or not specified.")
        return None, None
    if splits_config is None: 
        logger.log_error("[ModelUtils] Split index ('splits') not provided by server.")
        return None, None

    try:
        logger.log_info(f"[ModelUtils] Loading pre-trained model from: {model_file_path}")
        yolo_full_object = YOLO(model_file_path)
        pretrain_nn_module = yolo_full_object.model
        logger.log_info(f"[ModelUtils] Pre-trained nn.Module type: {type(pretrain_nn_module)}")

        logger.log_info(f"[ModelUtils] Initializing SplitDetectionModel with split_layer: {splits_config}")
        model_obj_for_worker = SplitDetectionModel(pretrain_nn_module,
                                                   split_layer=splits_config)
        model_obj_for_worker.to(device)
        model_obj_for_worker.eval() 
        logger.log_info(f"[ModelUtils] SplitDetectionModel initialized. Type: {type(model_obj_for_worker)}.")

        predictor_overrides = {
            "imgsz": imgsz_tuple,
            "device": device
        }
        predictor_obj_for_worker = SplitDetectionPredictor(model=model_obj_for_worker, overrides=predictor_overrides)
        

        logger.log_info(f"[ModelUtils] Predictor object initialized. Type: {type(predictor_obj_for_worker)}")
        return model_obj_for_worker, predictor_obj_for_worker

    except Exception as e:
        logger.log_error(f"[ModelUtils] Error during model/predictor initialization: {e}", exc_info=True)
        return None, None