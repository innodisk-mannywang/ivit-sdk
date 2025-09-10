"""
iVIT 2.0 SDK - Detection Inference Module
提供物件偵測的推理功能
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Dict, Any, Optional, Union, List
import numpy as np
from PIL import Image
import cv2

from ..core.base_inference import BaseInference, InferenceConfig
from ..core.yolo_wrapper import YOLOWrapper


class DetectionInferenceConfig(InferenceConfig):
    """Configuration for detection inference."""

    def __init__(self, 
                 model_name: str = 'yolov8n.pt',
                 img_size: int = 640,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 class_names: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize detection inference configuration.
        
        Args:
            model_name: Model name or path
            img_size: Input image size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            class_names: List of class names
        """
        super().__init__(
            model_name=model_name,
            img_size=img_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names,
            **kwargs
        )

    def get_model(self):
        """Get the detection model (YOLO)."""
        try:
            from ultralytics import YOLO
            return YOLO(self.config['model_name'])
        except ImportError:
            raise ImportError("ultralytics package is required for detection inference. Install with: pip install ultralytics")

    def get_preprocessing(self) -> None:
        """YOLO handles preprocessing internally."""
        return None

    def get_postprocessing(self) -> callable:
        """Get postprocessing function."""
        def postprocess(results, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
            if not results or len(results) == 0:
                return {
                    'detections': [],
                    'num_detections': 0,
                    'image_shape': None
                }
            
            # Get the first result (single image)
            result = results[0]
            
            # Extract detections
            detections = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class_{class_id}"
                    
                    detection = {
                        'bbox': {
                            'x1': float(box[0]),
                            'y1': float(box[1]),
                            'x2': float(box[2]),
                            'y2': float(box[3]),
                            'width': float(box[2] - box[0]),
                            'height': float(box[3] - box[1])
                        },
                        'confidence': float(conf),
                        'class_id': int(class_id),
                        'class_name': class_name,
                        'area': float((box[2] - box[0]) * (box[3] - box[1]))
                    }
                    
                    detections.append(detection)
            
            return {
                'detections': detections,
                'num_detections': len(detections),
                'image_shape': result.orig_shape if hasattr(result, 'orig_shape') else None
            }
        
        return postprocess


class DetectionInference(BaseInference):
    """Detection inference engine using YOLO."""

    def __init__(self, 
                 model_name: str = 'yolov8n.pt',
                 img_size: int = 640,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 class_names: Optional[List[str]] = None,
                 device: str = "auto",
                 **kwargs):
        """
        Initialize detection inference.
        
        Args:
            model_name: Model name or path
            img_size: Input image size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            class_names: List of class names
            device: Device to run inference on
        """
        config = DetectionInferenceConfig(
            model_name=model_name,
            img_size=img_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names,
            **kwargs
        )
        
        super().__init__(config, device)
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Run detection inference on a single image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Dictionary containing detection results
        """
        # Convert image to appropriate format for YOLO
        if isinstance(image, str):
            # YOLO can handle file paths directly
            image_input = image
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_input = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_input = image
        else:
            raise ValueError("Unsupported image format")
        
        # Run YOLO inference
        results = self.model(
            image_input,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.config.config['img_size']
        )
        
        # Postprocess results
        detection_results = self.postprocessing(results, self.class_names)
        
        return detection_results

    def load_model(self, model_path: str):
        """
        Override base loader for YOLO checkpoints.
        Ultralytics YOLO 模型需使用 YOLO 自帶的載入方式，不能用 torch.load 直接讀取。
        """
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_path)
            # 重新包裝為 wrapper，保持與 BaseInference 介面一致
            self.model = YOLOWrapper(self.yolo_model)
            print(f"✅ YOLO weights loaded from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO weights: {e}")

    def predict_with_custom_thresholds(self, 
                                     image: Union[str, np.ndarray, Image.Image], 
                                     conf_threshold: float = None,
                                     iou_threshold: float = None) -> Dict[str, Any]:
        """
        Run detection with custom confidence and IoU thresholds.
        
        Args:
            image: Input image
            conf_threshold: Custom confidence threshold
            iou_threshold: Custom IoU threshold
            
        Returns:
            Dictionary containing detection results
        """
        # Use custom thresholds or defaults
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou_threshold
        
        # Convert image to appropriate format
        if isinstance(image, str):
            image_input = image
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_input = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_input = image
        else:
            raise ValueError("Unsupported image format")
        
        # Run YOLO inference with custom thresholds
        results = self.model(
            image_input,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=self.config.config['img_size']
        )
        
        # Postprocess results
        detection_results = self.postprocessing(results, self.class_names)
        
        return detection_results

    def predict_class_specific(self, 
                             image: Union[str, np.ndarray, Image.Image], 
                             target_classes: List[int]) -> Dict[str, Any]:
        """
        Run detection for specific classes only.
        
        Args:
            image: Input image
            target_classes: List of class IDs to detect
            
        Returns:
            Dictionary containing filtered detection results
        """
        # Get all detections
        results = self.predict(image)
        
        # Filter by target classes
        filtered_detections = [
            det for det in results['detections'] 
            if det['class_id'] in target_classes
        ]
        
        results['detections'] = filtered_detections
        results['num_detections'] = len(filtered_detections)
        
        return results

    def draw_detections(self, 
                       image: Union[str, np.ndarray, Image.Image], 
                       detections: Dict[str, Any],
                       thickness: int = 2,
                       font_scale: float = 0.5) -> np.ndarray:
        """
        Draw detection boxes on image.
        
        Args:
            image: Input image
            detections: Detection results
            thickness: Box thickness
            font_scale: Font scale for labels
            
        Returns:
            Image with drawn detections
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported image format")
        
        # Draw detections
        for detection in detections['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Draw label background
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        return img

    def get_detection_statistics(self, detections: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about detections.
        
        Args:
            detections: Detection results
            
        Returns:
            Dictionary containing detection statistics
        """
        if not detections['detections']:
            return {
                'total_detections': 0,
                'class_counts': {},
                'confidence_stats': {},
                'size_stats': {}
            }
        
        # Count detections by class
        class_counts = {}
        confidences = []
        areas = []
        
        for detection in detections['detections']:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(detection['confidence'])
            areas.append(detection['area'])
        
        # Calculate statistics
        confidences = np.array(confidences)
        areas = np.array(areas)
        
        return {
            'total_detections': len(detections['detections']),
            'class_counts': class_counts,
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            },
            'size_stats': {
                'mean_area': float(np.mean(areas)),
                'std_area': float(np.std(areas)),
                'min_area': float(np.min(areas)),
                'max_area': float(np.max(areas))
            }
        }

    def filter_by_confidence(self, 
                           detections: Dict[str, Any], 
                           min_confidence: float) -> Dict[str, Any]:
        """
        Filter detections by minimum confidence.
        
        Args:
            detections: Detection results
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered detection results
        """
        filtered_detections = [
            det for det in detections['detections'] 
            if det['confidence'] >= min_confidence
        ]
        
        return {
            'detections': filtered_detections,
            'num_detections': len(filtered_detections),
            'image_shape': detections.get('image_shape')
        }

    def filter_by_size(self, 
                      detections: Dict[str, Any], 
                      min_area: float = None,
                      max_area: float = None) -> Dict[str, Any]:
        """
        Filter detections by bounding box size.
        
        Args:
            detections: Detection results
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            
        Returns:
            Filtered detection results
        """
        filtered_detections = []
        
        for det in detections['detections']:
            area = det['area']
            
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
                
            filtered_detections.append(det)
        
        return {
            'detections': filtered_detections,
            'num_detections': len(filtered_detections),
            'image_shape': detections.get('image_shape')
        }
