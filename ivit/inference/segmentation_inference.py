"""
iVIT 2.0 SDK - Segmentation Inference Module
提供語義分割的推理功能
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


class SegmentationInferenceConfig(InferenceConfig):
    """Configuration for segmentation inference."""

    def __init__(self, 
                 model_name: str = 'yolov8n-seg.pt',
                 num_classes: int = 21,
                 img_size: int = 640,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 class_names: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize segmentation inference configuration.
        
        Args:
            model_name: Model name or path
            num_classes: Number of classes
            img_size: Input image size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            class_names: List of class names
        """
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            img_size=img_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names,
            **kwargs
        )

    def get_model(self):
        """Get the segmentation model (YOLO-seg)."""
        try:
            from ultralytics import YOLO
            return YOLO(self.config['model_name'])
        except ImportError:
            raise ImportError("ultralytics package is required for segmentation inference. Install with: pip install ultralytics")

    def get_preprocessing(self) -> None:
        """YOLO handles preprocessing internally."""
        return None

    def get_postprocessing(self) -> callable:
        """Get postprocessing function."""
        def postprocess(results, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
            if not results or len(results) == 0:
                return {
                    'masks': [],
                    'num_masks': 0,
                    'image_shape': None
                }
            
            # Get the first result (single image)
            result = results[0]
            
            # Extract masks and detections
            masks = []
            
            if result.masks is not None and result.boxes is not None:
                # Get mask data
                mask_data = result.masks.data.cpu().numpy()  # Shape: (N, H, W)
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (mask, box, conf, class_id) in enumerate(zip(mask_data, boxes, confidences, class_ids)):
                    class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class_{class_id}"
                    
                    # Convert mask to binary
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    
                    mask_info = {
                        'mask': binary_mask,
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
                        'area': float(np.sum(binary_mask))
                    }
                    
                    masks.append(mask_info)
            
            return {
                'masks': masks,
                'num_masks': len(masks),
                'image_shape': result.orig_shape if hasattr(result, 'orig_shape') else None
            }
        
        return postprocess


class SegmentationInference(BaseInference):
    """Segmentation inference engine using YOLO-seg."""

    def __init__(self, 
                 model_name: str = 'yolov8n-seg.pt',
                 num_classes: int = 21,
                 img_size: int = 640,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 class_names: Optional[List[str]] = None,
                 device: str = "auto",
                 **kwargs):
        """
        Initialize segmentation inference.
        
        Args:
            model_name: Model name or path
            num_classes: Number of classes
            img_size: Input image size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            class_names: List of class names
            device: Device to run inference on
        """
        config = SegmentationInferenceConfig(
            model_name=model_name,
            num_classes=num_classes,
            img_size=img_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names,
            **kwargs
        )
        
        super().__init__(config, device)
        self.class_names = class_names
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Run segmentation inference on a single image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Dictionary containing segmentation results
        """
        # Convert image to appropriate format for YOLO
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
        
        # Run YOLO segmentation inference
        results = self.model(
            image_input,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.config.config['img_size']
        )
        
        # Postprocess results
        segmentation_results = self.postprocessing(results, self.class_names)
        
        return segmentation_results

    def load_model(self, model_path: str):
        """
        Override base loader for YOLO-seg checkpoints.
        Ultralytics YOLO 分割模型需用 YOLO 類別載入，避免 torch.load 觸發安全限制。
        """
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_path)
            # 重新包裝，保持 BaseInference 介面一致
            self.model = YOLOWrapper(self.yolo_model)
            # 若未提供 class_names，嘗試從 YOLO 權重帶出
            if not self.class_names and hasattr(self.yolo_model, 'names'):
                names = self.yolo_model.names
                if isinstance(names, dict):
                    # Ultralytics 有時以 {id: name} 字典提供
                    self.class_names = [names[i] for i in sorted(names.keys())]
                elif isinstance(names, list):
                    self.class_names = names
                else:
                    self.class_names = None
            print(f"✅ YOLO-seg weights loaded from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO-seg weights: {e}")

    def predict_with_custom_thresholds(self, 
                                     image: Union[str, np.ndarray, Image.Image], 
                                     conf_threshold: float = None,
                                     iou_threshold: float = None) -> Dict[str, Any]:
        """
        Run segmentation with custom confidence and IoU thresholds.
        
        Args:
            image: Input image
            conf_threshold: Custom confidence threshold
            iou_threshold: Custom IoU threshold
            
        Returns:
            Dictionary containing segmentation results
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
        
        # Run YOLO segmentation inference with custom thresholds
        results = self.model(
            image_input,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=self.config.config['img_size']
        )
        
        # Postprocess results
        segmentation_results = self.postprocessing(results, self.class_names)
        
        return segmentation_results

    def create_colored_mask(self, 
                           image: Union[str, np.ndarray, Image.Image], 
                           masks: List[Dict[str, Any]],
                           alpha: float = 0.5) -> np.ndarray:
        """
        Create colored mask overlay on image.
        
        Args:
            image: Input image
            masks: List of mask information
            alpha: Transparency for overlay
            
        Returns:
            Image with colored mask overlay
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
        
        # Create colored overlay
        overlay = img.copy()
        
        # Define colors for different classes
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        
        for i, mask_info in enumerate(masks):
            mask = mask_info['mask']
            class_id = mask_info['class_id']
            color = colors[class_id % len(colors)]
            
            # Create colored mask
            colored_mask = np.zeros_like(img)
            # 將遮罩大小對齊輸入影像尺寸
            if mask.shape[:2] != img.shape[:2]:
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_resized = mask
            colored_mask[mask_resized > 0] = color
            
            # Blend with overlay
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
        
        # Blend original image with overlay
        result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        return result

    def create_class_mask(self, 
                         image_shape: tuple, 
                         masks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create class-wise segmentation mask.
        
        Args:
            image_shape: (height, width) of the image
            masks: List of mask information
            
        Returns:
            Class mask with shape (height, width) where each pixel contains class_id
        """
        height, width = image_shape[:2]
        class_mask = np.zeros((height, width), dtype=np.int32)
        
        for mask_info in masks:
            mask = mask_info['mask']
            class_id = mask_info['class_id']
            
            # Resize mask to original image size if needed
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Set class ID where mask is present
            class_mask[mask > 0] = class_id
        
        return class_mask

    def get_mask_statistics(self, masks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about segmentation masks.
        
        Args:
            masks: List of mask information
            
        Returns:
            Dictionary containing mask statistics
        """
        if not masks:
            return {
                'total_masks': 0,
                'class_counts': {},
                'area_stats': {},
                'confidence_stats': {}
            }
        
        # Count masks by class
        class_counts = {}
        areas = []
        confidences = []
        
        for mask_info in masks:
            class_name = mask_info['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            areas.append(mask_info['area'])
            confidences.append(mask_info['confidence'])
        
        # Calculate statistics
        areas = np.array(areas)
        confidences = np.array(confidences)
        
        return {
            'total_masks': len(masks),
            'class_counts': class_counts,
            'area_stats': {
                'mean': float(np.mean(areas)),
                'std': float(np.std(areas)),
                'min': float(np.min(areas)),
                'max': float(np.max(areas))
            },
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }
        }

    def filter_masks_by_class(self, 
                            masks: List[Dict[str, Any]], 
                            target_classes: List[int]) -> List[Dict[str, Any]]:
        """
        Filter masks by target classes.
        
        Args:
            masks: List of mask information
            target_classes: List of class IDs to keep
            
        Returns:
            Filtered list of mask information
        """
        return [
            mask for mask in masks 
            if mask['class_id'] in target_classes
        ]

    def filter_masks_by_area(self, 
                           masks: List[Dict[str, Any]], 
                           min_area: float = None,
                           max_area: float = None) -> List[Dict[str, Any]]:
        """
        Filter masks by area.
        
        Args:
            masks: List of mask information
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            
        Returns:
            Filtered list of mask information
        """
        filtered_masks = []
        
        for mask in masks:
            area = mask['area']
            
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
                
            filtered_masks.append(mask)
        
        return filtered_masks

    def merge_masks(self, 
                   masks: List[Dict[str, Any]], 
                   merge_classes: List[int] = None) -> np.ndarray:
        """
        Merge multiple masks into a single mask.
        
        Args:
            masks: List of mask information
            merge_classes: List of class IDs to merge (None for all)
            
        Returns:
            Merged mask
        """
        if not masks:
            return np.array([])
        
        # Filter masks by classes if specified
        if merge_classes is not None:
            masks = [mask for mask in masks if mask['class_id'] in merge_classes]
        
        if not masks:
            return np.array([])
        
        # Get image shape from first mask
        image_shape = masks[0]['mask'].shape
        
        # Create merged mask
        merged_mask = np.zeros(image_shape, dtype=np.uint8)
        
        for mask_info in masks:
            mask = mask_info['mask']
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)
        
        return merged_mask

    def extract_contours(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract contours from masks.
        
        Args:
            masks: List of mask information
            
        Returns:
            List of contour information
        """
        contours_info = []
        
        for mask_info in masks:
            mask = mask_info['mask']
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if area > 0:  # Only include non-empty contours
                    contour_info = {
                        'contour': contour,
                        'area': float(area),
                        'perimeter': float(perimeter),
                        'class_id': mask_info['class_id'],
                        'class_name': mask_info['class_name'],
                        'confidence': mask_info['confidence']
                    }
                    contours_info.append(contour_info)
        
        return contours_info
