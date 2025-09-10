"""
iVIT 2.0 SDK - Classification Inference Module
提供圖像分類的推理功能
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Dict, Any, Optional, Union, List
import numpy as np
from PIL import Image

from ..core.base_inference import BaseInference, InferenceConfig
from ..trainer.classification import ClassificationConfig


class ClassificationInferenceConfig(InferenceConfig):
    """Configuration for classification inference."""

    def __init__(self, 
                 model_name: str = 'resnet18',
                 num_classes: int = 1000,
                 pretrained: bool = True,
                 img_size: int = 224,
                 class_names: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize classification inference configuration.
        
        Args:
            model_name: Model architecture name
            num_classes: Number of classes
            pretrained: Whether to use pretrained weights
            img_size: Input image size
            class_names: List of class names (optional)
        """
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            img_size=img_size,
            class_names=class_names,
            **kwargs
        )

    def get_model(self) -> nn.Module:
        """Get the classification model."""
        from ..trainer.classification import ClassificationConfig
        
        # Create classification config for model creation
        config = ClassificationConfig(
            model_name=self.config['model_name'],
            num_classes=self.config['num_classes'],
            pretrained=self.config['pretrained']
        )
        
        return config.get_model()

    def get_preprocessing(self) -> transforms.Compose:
        """Get preprocessing transforms."""
        img_size = self.config['img_size']
        
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_postprocessing(self) -> callable:
        """Get postprocessing function."""
        def postprocess(outputs: torch.Tensor, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(5, self.config['num_classes']))
            
            # Convert to CPU and numpy
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            
            # Prepare results
            results = {
                'predictions': [],
                'top_class': None,
                'top_confidence': 0.0
            }
            
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                class_name = class_names[idx] if class_names and idx < len(class_names) else f"Class_{idx}"
                
                prediction = {
                    'class_id': int(idx),
                    'class_name': class_name,
                    'confidence': float(prob),
                    'rank': i + 1
                }
                
                results['predictions'].append(prediction)
                
                if i == 0:  # Top prediction
                    results['top_class'] = class_name
                    results['top_confidence'] = float(prob)
            
            return results
        
        return postprocess


class ClassificationInference(BaseInference):
    """Classification inference engine."""

    def __init__(self, 
                 model_name: str = 'resnet18',
                 num_classes: int = 1000,
                 pretrained: bool = True,
                 img_size: int = 224,
                 class_names: Optional[List[str]] = None,
                 device: str = "auto",
                 **kwargs):
        """
        Initialize classification inference.
        
        Args:
            model_name: Model architecture name
            num_classes: Number of classes
            pretrained: Whether to use pretrained weights
            img_size: Input image size
            class_names: List of class names
            device: Device to run inference on
        """
        config = ClassificationInferenceConfig(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            img_size=img_size,
            class_names=class_names,
            **kwargs
        )
        
        super().__init__(config, device)
        self.class_names = class_names

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Run classification inference on a single image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Dictionary containing classification results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Postprocess results
        results = self.postprocessing(outputs, self.class_names)
        
        return results

    def predict_with_confidence_threshold(self, 
                                        image: Union[str, np.ndarray, Image.Image], 
                                        threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run classification with confidence threshold filtering.
        
        Args:
            image: Input image
            threshold: Confidence threshold for filtering predictions
            
        Returns:
            Dictionary containing filtered classification results
        """
        results = self.predict(image)
        
        # Filter predictions by confidence threshold
        filtered_predictions = [
            pred for pred in results['predictions'] 
            if pred['confidence'] >= threshold
        ]
        
        results['predictions'] = filtered_predictions
        
        # Update top prediction if it meets threshold
        if filtered_predictions:
            results['top_class'] = filtered_predictions[0]['class_name']
            results['top_confidence'] = filtered_predictions[0]['confidence']
        else:
            results['top_class'] = None
            results['top_confidence'] = 0.0
        
        return results

    def predict_top_k(self, 
                     image: Union[str, np.ndarray, Image.Image], 
                     k: int = 3) -> Dict[str, Any]:
        """
        Get top-k predictions.
        
        Args:
            image: Input image
            k: Number of top predictions to return
            
        Returns:
            Dictionary containing top-k classification results
        """
        results = self.predict(image)
        
        # Limit to top-k predictions
        results['predictions'] = results['predictions'][:k]
        
        return results

    def get_class_probabilities(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, float]:
        """
        Get probability distribution over all classes.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary mapping class names to probabilities
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Apply softmax
        probabilities = torch.softmax(outputs, dim=1)
        probabilities = probabilities.cpu().numpy()[0]
        
        # Create class name mapping
        if self.class_names:
            class_mapping = {i: name for i, name in enumerate(self.class_names)}
        else:
            class_mapping = {i: f"Class_{i}" for i in range(len(probabilities))}
        
        # Return probability distribution
        return {
            class_mapping[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }

    def compare_models(self, 
                      image: Union[str, np.ndarray, Image.Image], 
                      model_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare predictions from multiple models.
        
        Args:
            image: Input image
            model_paths: List of model file paths
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        results = {}
        
        # Save current model
        current_model = self.model
        
        for model_path in model_paths:
            try:
                # Load different model
                self.load_model(model_path)
                
                # Get prediction
                prediction = self.predict(image)
                results[model_path] = prediction
                
            except Exception as e:
                results[model_path] = {"error": str(e)}
        
        # Restore original model
        self.model = current_model
        
        return results
