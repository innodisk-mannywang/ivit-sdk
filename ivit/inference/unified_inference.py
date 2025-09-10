"""
iVIT 2.0 SDK - Unified Inference Module
提供統一的推理接口，支援所有任務類型
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from ..core.base_inference import BaseInference
from .classification_inference import ClassificationInference
from .detection_inference import DetectionInference
from .segmentation_inference import SegmentationInference


class UnifiedInference:
    """統一推理接口，支援分類、偵測、分割任務"""

    def __init__(self, device: str = "auto"):
        """
        Initialize unified inference interface.
        
        Args:
            device: Device to run inference on
        """
        self.device = device
        self.inference_engines = {}
        self.current_task = None

    def load_model(self, 
                  task_type: str, 
                  model_path: str = None, 
                  **kwargs) -> BaseInference:
        """
        Load a trained model for specific task.
        
        Args:
            task_type: Task type ('classification', 'detection', 'segmentation')
            model_path: Path to the model file
            **kwargs: Additional parameters for the inference engine
            
        Returns:
            Loaded inference engine
        """
        if task_type not in ['classification', 'detection', 'segmentation']:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Create inference engine based on task type
        if task_type == 'classification':
            engine = ClassificationInference(device=self.device, **kwargs)
        elif task_type == 'detection':
            engine = DetectionInference(device=self.device, **kwargs)
        elif task_type == 'segmentation':
            engine = SegmentationInference(device=self.device, **kwargs)
        
        # Load model (if path provided)
        if model_path:
            engine.load_model(model_path)
        
        # Store engine
        self.inference_engines[task_type] = engine
        self.current_task = task_type
        
        print(f"✅ {task_type} model loaded from: {model_path}")
        return engine

    def predict(self, 
               image: Union[str, List, np.ndarray], 
               task_type: str = None,
               **kwargs) -> Dict[str, Any]:
        """
        Run inference on image(s).
        
        Args:
            image: Input image(s) (path, numpy array, PIL Image, or list)
            task_type: Task type (if None, uses current task)
            **kwargs: Additional parameters for prediction
            
        Returns:
            Prediction results
        """
        # Determine task type
        if task_type is None:
            if self.current_task is None:
                raise ValueError("No task type specified and no current task set")
            task_type = self.current_task
        
        # Get inference engine
        if task_type not in self.inference_engines:
            raise ValueError(f"No {task_type} model loaded. Use load_model() first.")
        
        engine = self.inference_engines[task_type]
        
        # Handle batch prediction
        if isinstance(image, list):
            return engine.predict_batch(image, **kwargs)
        else:
            return engine.predict(image, **kwargs)

    def predict_classification(self, 
                             image: Union[str, np.ndarray], 
                             model_path: str = None,
                             class_names: List[str] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Run classification inference.
        
        Args:
            image: Input image
            model_path: Path to classification model
            class_names: List of class names
            **kwargs: Additional parameters
            
        Returns:
            Classification results
        """
        if 'classification' not in self.inference_engines and model_path:
            self.load_model('classification', model_path, class_names=class_names)
        
        return self.predict(image, 'classification', **kwargs)

    def predict_detection(self, 
                        image: Union[str, np.ndarray], 
                        model_path: str = None,
                        class_names: List[str] = None,
                        conf_threshold: float = 0.25,
                        **kwargs) -> Dict[str, Any]:
        """
        Run detection inference.
        
        Args:
            image: Input image
            model_path: Path to detection model
            class_names: List of class names
            conf_threshold: Confidence threshold
            **kwargs: Additional parameters
            
        Returns:
            Detection results
        """
        if 'detection' not in self.inference_engines and model_path:
            self.load_model('detection', model_path, class_names=class_names)
        
        return self.predict(image, 'detection', conf_threshold=conf_threshold, **kwargs)

    def predict_segmentation(self, 
                           image: Union[str, np.ndarray], 
                           model_path: str = None,
                           class_names: List[str] = None,
                           conf_threshold: float = 0.25,
                           **kwargs) -> Dict[str, Any]:
        """
        Run segmentation inference.
        
        Args:
            image: Input image
            model_path: Path to segmentation model
            class_names: List of class names
            conf_threshold: Confidence threshold
            **kwargs: Additional parameters
            
        Returns:
            Segmentation results
        """
        if 'segmentation' not in self.inference_engines and model_path:
            self.load_model('segmentation', model_path, class_names=class_names)
        
        return self.predict(image, 'segmentation', conf_threshold=conf_threshold, **kwargs)

    def predict_multi_task(self, 
                          image: Union[str, np.ndarray], 
                          tasks: List[str],
                          **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple task inference on the same image.
        
        Args:
            image: Input image
            tasks: List of task types to run
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping task types to their results
        """
        results = {}
        
        for task in tasks:
            if task in self.inference_engines:
                try:
                    result = self.predict(image, task, **kwargs)
                    results[task] = result
                except Exception as e:
                    results[task] = {"error": str(e)}
            else:
                results[task] = {"error": f"No {task} model loaded"}
        
        return results

    def benchmark_all_models(self, 
                           image: Union[str, np.ndarray], 
                           num_runs: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Benchmark all loaded models.
        
        Args:
            image: Input image for benchmarking
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary mapping task types to their benchmark results
        """
        results = {}
        
        for task_type, engine in self.inference_engines.items():
            try:
                benchmark_result = engine.benchmark(image, num_runs)
                results[task_type] = benchmark_result
            except Exception as e:
                results[task_type] = {"error": str(e)}
        
        return results

    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models."""
        info = {}
        
        for task_type, engine in self.inference_engines.items():
            try:
                model_info = engine.get_model_info()
                info[task_type] = model_info
            except Exception as e:
                info[task_type] = {"error": str(e)}
        
        return info

    def predict_from_folder(self, 
                          folder_path: str, 
                          task_type: str = None,
                          output_dir: str = None,
                          save_results: bool = False,
                          **kwargs) -> Dict[str, Any]:
        """
        Run inference on all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            task_type: Task type (if None, uses current task)
            output_dir: Directory to save results
            save_results: Whether to save results to files
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping filenames to results
        """
        # Determine task type
        if task_type is None:
            if self.current_task is None:
                raise ValueError("No task type specified and no current task set")
            task_type = self.current_task
        
        # Get inference engine
        if task_type not in self.inference_engines:
            raise ValueError(f"No {task_type} model loaded. Use load_model() first.")
        
        engine = self.inference_engines[task_type]
        
        # Run inference on folder
        results = engine.predict_from_folder(folder_path, **kwargs)
        
        # Save results if requested
        if save_results and output_dir:
            self._save_results(results, output_dir, task_type)
        
        return results

    def _save_results(self, 
                     results: Dict[str, Any], 
                     output_dir: str, 
                     task_type: str):
        """Save inference results to files."""
        import json
        from datetime import datetime
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_type}_results_{timestamp}.json"
        filepath = output_path / filename
        
        # Save results
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")

    def switch_task(self, task_type: str):
        """
        Switch current task type.
        
        Args:
            task_type: Task type to switch to
        """
        if task_type not in self.inference_engines:
            raise ValueError(f"No {task_type} model loaded. Use load_model() first.")
        
        self.current_task = task_type
        print(f"🔄 Switched to {task_type} task")

    def list_loaded_models(self) -> List[str]:
        """List all loaded model types."""
        return list(self.inference_engines.keys())

    def unload_model(self, task_type: str):
        """
        Unload a specific model.
        
        Args:
            task_type: Task type to unload
        """
        if task_type in self.inference_engines:
            del self.inference_engines[task_type]
            if self.current_task == task_type:
                self.current_task = None
            print(f"🗑️ Unloaded {task_type} model")
        else:
            print(f"⚠️ No {task_type} model loaded")

    def clear_all_models(self):
        """Clear all loaded models."""
        self.inference_engines.clear()
        self.current_task = None
        print("🗑️ Cleared all models")
