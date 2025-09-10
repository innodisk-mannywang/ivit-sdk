"""
iVIT 2.0 SDK - Base Inference Module
提供推理的基礎抽象類別和配置
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class InferenceConfig(ABC):
    """Abstract base class for inference configurations."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the model for inference."""
        pass

    @abstractmethod
    def get_preprocessing(self) -> transforms.Compose:
        """Return the preprocessing transforms."""
        pass

    @abstractmethod
    def get_postprocessing(self) -> callable:
        """Return the postprocessing function."""
        pass


class BaseInference(ABC):
    """Base inference class for all AI vision tasks."""

    def __init__(self, config: InferenceConfig, device: str = "auto"):
        """
        Initialize the inference engine.
        
        Args:
            config: Inference configuration
            device: Device to run inference on ("auto", "cpu", "cuda", "cuda:0", etc.)
        """
        self.config = config
        self.device = self._setup_device(device)
        self.model = None
        self.preprocessing = None
        self.postprocessing = None
        
        # Initialize components
        self._setup_components()

    def _setup_device(self, device: str) -> torch.device:
        """Setup the device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        return torch.device(device)

    def _setup_components(self):
        """Setup model and preprocessing/postprocessing components."""
        self.model = self.config.get_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.preprocessing = self.config.get_preprocessing()
        self.postprocessing = self.config.get_postprocessing()

    def load_model(self, model_path: str):
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the model file
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        
        print(f"✅ Model loaded from: {model_path}")

    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess input image for inference.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed tensor
        """
        # Convert to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image format")
        
        # Apply preprocessing
        if self.preprocessing:
            image_tensor = self.preprocessing(image)
        else:
            # Default preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor.to(self.device)

    @abstractmethod
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Dictionary containing prediction results
        """
        pass

    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction results
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results

    def predict_from_folder(self, folder_path: str, extensions: List[str] = None) -> Dict[str, Any]:
        """
        Run inference on all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            extensions: List of file extensions to process (default: ['.jpg', '.jpeg', '.png'])
            
        Returns:
            Dictionary mapping filenames to prediction results
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        results = {}
        image_files = []
        
        # Find all image files
        for ext in extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))
        
        print(f"🔍 Found {len(image_files)} images in {folder_path}")
        
        # Process each image
        for image_file in image_files:
            try:
                result = self.predict(str(image_file))
                results[image_file.name] = result
                print(f"✅ Processed: {image_file.name}")
            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {str(e)}")
                results[image_file.name] = {"error": str(e)}
        
        return results

    def benchmark(self, image: Union[str, np.ndarray, Image.Image], num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            image: Input image for benchmarking
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary containing timing statistics
        """
        # Preprocess image once
        image_tensor = self.preprocess_image(image)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(image_tensor)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(image_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "fps": 1.0 / np.mean(times)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.config.get('model_name', 'unknown'),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }


