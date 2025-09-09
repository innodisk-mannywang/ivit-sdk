"""
Dataset Analyzer Module
======================
Analyze datasets and extract statistics for intelligent recommendations.
"""

import os
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image
import pandas as pd


class DatasetAnalyzer:
    """Analyze datasets and extract statistical information."""

    def __init__(self):
        """Initialize DatasetAnalyzer."""
        print("🔍 DatasetAnalyzer initialized")

    def extract_statistics(self, dataset_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive statistics from a classification dataset.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary containing dataset statistics
        """
        dataset_path = Path(dataset_path)
        print(f"📊 Analyzing dataset: {dataset_path}")

        stats = {
            'dataset_path': str(dataset_path),
            'task_type': 'classification',
            'total_samples': 0,
            'num_classes': 0,
            'class_distribution': {},
            'image_stats': {
                'mean_width': 0,
                'mean_height': 0,
                'min_width': float('inf'),
                'max_width': 0,
                'min_height': float('inf'),
                'max_height': 0,
                'mean_channels': 0,
                'file_formats': set()
            },
            'data_quality': {
                'corrupted_files': [],
                'duplicate_files': [],
                'size_anomalies': []
            }
        }

        # Find image directories
        image_dirs = []

        # Check for standard structure (train/val splits)
        if (dataset_path / 'train').exists():
            image_dirs.append(dataset_path / 'train')
        if (dataset_path / 'val').exists():
            image_dirs.append(dataset_path / 'val')
        if (dataset_path / 'test').exists():
            image_dirs.append(dataset_path / 'test')

        # If no standard splits, use root directory
        if not image_dirs:
            image_dirs = [dataset_path]

        all_image_paths = []
        class_counts = {}

        # Collect all images
        for img_dir in image_dirs:
            if not img_dir.exists():
                continue

            # For each subdirectory (class)
            for class_dir in img_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    class_count = 0

                    # Get image files
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                    for ext in image_extensions:
                        class_images = list(class_dir.glob(f'*{ext}')) + list(class_dir.glob(f'*{ext.upper()}'))
                        class_count += len(class_images)
                        all_image_paths.extend(class_images)

                    if class_count > 0:
                        class_counts[class_name] = class_counts.get(class_name, 0) + class_count

        # Update basic stats
        stats['total_samples'] = len(all_image_paths)
        stats['num_classes'] = len(class_counts)
        stats['class_distribution'] = class_counts

        # Analyze image properties
        if all_image_paths:
            widths, heights, channels = [], [], []
            file_hashes = set()

            for i, img_path in enumerate(all_image_paths[:1000]):  # Sample first 1000 images
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        stats['data_quality']['corrupted_files'].append(str(img_path))
                        continue

                    h, w = img.shape[:2]
                    c = img.shape[2] if len(img.shape) == 3 else 1

                    widths.append(w)
                    heights.append(h)
                    channels.append(c)

                    # Check file format
                    stats['image_stats']['file_formats'].add(img_path.suffix.lower())

                    # Simple duplicate detection (by file size)
                    file_size = img_path.stat().st_size
                    if file_size in file_hashes:
                        stats['data_quality']['duplicate_files'].append(str(img_path))
                    else:
                        file_hashes.add(file_size)

                    # Detect size anomalies
                    if w < 32 or h < 32:
                        stats['data_quality']['size_anomalies'].append(str(img_path))

                except Exception as e:
                    stats['data_quality']['corrupted_files'].append(str(img_path))

                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{min(1000, len(all_image_paths))} images...")

            # Calculate statistics
            if widths:
                stats['image_stats'].update({
                    'mean_width': np.mean(widths),
                    'mean_height': np.mean(heights),
                    'min_width': min(widths),
                    'max_width': max(widths),
                    'min_height': min(heights),
                    'max_height': max(heights),
                    'mean_channels': np.mean(channels),
                    'file_formats': list(stats['image_stats']['file_formats'])
                })

        # Calculate additional metrics
        stats['class_balance_ratio'] = self._calculate_balance_ratio(class_counts)
        stats['dataset_size_category'] = self._categorize_dataset_size(stats['total_samples'])
        stats['complexity_score'] = self._calculate_complexity_score(stats)

        print(f"✅ Dataset analysis completed:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Number of classes: {stats['num_classes']}")
        print(f"   Mean image size: {stats['image_stats']['mean_width']:.0f}x{stats['image_stats']['mean_height']:.0f}")
        print(f"   Dataset complexity: {stats['complexity_score']:.2f}")

        return stats

    def analyze_classification_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Analyze classification dataset and return statistics.
        This is an alias for extract_statistics for backward compatibility.
        """
        return self.extract_statistics(dataset_path)

    def analyze_yolo_dataset(self, data_yaml_path: str) -> Dict[str, Any]:
        """
        Analyze YOLO dataset and return statistics.
        This is an alias for extract_detection_statistics for backward compatibility.
        """
        return self.extract_detection_statistics(data_yaml_path)

    def extract_detection_statistics(self, dataset_path: str) -> Dict[str, Any]:
        """Extract statistics from a YOLO detection dataset."""
        dataset_path = Path(dataset_path)
        print(f"📊 Analyzing YOLO detection dataset: {dataset_path}")

        stats = {
            'dataset_path': str(dataset_path),
            'task_type': 'detection',
            'total_images': 0,
            'total_annotations': 0,
            'num_classes': 0,
            'class_distribution': {},
            'bbox_stats': {
                'mean_boxes_per_image': 0,
                'mean_box_area': 0,
                'small_objects': 0,
                'medium_objects': 0,
                'large_objects': 0
            },
            'image_stats': {
                'mean_width': 0,
                'mean_height': 0,
                'aspect_ratios': []
            }
        }

        # Check dataset format
        data_yaml_path = dataset_path / 'data.yaml'
        if not data_yaml_path.exists():
            print("❌ No data.yaml found. Analyzing directory structure...")
        else:
            # Load data.yaml to get correct paths
            try:
                import yaml
                with open(data_yaml_path, 'r') as f:
                    data_config = yaml.safe_load(f)
                
                # Get class information
                stats['num_classes'] = data_config.get('nc', 0)
                class_names = data_config.get('names', [])
                if isinstance(class_names, list):
                    stats['class_distribution'] = {i: name for i, name in enumerate(class_names)}
                
                print(f"✅ Loaded data.yaml: {stats['num_classes']} classes")
                print(f"   Classes: {class_names}")
                
            except Exception as e:
                print(f"⚠️  Could not load data.yaml: {e}")

        # Analyze images and labels - try both directory structures
        splits = ['train', 'val', 'test']
        all_image_files = []
        all_label_files = []

        for split in splits:
            # Try structure 1: dataset/images/split/ and dataset/labels/split/
            img_dir1 = dataset_path / 'images' / split
            label_dir1 = dataset_path / 'labels' / split
            
            # Try structure 2: dataset/split/images/ and dataset/split/labels/
            img_dir2 = dataset_path / split / 'images'
            label_dir2 = dataset_path / split / 'labels'
            
            # Try structure 3: dataset/split/ (images and labels in same directory)
            img_dir3 = dataset_path / split
            label_dir3 = dataset_path / split

            # Check which structure exists
            img_dir = None
            label_dir = None
            
            if img_dir1.exists():
                img_dir = img_dir1
                label_dir = label_dir1
                print(f"📁 Found structure 1: {img_dir}")
            elif img_dir2.exists():
                img_dir = img_dir2
                label_dir = label_dir2
                print(f"📁 Found structure 2: {img_dir}")
            elif img_dir3.exists():
                img_dir = img_dir3
                label_dir = img_dir3
                print(f"📁 Found structure 3: {img_dir}")

            if img_dir and img_dir.exists():
                for ext in ['.jpg', '.jpeg', '.png']:
                    all_image_files.extend(list(img_dir.glob(f'*{ext}')))

            if label_dir and label_dir.exists():
                all_label_files.extend(list(label_dir.glob('*.txt')))

        stats['total_images'] = len(all_image_files)

        # Analyze annotations
        class_counts = {}
        total_boxes = 0
        box_areas = []

        for label_file in all_label_files[:500]:  # Sample first 500 labels
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])

                        # Count classes
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1

                        # Calculate box area
                        area = w * h
                        box_areas.append(area)
                        total_boxes += 1

            except Exception as e:
                continue

        stats['total_annotations'] = total_boxes
        stats['num_classes'] = len(class_counts)
        stats['class_distribution'] = class_counts

        if box_areas:
            stats['bbox_stats']['mean_box_area'] = np.mean(box_areas)
            stats['bbox_stats']['mean_boxes_per_image'] = total_boxes / len(all_label_files) if all_label_files else 0

            # Categorize object sizes (COCO standard)
            for area in box_areas:
                if area < 0.01:  # Small objects
                    stats['bbox_stats']['small_objects'] += 1
                elif area < 0.09:  # Medium objects  
                    stats['bbox_stats']['medium_objects'] += 1
                else:  # Large objects
                    stats['bbox_stats']['large_objects'] += 1

        # Analyze image properties
        if all_image_files:
            widths, heights = [], []
            for img_file in all_image_files[:200]:  # Sample first 200 images
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        widths.append(w)
                        heights.append(h)
                except:
                    continue

            if widths:
                stats['image_stats']['mean_width'] = np.mean(widths)
                stats['image_stats']['mean_height'] = np.mean(heights)
                stats['image_stats']['aspect_ratios'] = [w/h for w, h in zip(widths, heights)]

        stats['complexity_score'] = self._calculate_detection_complexity(stats)

        print(f"✅ Detection dataset analysis completed:")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Total annotations: {stats['total_annotations']}")
        print(f"   Number of classes: {stats['num_classes']}")

        return stats

    def extract_segmentation_statistics(self, dataset_path: str) -> Dict[str, Any]:
        """Extract statistics from a segmentation dataset."""
        dataset_path = Path(dataset_path)
        print(f"📊 Analyzing segmentation dataset: {dataset_path}")

        stats = {
            'dataset_path': str(dataset_path),
            'task_type': 'segmentation',
            'total_samples': 0,
            'num_classes': 0,
            'mask_stats': {
                'mean_mask_coverage': 0,
                'class_pixel_counts': {}
            },
            'image_stats': {
                'mean_width': 0,
                'mean_height': 0
            }
        }

        # Find image and mask directories
        splits = ['train', 'val', 'test']
        all_image_files = []
        all_mask_files = []

        for split in splits:
            img_dir = dataset_path / 'images' / split
            mask_dir = dataset_path / 'masks' / split

            if img_dir.exists():
                for ext in ['.jpg', '.jpeg', '.png']:
                    all_image_files.extend(list(img_dir.glob(f'*{ext}')))

            if mask_dir.exists():
                all_mask_files.extend(list(mask_dir.glob('*.png')))

        stats['total_samples'] = len(all_image_files)

        # Analyze masks
        unique_classes = set()
        mask_coverages = []

        for mask_file in all_mask_files[:200]:  # Sample first 200 masks
            try:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    unique_pixels = np.unique(mask)
                    unique_classes.update(unique_pixels)

                    # Calculate mask coverage (non-zero pixels)
                    coverage = np.sum(mask > 0) / mask.size
                    mask_coverages.append(coverage)
            except:
                continue

        stats['num_classes'] = len(unique_classes)
        if mask_coverages:
            stats['mask_stats']['mean_mask_coverage'] = np.mean(mask_coverages)

        # Analyze image properties
        if all_image_files:
            widths, heights = [], []
            for img_file in all_image_files[:200]:  # Sample first 200 images
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        widths.append(w)
                        heights.append(h)
                except:
                    continue

            if widths:
                stats['image_stats']['mean_width'] = np.mean(widths)
                stats['image_stats']['mean_height'] = np.mean(heights)

        stats['complexity_score'] = self._calculate_segmentation_complexity(stats)

        print(f"✅ Segmentation dataset analysis completed:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Number of classes: {stats['num_classes']}")

        return stats

    def _calculate_balance_ratio(self, class_counts: Dict[str, int]) -> float:
        """Calculate class balance ratio (1.0 = perfectly balanced)."""
        if not class_counts or len(class_counts) < 2:
            return 1.0

        counts = list(class_counts.values())
        min_count = min(counts)
        max_count = max(counts)

        return min_count / max_count if max_count > 0 else 0.0

    def _categorize_dataset_size(self, num_samples: int) -> str:
        """Categorize dataset size."""
        if num_samples < 100:
            return 'very_small'
        elif num_samples < 1000:
            return 'small'
        elif num_samples < 10000:
            return 'medium'
        elif num_samples < 100000:
            return 'large'
        else:
            return 'very_large'

    def _calculate_complexity_score(self, stats: Dict[str, Any]) -> float:
        """Calculate overall dataset complexity score."""
        score = 0.0

        # Number of classes contribution
        num_classes = stats['num_classes']
        if num_classes <= 2:
            score += 0.1
        elif num_classes <= 10:
            score += 0.3
        elif num_classes <= 100:
            score += 0.5
        else:
            score += 0.7

        # Dataset size contribution
        size_category = stats['dataset_size_category']
        size_scores = {
            'very_small': 0.1,
            'small': 0.2,
            'medium': 0.4,
            'large': 0.6,
            'very_large': 0.8
        }
        score += size_scores.get(size_category, 0.3)

        # Class balance contribution
        balance_ratio = stats['class_balance_ratio']
        if balance_ratio > 0.8:
            score += 0.1  # Well balanced
        elif balance_ratio > 0.5:
            score += 0.2  # Moderately imbalanced
        else:
            score += 0.3  # Highly imbalanced

        return min(score, 1.0)

    def _calculate_detection_complexity(self, stats: Dict[str, Any]) -> float:
        """Calculate detection dataset complexity."""
        score = 0.0

        # Object density
        mean_boxes = stats['bbox_stats']['mean_boxes_per_image']
        if mean_boxes > 10:
            score += 0.4
        elif mean_boxes > 5:
            score += 0.3
        else:
            score += 0.2

        # Object size distribution
        small_ratio = stats['bbox_stats']['small_objects'] / max(1, stats['total_annotations'])
        if small_ratio > 0.5:
            score += 0.3  # Many small objects increase complexity
        else:
            score += 0.1

        # Number of classes
        if stats['num_classes'] > 20:
            score += 0.3
        elif stats['num_classes'] > 10:
            score += 0.2
        else:
            score += 0.1

        return min(score, 1.0)

    def _calculate_segmentation_complexity(self, stats: Dict[str, Any]) -> float:
        """Calculate segmentation dataset complexity."""
        score = 0.0

        # Number of classes
        if stats['num_classes'] > 50:
            score += 0.4
        elif stats['num_classes'] > 20:
            score += 0.3
        else:
            score += 0.2

        # Mask coverage (fine-grained segmentation is more complex)
        coverage = stats['mask_stats']['mean_mask_coverage']
        if coverage < 0.1:
            score += 0.3  # Fine details
        elif coverage < 0.5:
            score += 0.2
        else:
            score += 0.1

        # Dataset size
        if stats['total_samples'] > 10000:
            score += 0.3
        elif stats['total_samples'] > 1000:
            score += 0.2
        else:
            score += 0.1

        return min(score, 1.0)
