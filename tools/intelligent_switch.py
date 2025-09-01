import sys
import os
import cv2
import numpy as np
import json
import yaml
from enum import Enum
import subprocess
import importlib.util
from typing import Dict, Any, Optional, Union

class ProcessingMode(Enum):
    DERAIN = 1
    DEGLARE = 2
    ENHANCE = 3
    NONE = 4

class ConfigManager:
    """Centralized configuration management for dashcam enhancement"""
    
    DEFAULT_CONFIG = {
        'system': {
            'mode': 'auto',
            'use_gpu': True,
            'max_image_dimension': 1024,
            'default_output_dir': 'enhanced_results'
        },
        'rain_detection': {
            'threshold': 0.15,
        },
        'glare_detection': {
            'brightness_threshold': 220,
            'saturation_threshold': 30,
            'min_glare_area': 10,
            'dilation_kernel_size': 5,
            'threshold': 0.25
        },
        'enhancement': {
            'min_contrast': 40,
            'low_exposure_threshold': 0.3,
            'high_exposure_threshold': 0.3
        },
        'retinex_enhancement': {
            'scales': [15, 80, 250],
            'weights': [0.4, 0.4, 0.2],
            'gamma_correction': 0.75,
            'contrast_strength': 1.2,
            'percentile_low': 2,
            'percentile_high': 98
        },
        'selective_deglaring': {
            'enabled': True,
            'feather_edges': True,
            'feather_radius': 10,
            'opacity': 0.9,
            'enhance_only_glare_areas': True
        },
        'performance': {
            'batch_size': 4,
            'num_workers': 2,
            'cache_size': 100
        },
        'logging': {
            'level': 'INFO',
            'save_reports': True,
            'generate_visualizations': True,
            'metrics_calculation': True
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        self.config_path = config_path
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> bool:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Deep merge with existing config
            self._deep_merge(self.config, loaded_config)
            self.config_path = config_path
            print(f"Configuration loaded from {config_path}")
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """Save current configuration to YAML file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config_ptr = self.config
        
        try:
            for k in keys[:-1]:
                if k not in config_ptr:
                    config_ptr[k] = {}
                config_ptr = config_ptr[k]
            
            config_ptr[keys[-1]] = value
            return True
        except (KeyError, TypeError):
            return False
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Recursively merge two dictionaries"""
        for key, value in update.items():
            if (key in base and isinstance(base[key], dict) and 
                isinstance(value, dict)):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def validate_config(self) -> bool:
        """Validate the current configuration"""
        # Add validation logic here
        return True
    
    def generate_config_template(self, output_path: str) -> bool:
        """Generate a configuration template file"""
        return self.save_config(output_path)

class IntelligentImageProcessor:
    def __init__(self, config_manager=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        
        # Initialize configuration
        self.config = config_manager or ConfigManager()
        
        # Set parameters from config
        self.rain_threshold = self.config.get('rain_detection.threshold', 0.15)
        self.glare_threshold = self.config.get('glare_detection.threshold', 0.25)
        self.min_contrast = self.config.get('enhancement.min_contrast', 40)
        
        # Glare detection parameters
        self.glare_brightness_threshold = self.config.get('glare_detection.brightness_threshold', 220)
        self.glare_saturation_threshold = self.config.get('glare_detection.saturation_threshold', 30)
        self.min_glare_area = self.config.get('glare_detection.min_glare_area', 10)
        self.dilation_kernel_size = self.config.get('glare_detection.dilation_kernel_size', 5)
        
        # Rain detection parameters
        self.edge_density_weight = 0.7
        self.texture_score_weight = 0.3
        self.laplacian_threshold = 100
        self.min_rain_score = 0.15
        self.rain_score_multiplier = 1.2
        
        # Enhancement parameters
        self.retinex_scales = self.config.get('retinex_enhancement.scales', [15, 80, 250])
        self.retinex_weights = self.config.get('retinex_enhancement.weights', [0.4, 0.4, 0.2])
        self.gamma_correction = self.config.get('retinex_enhancement.gamma_correction', 0.75)
        self.contrast_strength = self.config.get('retinex_enhancement.contrast_strength', 1.2)
        self.percentile_low = self.config.get('retinex_enhancement.percentile_low', 2)
        self.percentile_high = self.config.get('retinex_enhancement.percentile_high', 98)
        
        # Selective deglaring parameters
        self.selective_deglaring_enabled = self.config.get('selective_deglaring.enabled', True)
        self.feather_edges = self.config.get('selective_deglaring.feather_edges', True)
        self.feather_radius = self.config.get('selective_deglaring.feather_radius', 10)
        self.opacity = self.config.get('selective_deglaring.opacity', 0.9)
        self.enhance_only_glare_areas = self.config.get('selective_deglaring.enhance_only_glare_areas', True)
        
        # Performance parameters
        self.max_image_dimension = self.config.get('system.max_image_dimension', 1024)
        self.use_gpu = self.config.get('system.use_gpu', True)

    def detect_rain(self, image):
        """Enhanced rain detection with configurable parameters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        texture_score = np.mean(grad_mag) / 255.0
        
        # Use configurable weights
        rain_score = (edge_density * self.edge_density_weight + 
                     texture_score * self.texture_score_weight)
        
        # Add Laplacian check for fine droplets
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian > self.laplacian_threshold:
            rain_score = max(rain_score, self.min_rain_score)
    
        return min(rain_score * self.rain_score_multiplier, 1.0)

    def detect_glare(self, image):
        """Enhanced glare detection with better debugging"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        s = hsv[:,:,1]
        
        # Debug: Print HSV values
        print(f"HSV Stats - V: mean={np.mean(v):.1f}, max={np.max(v)}, min={np.min(v)}")
        print(f"HSV Stats - S: mean={np.mean(s):.1f}, max={np.max(s)}, min={np.min(s)}")
        
        # Create glare mask
        brightness_mask = v > self.glare_brightness_threshold
        saturation_mask = s < self.glare_saturation_threshold
        glare_mask = brightness_mask & saturation_mask
        
        print(f"Brightness mask: {np.sum(brightness_mask)} pixels")
        print(f"Saturation mask: {np.sum(saturation_mask)} pixels")
        print(f"Combined glare mask: {np.sum(glare_mask)} pixels")
        
        # Filter out small glare areas
        if self.min_glare_area > 0 and np.sum(glare_mask) > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                glare_mask.astype(np.uint8), connectivity=8
            )
            
            print(f"Found {num_labels-1} glare regions before filtering")
            
            # Create a new mask that only includes components larger than min_glare_area
            filtered_mask = np.zeros_like(glare_mask)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_glare_area:
                    filtered_mask[labels == i] = True
                    print(f"Glare region {i}: {area} pixels (kept)")
                else:
                    print(f"Glare region {i}: {area} pixels (filtered out)")
                    
            glare_mask = filtered_mask
        
        # Dilate glare regions to capture surrounding affected areas
        if self.dilation_kernel_size > 0 and np.sum(glare_mask) > 0:
            kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
            glare_mask = cv2.dilate(glare_mask.astype(np.uint8), kernel)
            print(f"After dilation: {np.sum(glare_mask)} pixels")
        
        glare_score = np.sum(glare_mask) / glare_mask.size
        print(f"Final glare score: {glare_score:.4f}")
        
        return glare_score, glare_mask.astype(bool)

    def needs_enhancement(self, image):
        """Check if image needs general enhancement with configurable parameters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check for low contrast
        if np.std(gray) < self.min_contrast:
            return True
        
        # Check for poor exposure with configurable thresholds
        low_exposure_threshold = self.config.get('enhancement.low_exposure_threshold', 0.3)
        high_exposure_threshold = self.config.get('enhancement.high_exposure_threshold', 0.3)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = hist / hist.sum()
        if hist[0] > low_exposure_threshold or hist[-1] > high_exposure_threshold:
            return True
            
        return False

    def selective_deglaring(self, image_path, output_path, glare_mask=None):
        """
        Apply deglaring only to areas affected by glare with configurable parameters
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not read image'}
                
            original_size = image.shape[:2]  # Store original size
            
            # Resize if needed for performance
            if self.max_image_dimension:
                h, w = image.shape[:2]
                if max(h, w) > self.max_image_dimension:
                    scale = self.max_image_dimension / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                
            # Convert to RGB for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # If no mask provided, detect glare areas
            if glare_mask is None:
                glare_score, glare_mask = self.detect_glare(image)
                glare_mask = glare_mask.astype(bool)
            
            print(f"Glare areas detected: {np.sum(glare_mask)} pixels")
            
            # Save glare mask for debugging
            debug_dir = os.path.join(os.path.dirname(output_path), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            mask_vis = glare_mask.astype(np.uint8) * 255
            cv2.imwrite(os.path.join(debug_dir, "glare_mask.png"), mask_vis)
            
            # Apply enhancement
            if np.any(glare_mask) and self.enhance_only_glare_areas:
                print("Applying selective enhancement to glare areas only")
                enhanced_image = self._enhance_glare_regions(image_rgb, glare_mask)
            else:
                print("Applying full image enhancement")
                enhanced_image = self._multi_scale_retinex(image_rgb)
            
            # Convert back to BGR
            enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
            
            # Resize back to original size if needed
            if enhanced_bgr.shape[:2] != original_size:
                enhanced_bgr = cv2.resize(enhanced_bgr, (original_size[1], original_size[0]))
            
            # Save final enhanced image
            cv2.imwrite(output_path, enhanced_bgr)
            
            # Save intermediate results for comparison (resize both to same size)
            original_resized = cv2.resize(image, (enhanced_bgr.shape[1], enhanced_bgr.shape[0]))
            cv2.imwrite(os.path.join(debug_dir, "original.png"), original_resized)
            cv2.imwrite(os.path.join(debug_dir, "enhanced.png"), enhanced_bgr)
            
            # Create side-by-side comparison (ensure same dimensions)
            if original_resized.shape == enhanced_bgr.shape:
                comparison = np.hstack([original_resized, enhanced_bgr])
                cv2.imwrite(os.path.join(debug_dir, "comparison.png"), comparison)
            else:
                print("Warning: Could not create comparison image due to dimension mismatch")
            
            return {'success': True}
            
        except Exception as e:
            import traceback
            print(f"Error in selective_deglaring: {str(e)}")
            print(traceback.format_exc())
            return {'success': False, 'error': str(e)}
        """
        Apply deglaring only to areas affected by glare with configurable parameters
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not read image'}
                
            # Resize if needed for performance
            original_size = image.shape[:2]
            if self.max_image_dimension:
                h, w = image.shape[:2]
                if max(h, w) > self.max_image_dimension:
                    scale = self.max_image_dimension / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                
            # Convert to RGB for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # If no mask provided, detect glare areas
            if glare_mask is None:
                glare_score, glare_mask = self.detect_glare(image)
                glare_mask = glare_mask.astype(bool)
            
            print(f"Glare areas detected: {np.sum(glare_mask)} pixels")
            
            # Save glare mask for debugging
            debug_dir = os.path.join(os.path.dirname(output_path), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            mask_vis = glare_mask.astype(np.uint8) * 255
            cv2.imwrite(os.path.join(debug_dir, "glare_mask.png"), mask_vis)
            
            # Apply enhancement
            if np.any(glare_mask) and self.enhance_only_glare_areas:
                print("Applying selective enhancement to glare areas only")
                enhanced_image = self._enhance_glare_regions(image_rgb, glare_mask)
            else:
                print("Applying full image enhancement")
                enhanced_image = self._multi_scale_retinex(image_rgb)
            
            # Resize back to original size if needed
            if enhanced_image.shape[:2] != original_size:
                enhanced_image = cv2.resize(enhanced_image, (original_size[1], original_size[0]))
            
            # Convert back to BGR and save
            enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, enhanced_bgr)
            
            # Save intermediate results for comparison
            cv2.imwrite(os.path.join(debug_dir, "original.png"), image)
            cv2.imwrite(os.path.join(debug_dir, "enhanced.png"), enhanced_bgr)
            
            # Create side-by-side comparison
            comparison = np.hstack([image, enhanced_bgr])
            cv2.imwrite(os.path.join(debug_dir, "comparison.png"), comparison)
            
            return {'success': True}
            
        except Exception as e:
            import traceback
            print(f"Error in selective_deglaring: {str(e)}")
            print(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    




    def _enhance_glare_regions(self, image, glare_mask):
        """
        Enhance only the glare-affected regions using Multi-Scale Retinex
        with configurable parameters
        """
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        # Create enhanced version of the image (ensure same dimensions)
        enhanced = self._multi_scale_retinex(image)
        
        # Feather edges if configured
        if self.feather_edges:
            glare_mask = self._feather_mask(glare_mask, self.feather_radius)
        
        # Ensure both images have the same dimensions
        if enhanced.shape[:2] != image.shape[:2]:
            print(f"Warning: Enhanced image dimensions {enhanced.shape[:2]} don't match original {image.shape[:2]}")
            enhanced = cv2.resize(enhanced, (image.shape[1], image.shape[0]))
        
        # Blend enhanced regions with original
        result = image.copy().astype(np.float32)
        enhanced = enhanced.astype(np.float32)
        
        # Apply enhancement only to masked regions
        for c in range(3):
            result[:, :, c] = np.where(
                glare_mask,
                result[:, :, c] * (1 - self.opacity) + enhanced[:, :, c] * self.opacity,
                result[:, :, c]
            )
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _multi_scale_retinex(self, image):
        """Improved Multi-Scale Retinex for glare reduction"""
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        # Convert to float and add small value to avoid log(0)
        img = image.astype(np.float64) / 255.0 + 1e-8
        log_img = np.log(img)

        msr_result = np.zeros_like(log_img)

        for scale, weight in zip(self.retinex_scales, self.retinex_weights):
            # Use different sigma values for each scale
            sigma = scale / 3.0  # Empirical relationship
            kernel_size = int(6 * sigma) + 1  # Ensure odd kernel size
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            blurred = cv2.GaussianBlur(log_img, (kernel_size, kernel_size), sigma)
            single_scale_retinex = log_img - blurred
            msr_result += weight * single_scale_retinex

        # Apply gain and offset
        enhanced = np.exp(msr_result) - 1.0

        # Normalize each channel separately with adaptive stretching
        for c in range(3):
            channel = enhanced[:, :, c]
            
            # Use configurable percentiles
            p_low = np.percentile(channel, self.percentile_low)
            p_high = np.percentile(channel, self.percentile_high)
            
            # Avoid division by zero
            if p_high - p_low < 1e-8:
                p_low = np.min(channel)
                p_high = np.max(channel)
                if p_high - p_low < 1e-8:
                    p_high = p_low + 1e-8
            
            # Stretch contrast
            channel = (channel - p_low) / (p_high - p_low)
            
            # Apply gamma correction
            channel = np.power(np.clip(channel, 0, 1), self.gamma_correction)
            
            # Apply contrast strength
            channel = channel * self.contrast_strength
            
            enhanced[:, :, c] = np.clip(channel, 0, 1)

        # Ensure output has same dimensions as input
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        
        # Double-check dimensions
        if enhanced_uint8.shape[:2] != (original_height, original_width):
            print(f"Warning: Resizing enhanced image from {enhanced_uint8.shape[:2]} to {(original_height, original_width)}")
            enhanced_uint8 = cv2.resize(enhanced_uint8, (original_width, original_height))
        
        return enhanced_uint8
    
    
    def _feather_mask(self, mask, radius):
        """Feather the edges of a mask for smoother blending"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        feathered = cv2.GaussianBlur(mask.astype(np.float32), (radius*2+1, radius*2+1), radius)
        return feathered / 255.0

    def visualize_glare_detection(self, image, glare_mask, output_path):
        """Create visualization of glare detection results"""
        # Create visualization image
        vis_image = image.copy()
        
        # Create colored mask (red for glare areas)
        color_mask = np.zeros_like(vis_image)
        color_mask[glare_mask] = [0, 0, 255]  # Red color for glare
        
        # Blend mask with image
        alpha = 0.3
        vis_image = cv2.addWeighted(vis_image, 1.0, color_mask, alpha, 0)
        
        # Add text information
        glare_pixels = np.sum(glare_mask)
        total_pixels = glare_mask.size
        glare_percentage = (glare_pixels / total_pixels) * 100
        
        cv2.putText(vis_image, f"Glare: {glare_pixels} pixels ({glare_percentage:.1f}%)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(vis_image, f"Threshold: V>{self.glare_brightness_threshold}, S<{self.glare_saturation_threshold}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save visualization
        cv2.imwrite(output_path, vis_image)
        
        return vis_image

    def run_deraining(self, image_path, output_path):
        """Execute Attentive GAN deraining model with better error handling"""
        try:
            weights_path = os.path.join(self.project_root, "weights", "derain_gan", "derain_gan.ckpt-100000")
            
            # Check if weights exist with different extensions
            weights_found = False
            weight_files = []
            
            # Check for all possible weight file components
            for ext in ['', '.meta', '.index', '.data-00000-of-00001']:
                check_path = weights_path + ext
                if os.path.exists(check_path):
                    weight_files.append(check_path)
                    weights_found = True
            
            if not weights_found:
                error_msg = f'Weights files not found. Expected at: {weights_path}.*'
                print(error_msg)
                
                # Try to find any checkpoint files in the directory
                weights_dir = os.path.dirname(weights_path)
                if os.path.exists(weights_dir):
                    available_files = os.listdir(weights_dir)
                    checkpoint_files = [f for f in available_files if 'ckpt' in f]
                    if checkpoint_files:
                        error_msg += f'\nFound these files instead: {checkpoint_files}'
                        print(f"Available checkpoint files: {checkpoint_files}")
                        
                        # Try to use the latest checkpoint if available
                        if checkpoint_files:
                            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('-')[-1]) if '-' in x else 0)
                            weights_path = os.path.join(weights_dir, latest_checkpoint.split('.')[0])
                            print(f"Trying to use: {weights_path}")
                
                return {'success': False, 'error': error_msg}
            
            print(f"Weight files found: {weight_files}")
            
            # Check if test_model.py exists
            test_script_path = os.path.join(self.project_root, "tools", "test_model.py")
            if not os.path.exists(test_script_path):
                return {'success': False, 'error': f'Test script not found: {test_script_path}'}
            
            cmd = [
                sys.executable,
                test_script_path,
                "--image_path", os.path.abspath(image_path),
                "--weights_path", os.path.abspath(weights_path),
                "--output_file", os.path.join(self.project_root, "derain_results.txt")
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            
            # Set environment to include project root
            env = os.environ.copy()
            env['PYTHONPATH'] = self.project_root + os.pathsep + env.get('PYTHONPATH', '')
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=self.project_root)
            
            print(f"Command stdout: {result.stdout}")
            if result.stderr:
                print(f"Command stderr: {result.stderr}")
            
            if result.returncode == 0:
                output_file = os.path.join(self.project_root, "derain_ret.png")
                if os.path.exists(output_file):
                    os.replace(output_file, os.path.abspath(output_path))
                    return {'success': True}
                else:
                    return {'success': False, 'error': 'Output file was not created'}
            return {'success': False, 'error': result.stderr or 'Unknown error'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_deglaring(self, image_path, output_path):
        """Execute selective deglaring based on configuration"""
        try:
            # Read image to detect glare areas
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not read image'}
                
            # Detect glare areas
            glare_score, glare_mask = self.detect_glare(image)
            
            # Apply selective deglaring
            result = self.selective_deglaring(image_path, output_path, glare_mask)
            
            if result['success']:
                # Calculate metrics if needed
                enhanced = cv2.imread(output_path)
                if enhanced is not None:
                    metrics = self.calculate_deglaring_metrics(image, enhanced, glare_mask)
                    result['metrics'] = metrics
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def calculate_deglaring_metrics(self, original, enhanced, glare_mask):
        """Calculate metrics specific to deglaring performance"""
        # Convert to appropriate color spaces
        orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        enh_hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        # Calculate metrics only in glare regions
        glare_pixels = np.where(glare_mask)
        
        if len(glare_pixels[0]) == 0:
            return {
                'glare_area_reduced': 0,
                'brightness_reduction': 0,
                'saturation_improvement': 0
            }
        
        # Original values in glare regions
        orig_v = orig_hsv[:,:,2][glare_pixels]
        orig_s = orig_hsv[:,:,1][glare_pixels]
        
        # Enhanced values in glare regions
        enh_v = enh_hsv[:,:,2][glare_pixels]
        enh_s = enh_hsv[:,:,1][glare_pixels]
        
        # Calculate improvements
        brightness_reduction = np.mean(orig_v) - np.mean(enh_v)
        saturation_improvement = np.mean(enh_s) - np.mean(orig_s)
        
        # Check if glare is still present after processing
        still_glare = (enh_v > self.glare_brightness_threshold) & \
                      (enh_s < self.glare_saturation_threshold)
        glare_area_reduced = 1 - (np.sum(still_glare) / len(glare_pixels[0]))
        
        return {
            'glare_area_reduced': float(glare_area_reduced),
            'brightness_reduction': float(brightness_reduction),
            'saturation_improvement': float(saturation_improvement)
        }

    def process_image(self, image_path, output_path):
        """Main processing function with enhanced debugging"""
        try:
            # Verify input image exists
            if not os.path.exists(image_path):
                return {'final_status': 'error', 'error': f'Input image not found: {image_path}'}
            
            img = cv2.imread(image_path)
            if img is None:
                return {'final_status': 'error', 'error': 'Could not read image file'}

            print("=" * 50)
            print("IMAGE ANALYSIS DEBUG INFORMATION")
            print("=" * 50)
            
            # Analysis with detailed debugging
            rain_score = self.detect_rain(img)
            glare_score, glare_mask = self.detect_glare(img)
            needs_enhance = self.needs_enhancement(img)
            
            print("\nSUMMARY:")
            print(f"Rain Score: {rain_score:.3f} (Threshold: {self.rain_threshold})")
            print(f"Glare Score: {glare_score:.3f} (Threshold: {self.glare_threshold})")
            print(f"Needs Enhancement: {needs_enhance}")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Create debug directory
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save glare detection visualization
            self.visualize_glare_detection(img, glare_mask, os.path.join(debug_dir, "glare_detection.png"))
            
            # Determine processing mode
            processing_mode = self.config.get('system.mode', 'auto')
            if processing_mode == 'auto':
                if args.force_derain or rain_score > self.rain_threshold:
                    mode = 'DERAIN'
                elif glare_score > self.glare_threshold:
                    mode = 'DEGLARE'
                elif needs_enhance:
                    mode = 'ENHANCE'
                else:
                    mode = 'NONE'
            else:
                mode = processing_mode.upper()
            
            # Processing logic
            if mode == 'DERAIN':
                print("\nExecuting rain removal...")
                result = self.run_deraining(image_path, output_path)
            elif mode == 'DEGLARE':
                print("\nExecuting glare reduction...")
                result = self.run_deglaring(image_path, output_path)
            elif mode == 'ENHANCE':
                print("\nExecuting general enhancement...")
                result = self.run_deglaring(image_path, output_path)  # Use deglaring for enhancement too
            else:
                print("\nImage is clear - no enhancement needed")
                cv2.imwrite(output_path, img)
                return {
                    'final_status': 'success',
                    'processing_mode': 'NONE',
                    'metrics': {
                        'rain_score': rain_score,
                        'glare_score': glare_score,
                        'needed_enhancement': needs_enhance
                    }
                }

            if result['success']:
                print("\nProcessing successful!")
                # Calculate metrics for enhanced image
                enhanced_img = cv2.imread(output_path)
                if enhanced_img is not None:
                    metrics = self.calculate_deglaring_metrics(img, enhanced_img, glare_mask)
                    result['metrics'] = metrics
                    
                    print("\nENHANCEMENT METRICS:")
                    print(f"Glare Area Reduced: {metrics.get('glare_area_reduced', 0):.1%}")
                    print(f"Brightness Reduction: {metrics.get('brightness_reduction', 0):.1f}")
                    print(f"Saturation Improvement: {metrics.get('saturation_improvement', 0):.1f}")
                
                return {
                    'final_status': 'success',
                    'processing_mode': mode,
                    'metrics': result.get('metrics', {
                        'rain_score': rain_score,
                        'glare_score': glare_score,
                        'needed_enhancement': needs_enhance
                    })
                }
            else:
                print(f"\nProcessing failed: {result.get('error', 'Unknown error')}")
                print("Falling back to original image")
                cv2.imwrite(output_path, img)
                return {
                    'final_status': 'fallback',
                    'error': result.get('error', 'Processing failed'),
                    'processing_mode': mode,
                    'metrics': {
                        'rain_score': rain_score,
                        'glare_score': glare_score,
                        'needed_enhancement': needs_enhance
                    }
                }

        except Exception as e:
            import traceback
            print(f"\nError during processing: {str(e)}")
            print(traceback.format_exc())
            return {'final_status': 'error', 'error': str(e)}


# Main execution with config loading
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Intelligent Image Processing for Dashcam Footage')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--mode', choices=['auto', 'derain', 'deglare', 'enhance'], 
                       help='Processing mode override')
    parser.add_argument('--report', help='Optional report file path')
    parser.add_argument('--force_derain', action='store_true', 
                       help='Override rain detection and force deraining')
    
    args = parser.parse_args()

    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Override mode if specified
    if args.mode:
        config_manager.set('system.mode', args.mode)
    
    # Initialize processor with configuration
    processor = IntelligentImageProcessor(config_manager)
    
    # Process the image
    result = processor.process_image(args.image, args.output)

    # Generate report if requested
    if args.report and config_manager.get('logging.save_reports', True):
        try:
            with open(args.report, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Report saved to: {args.report}")
        except Exception as e:
            print(f"Failed to save report: {str(e)}")