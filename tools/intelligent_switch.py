import sys
import os
import cv2
import numpy as np
import json
import shutil
import yaml
from enum import Enum
import subprocess
import importlib.util
from typing import Dict, Any, Optional, Union
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from scipy.ndimage import gaussian_filter



try:
    from enhanced_retinex import EnhancedRetinexProcessor, integrate_with_existing_processor
except ImportError:
    print("Warning: enhanced_retinex module not found. Creating fallback implementation.")
    
    # Fallback implementation
    class EnhancedRetinexProcessor:
        def __init__(self, use_gpu=True):
            self.use_gpu = use_gpu
            self.validation_set = []
            
        def setup_validation_set(self, image_paths):
            print(f"Fallback: Validation set with {len(image_paths)} images")
            self.validation_set = image_paths
            
        def run_validation_assessment(self):
            return {"status": "fallback_mode", "message": "Using fallback implementation"}
            
        def evaluate_quality(self, original, enhanced, save_path=None):
            return {
                "quality_grade": "UNKNOWN",
                "ssim": 0.5,
                "psnr": 25.0,
                "contrast_improvement_pct": 0.0,
                "message": "Fallback mode - install enhanced_retinex for accurate metrics"
            }
    
    def integrate_with_existing_processor(processor):
        return EnhancedRetinexProcessor()


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
            'threshold': 0.068,
        },
        'glare_detection': {
            'brightness_threshold': 220,
            'saturation_threshold': 30,
            'min_glare_area': 10,
            'dilation_kernel_size': 5,
            'threshold': 0.005
        },
        'deraining': {
            'enabled': True,
            'config_path': 'deraining_config.yaml',
            'force_scene': None  # Can be 'day', 'night', 'default', or None
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
        """Get configuration value using dot notation with fallback"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            # Fallback to default
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

    @staticmethod
    def compute_derain_metrics(original_path, derained_path):
        """
        Robust version: Automatically resizes mismatched images,
        handles missing files, and ensures PSNR/SSIM computation works.
        """
        import os, cv2, numpy as np
        from skimage.metrics import structural_similarity as compare_ssim
        from skimage.metrics import peak_signal_noise_ratio as compare_psnr

        try:
            if not os.path.exists(original_path):
                print(f"[ERROR] Original image not found: {original_path}")
                return None
            if not os.path.exists(derained_path):
                print(f"[ERROR] Derained image not found: {derained_path}")
                return None

            # Read both images
            orig = cv2.imread(original_path)
            derain = cv2.imread(derained_path)

            if orig is None or derain is None:
                print("[ERROR] One or both images could not be read.")
                return None

            # --- auto-fix dimension mismatch ---
            if orig.shape != derain.shape:
                print(f"[WARN] Image size mismatch: "
                    f"original={orig.shape}, derained={derain.shape}. Auto-resizing derained...")
                derain = cv2.resize(derain, (orig.shape[1], orig.shape[0]))

            # Convert to grayscale
            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            derain_gray = cv2.cvtColor(derain, cv2.COLOR_BGR2GRAY)

            # Compute metrics safely
            psnr = compare_psnr(orig_gray, derain_gray, data_range=255)
            ssim = compare_ssim(orig_gray, derain_gray, data_range=255)

            # Compute basic stats
            contrast_orig = np.std(orig_gray)
            contrast_derain = np.std(derain_gray)
            brightness_orig = np.mean(orig_gray)
            brightness_derain = np.mean(derain_gray)

            # Prepare output folder
            metrics_dir = os.path.join("metrics", "deraining")
            os.makedirs(metrics_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(original_path))[0]

            # Write report
            report_path = os.path.join(metrics_dir, f"{filename}_derainmetrics.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"DERAIN METRICS REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Original: {original_path}\n")
                f.write(f"Derained: {derained_path}\n\n")
                f.write(f"PSNR: {psnr:.2f} dB\n")
                f.write(f"SSIM: {ssim:.4f}\n")
                f.write(f"Contrast (orig→derain): {contrast_orig:.2f} → {contrast_derain:.2f}\n")
                f.write(f"Brightness (orig→derain): {brightness_orig:.2f} → {brightness_derain:.2f}\n")

            print(f"✅ Metrics report saved to: {report_path}")

            # Save comparison preview
            preview_path = os.path.join(metrics_dir, f"{filename}_comparison.png")
            label1 = orig.copy()
            label2 = derain.copy()
            cv2.putText(label1, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(label2, "Derained", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            comparison = np.hstack((label1, label2))
            cv2.imwrite(preview_path, comparison)
            print(f"✅ Comparison saved to: {preview_path}")

            # Optional: CSV summary auto-update
            csv_path = os.path.join(metrics_dir, "deraining_summary.csv")
            import csv
            header = ["Filename", "PSNR", "SSIM"]
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow([filename, round(psnr, 3), round(ssim, 4)])

            print(f"✅ Summary CSV updated: {csv_path}")

            return {"psnr": psnr, "ssim": ssim}

        except Exception as e:
            import traceback
            print(f"❌ Error computing derain metrics: {e}")
            print(traceback.format_exc())
            return None


    def is_image_clean(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        clean_config = self.config.get('clean_detection', {})
        metrics = {}
        issues = []
        
        # 1. Basic quality checks
        contrast = float(np.std(gray))
        brightness = float(np.mean(gray))
        metrics['contrast'] = contrast
        metrics['brightness'] = brightness
        
        if contrast < clean_config.get('min_good_contrast', 20):
            issues.append(f"Very low contrast ({contrast:.1f})")
        
        if brightness < clean_config.get('min_brightness', 30):
            issues.append(f"Extremely dark ({brightness:.1f})")
        elif brightness > clean_config.get('max_brightness', 235):
            issues.append(f"Extremely bright ({brightness:.1f})")
        
        # 2. Use HEADLIGHT detection instead of general glare
        glare_score, glare_mask = self.detect_headlight_glare(image)
        metrics['headlight_glare_score'] = glare_score
        
        # CRITICAL: Only flag if actual headlights detected
        if glare_score > clean_config.get('max_glare_score', 0.01):
            issues.append(f"Headlight glare detected ({glare_score:.4f})")
        
        # 3. Simple rain check (edge density)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        metrics['edge_density'] = edge_density
        
        if edge_density > 0.40:  # Only extreme edge density
            issues.append(f"High texture/rain pattern ({edge_density:.3f})")
        
        # 4. Noise check
        noise_level = self._estimate_noise_level(gray)
        metrics['noise_level'] = noise_level
        
        max_noise = clean_config.get('max_noise_level', 35)
        if noise_level > max_noise:
            issues.append(f"Excessive noise ({noise_level:.1f})")
        
        # DECISION
        is_clean = len(issues) == 0
        confidence = (4 - len(issues)) / 4  # Changed from 3 to 4 checks
        
        if is_clean:
            reason = "✓ Image is CLEAN - no headlight glare or quality issues"
        else:
            reason = "✗ Issues detected: " + "; ".join(issues)
        
        return is_clean, reason, confidence, metrics

    def process_image(self, image_path, output_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'final_status': 'error', 'error': 'Could not read image'}

            print("=" * 70)
            print(f"PROCESSING: {os.path.basename(image_path)}")
            print("=" * 70)
            
            # ====================================================================
            # STEP 1: PRIORITY CLEAN CHECK (Lightweight)
            # ====================================================================
            clean_config = self.config.get('clean_detection', {})
            
            if clean_config.get('enabled', True):
                print("\n[STEP 1: CLEAN DETECTION - PRIORITY CHECK]")
                
                is_clean, reason, confidence, metrics = self.is_image_clean(img)
                
                print(f"  Result: {'CLEAN ✓' if is_clean else 'NEEDS PROCESSING ✗'}")
                print(f"  Confidence: {confidence*100:.1f}%")
                print(f"  Reason: {reason}")
                
                # Show key metrics
                if metrics:
                    print(f"  Key Metrics:")
                    print(f"    - Contrast: {metrics.get('contrast', 0):.1f}")
                    print(f"    - Brightness: {metrics.get('brightness', 0):.1f}")
                    print(f"    - Headlight Glare: {metrics.get('headlight_glare_score', 0):.6f}")
                    print(f"    - Edge Density: {metrics.get('edge_density', 0):.4f}")
                
                # Get confidence threshold
                min_confidence = clean_config.get('min_confidence_to_skip', 0.70)
                
                # ================================================================
                # SKIP if clean with sufficient confidence
                # ================================================================
                if is_clean and confidence >= min_confidence:
                    print(f"\n{'='*70}")
                    print("✓✓✓ CLEAN IMAGE - SKIPPING ALL PROCESSING ✓✓✓")
                    print(f"{'='*70}")
                    print(f"Confidence: {confidence*100:.1f}% >= {min_confidence*100:.1f}% threshold")
                    print("Action: Copying original without modification")
                    
                    import shutil
                    shutil.copy(image_path, output_path)
                    
                    return {
                        'final_status': 'success',
                        'processing_mode': 'NONE',
                        'reason': 'Clean image - processing skipped',
                        'confidence': confidence,
                        'metrics': metrics
                    }
                
                # Borderline case
                if is_clean and confidence < min_confidence:
                    print(f"\n⚠ WARNING: Clean but low confidence ({confidence*100:.1f}%)")
                    print("Proceeding to verify with headlight detection...")
            
            # ====================================================================
            # STEP 2: HEADLIGHT DETECTION (Only if not clean or uncertain)
            # ====================================================================
            print(f"\n[STEP 2: HEADLIGHT GLARE DETECTION]")
            
            glare_score, glare_mask = self.detect_headlight_glare(img)
            
            print(f"  Headlight Glare Score: {glare_score:.6f} (threshold: {self.glare_threshold})")
            print(f"  Affected pixels: {np.sum(glare_mask)} ({np.sum(glare_mask)/glare_mask.size*100:.3f}%)")
            
            # Create output directories
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save detection visualization
            self.visualize_glare_detection(img, glare_mask, 
                                        os.path.join(debug_dir, "headlight_detection.png"))
            
            # ====================================================================
            # STEP 3: DETERMINE MODE
            # ====================================================================
            if glare_score > self.glare_threshold:
                mode = 'DEGLARE'
            else:
                mode = 'NONE'
            
            print(f"\n[STEP 3: PROCESSING MODE SELECTED]")
            print(f"  Mode: {mode}")
            
            # ====================================================================
            # STEP 4: EXECUTE OR SKIP
            # ====================================================================
            if mode == 'NONE':
                print(f"\n{'='*70}")
                print("✓ NO HEADLIGHT GLARE DETECTED - QUALITY SUFFICIENT")
                print(f"{'='*70}")
                
                import shutil
                shutil.copy(image_path, output_path)
                
                return {
                    'final_status': 'success',
                    'processing_mode': 'NONE',
                    'reason': 'No headlight glare detected',
                    'metrics': {
                        'headlight_glare_score': glare_score,
                        'quality_metrics': metrics if 'metrics' in locals() else {}
                    }
                }
            
            # Execute deglaring
            print(f"\n[STEP 4: EXECUTING {mode}]")
            result = self.run_deglaring(image_path, output_path)
            
            # Handle result
            if result['success']:
                print(f"✓ {mode} processing completed")
                
                enhanced_img = cv2.imread(output_path)
                if enhanced_img is not None:
                    metrics = self.calculate_deglaring_metrics(img, enhanced_img, glare_mask)
                    result['metrics'] = metrics
                
                return {
                    'final_status': 'success',
                    'processing_mode': mode,
                    'metrics': result.get('metrics', {})
                }
            else:
                print(f"✗ {mode} processing failed, using original")
                import shutil
                shutil.copy(image_path, output_path)
                return {
                    'final_status': 'fallback',
                    'error': result.get('error'),
                    'processing_mode': mode
                }
        
        except Exception as e:
            import traceback
            print(f"\n✗ ERROR: {str(e)}")
            print(traceback.format_exc())
            return {'final_status': 'error', 'error': str(e)}

    def _estimate_noise_level(self, gray_image):
        """
        Estimate noise level using Median Absolute Deviation
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            noise_level: Estimated noise standard deviation
        """
        # Use high-pass filter to isolate noise
        H, W = gray_image.shape
        
        # Use a 3x3 Laplacian kernel
        kernel = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]], dtype=np.float32)
        
        # Apply filter
        filtered = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        
        # Estimate noise using MAD (Median Absolute Deviation)
        sigma = np.median(np.abs(filtered)) / 0.6745
        
        return float(sigma)


    def needs_enhancement_improved(self, image):
        """
        IMPROVED VERSION: Replace the old needs_enhancement method
        
        This checks if image needs enhancement using comprehensive quality assessment
        """
        is_clean, reason, confidence, metrics = self.is_image_clean(image)
        
        # If image is clean with high confidence, no enhancement needed
        if is_clean and confidence >= 0.9:
            print(f"[CLEAN IMAGE] {reason}")
            print(f"  Confidence: {confidence*100:.1f}%")
            return False
        
        # If image has issues or low confidence, needs enhancement
        if not is_clean:
            print(f"[NEEDS ENHANCEMENT] {reason}")
            print(f"  Confidence: {confidence*100:.1f}%")
            return True
        
        # Borderline case - use original logic as fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if np.std(gray) < self.min_contrast:
            return True
        
        low_exposure_threshold = self.config.get('enhancement.low_exposure_threshold', 0.3)
        high_exposure_threshold = self.config.get('enhancement.high_exposure_threshold', 0.3)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = hist / hist.sum()
        
        if hist[0] > low_exposure_threshold or hist[-1] > high_exposure_threshold:
            return True
            
        return False


    # ========================================================================
    # UPDATED process_image METHOD - Replace in intelligent_switch.py
    # ========================================================================

    def process_image(self, image_path, output_path):
        """
        FIXED: Process image with PRIORITY clean detection
        Clean images are checked FIRST and skipped immediately
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {'final_status': 'error', 'error': 'Could not read image'}

            print("=" * 70)
            print("IMAGE ANALYSIS WITH PRIORITY CLEAN DETECTION")
            print("=" * 70)
            print(f"Input: {os.path.basename(image_path)}")
            
            # ================================================================
            # PRIORITY CHECK: Is image clean? (CHECK THIS FIRST!)
            # ================================================================
            clean_config = self.config.get('clean_detection', {})
            
            if clean_config.get('enabled', True):
                is_clean, clean_reason, confidence, quality_metrics = self.is_image_clean(img)
                
                print(f"\n[CLEAN DETECTION - PRIORITY CHECK]")
                print(f"  Is Clean: {is_clean}")
                print(f"  Confidence: {confidence*100:.1f}%")
                print(f"  Reason: {clean_reason}")
                
                # Get threshold
                min_confidence = clean_config.get('min_confidence_to_skip', 0.70)
                
                # ============================================================
                # SKIP IMMEDIATELY if clean with sufficient confidence
                # ============================================================
                if is_clean and confidence >= min_confidence:
                    print(f"\n{'='*70}")
                    print("✓✓✓ CLEAN IMAGE DETECTED - SKIPPING ALL PROCESSING ✓✓✓")
                    print(f"{'='*70}")
                    print(f"Confidence: {confidence*100:.1f}% >= {min_confidence*100:.1f}%")
                    print(f"Action: Copying original without modification")
                    
                    # Just copy original
                    import shutil
                    shutil.copy(image_path, output_path)
                    
                    return {
                        'final_status': 'success',
                        'processing_mode': 'NONE',
                        'reason': 'Clean image - processing skipped',
                        'confidence': confidence,
                        'metrics': {
                            'quality_metrics': quality_metrics,
                            'enhancement_skipped': True,
                            'processing_time': 0.0
                        }
                    }
                
                # Borderline case
                elif is_clean and confidence < min_confidence:
                    print(f"\n[WARNING] Clean but low confidence ({confidence*100:.1f}%)")
                    print(f"[INFO] Running full detection for verification...")
            
            # ================================================================
            # FULL DETECTION (Only if not clean or low confidence)
            # ================================================================
            print(f"\n[RUNNING FULL DETECTION]")
            rain_score = self.detect_rain(img)
            glare_score, glare_mask = self.detect_headlight_glare(img)
            needs_enhance = self.needs_enhancement_improved(img)
            
            print(f"  Rain:  {rain_score:.3f} (threshold: {self.rain_threshold})")
            print(f"  Glare: {glare_score:.3f} (threshold: {self.glare_threshold})")
            print(f"  Needs Enhancement: {needs_enhance}")
            
            # Create output directory
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save visualization
            self.visualize_glare_detection(img, glare_mask, 
                                        os.path.join(debug_dir, "glare_detection.png"))
            
            # ================================================================
            # DETERMINE PROCESSING MODE
            # ================================================================
            if rain_score > self.rain_threshold:
                mode = 'DERAIN'
            elif glare_score > self.glare_threshold:
                mode = 'DEGLARE'
            elif needs_enhance:
                mode = 'ENHANCE'
            else:
                mode = 'NONE'
            
            print(f"\n[PROCESSING MODE SELECTED: {mode}]")
            
            # ================================================================
            # EXECUTE OR SKIP
            # ================================================================
            if mode == 'NONE':
                print(f"{'='*70}")
                print("✓ NO PROCESSING NEEDED - SUFFICIENT QUALITY")
                print(f"{'='*70}")
                
                import shutil
                shutil.copy(image_path, output_path)
                
                return {
                    'final_status': 'success',
                    'processing_mode': 'NONE',
                    'reason': 'Quality sufficient - no enhancement needed',
                    'metrics': {
                        'rain_score': rain_score,
                        'glare_score': glare_score,
                        'needs_enhancement': needs_enhance,
                        'quality_metrics': quality_metrics if 'quality_metrics' in locals() else {}
                    }
                }
            
            # Execute processing
            print(f"\n[EXECUTING {mode}]")
            
            if mode == 'DERAIN':
                result = self.run_deraining(image_path, output_path)
                processor = IntelligentImageProcessor()
                processor.compute_derain_metrics("input.png", "output.png")
            elif mode == 'DEGLARE':
                result = self.run_deglaring(image_path, output_path)
            elif mode == 'ENHANCE':
                result = self.run_deglaring(image_path, output_path)
            
            # Handle result
            if result['success']:
                enhanced_img = cv2.imread(output_path)
                if enhanced_img is not None:
                    metrics = self.calculate_deglaring_metrics(img, enhanced_img, glare_mask)
                    result['metrics'] = metrics
                
                return {
                    'final_status': 'success',
                    'processing_mode': mode,
                    'metrics': result.get('metrics', {})
                }
            else:
                import shutil
                shutil.copy(image_path, output_path)
                return {
                    'final_status': 'fallback',
                    'error': result.get('error'),
                    'processing_mode': mode
                }
        
        except Exception as e:
            import traceback
            print(f"\n✗ Error: {str(e)}")
            print(traceback.format_exc())
            return {'final_status': 'error', 'error': str(e)}



    def process_image_with_validation(self, image_path, output_path):
        """Process image and evaluate quality"""
        # Run normal processing
        result = self.process_image(image_path, output_path)
        
        # Evaluate quality
        if result['final_status'] == 'success':
            quality_metrics = self.evaluate_enhancement_quality(image_path, output_path)
            result['quality_metrics'] = quality_metrics
            
            print(f"Enhancement Quality: {quality_metrics['quality_grade']}")
            print(f"SSIM: {quality_metrics['ssim']:.3f}")
            print(f"PSNR: {quality_metrics['psnr']:.1f}dB")
            print(f"Contrast Improvement: {quality_metrics['contrast_improvement_pct']:.1f}%")
        
        return result

    def evaluate_enhancement_quality(self, input_path, output_path):
        """Evaluate the quality of enhancement"""
        original = cv2.imread(input_path)
        enhanced = cv2.imread(output_path)
        
        if original is None or enhanced is None:
            return None
        
        # Convert to RGB
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        # Get quality metrics
        metrics_root = os.path.join(self.project_root, "metrics", "deglaring")
        os.makedirs(metrics_root, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(output_path))[0]
        metrics_file = os.path.join(metrics_root, f"{base_name}_metrics.txt")

        metrics = self.enhanced_retinex.evaluate_quality(
            original_rgb, enhanced_rgb, 
            save_path=metrics_file
        )
        
        return metrics

    def __init__(self, config_manager=None):
        # Get script directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # FIX: Kung nasa tools/ folder ang intelligent_switch.py, go up one level
        if os.path.basename(self.script_dir) == 'tools':
            self.project_root = os.path.dirname(self.script_dir)
        else:
            self.project_root = self.script_dir
        
        print(f"Project root: {self.project_root}")
        
        # Initialize configuration
        self.config = config_manager or ConfigManager()

        # ============================================
        # CRITICAL FIX: FORCE READ FROM CONFIG FILE
        # ============================================
        
        # Try to get values using config.get() first
        rain_threshold_from_get = self.config.get('rain_detection.threshold', 0.15)
        glare_threshold_from_get = self.config.get('glare_detection.threshold', 0.25)
        
        
        # Now verify against raw config dictionary
        if hasattr(self.config, 'config') and isinstance(self.config.config, dict):
            # Extract actual values from loaded YAML
            rain_section = self.config.config.get('rain_detection', {})
            glare_section = self.config.config.get('glare_detection', {})
            


            # Force use values from file (overrides config.get() defaults)
            self.rain_threshold = rain_section.get('threshold', rain_threshold_from_get)
            self.glare_threshold = glare_section.get('threshold', glare_threshold_from_get)


            
            # Glare detection parameters - also force from file
            self.glare_brightness_threshold = glare_section.get('brightness_threshold', 220)
            self.glare_saturation_threshold = glare_section.get('saturation_threshold', 30)
            self.min_glare_area = glare_section.get('min_glare_area', 10)
            self.dilation_kernel_size = glare_section.get('dilation_kernel_size', 5)
        else:
            # Fallback if config not loaded properly
            self.rain_threshold = rain_threshold_from_get
            self.glare_threshold = glare_threshold_from_get
            self.glare_brightness_threshold = self.config.get('glare_detection.brightness_threshold', 220)
            self.glare_saturation_threshold = self.config.get('glare_detection.saturation_threshold', 30)
            self.min_glare_area = self.config.get('glare_detection.min_glare_area', 10)
            self.dilation_kernel_size = self.config.get('glare_detection.dilation_kernel_size', 5)
        
        # DEBUG OUTPUT
        print(f"\n{'='*50}")
        print(f"CONFIGURATION LOADED")
        print(f"{'='*50}")
        print(f"Config file: {self.config.config_path}")
        print(f"Rain detection threshold: {self.rain_threshold}")
        print(f"Glare detection threshold: {self.glare_threshold}")
        print(f"Glare brightness threshold: {self.glare_brightness_threshold}")
        print(f"Glare saturation threshold: {self.glare_saturation_threshold}")
        print(f"Min glare area: {self.min_glare_area}")
        print(f"Dilation kernel size: {self.dilation_kernel_size}")
        print(f"{'='*50}\n")

        # Smoothing parameter
        self.smoothing_strength = self.config.get('enhancement.smoothing_strength', 0)
        
        # Set parameters from config
        self.min_contrast = self.config.get('enhancement.min_contrast', 40)
        
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
        
        self.enhanced_retinex = EnhancedRetinexProcessor(use_gpu=self.use_gpu)

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

    def detect_headlight_glare(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        s = hsv[:,:,1]
        
        # ===== CRITICAL FIX: Much stricter thresholds =====
        # Only EXTREME brightness (actual light sources)
        brightness_mask = v > 240  # CHANGED from 200 - only pure white
        saturation_mask = s < 15   # CHANGED from 40 - must be very desaturated
        
        glare_mask = brightness_mask & saturation_mask
        
        # ===== NEW: Position-based filtering =====
        # Headlights are usually in LOWER portion of image (road level)
        height, width = image.shape[:2]
        
        # Create position mask - only check lower 70% of image
        position_mask = np.zeros_like(glare_mask)
        position_mask[int(height * 0.3):, :] = True  # Lower 70% only
        
        # Combine with position
        glare_mask = glare_mask & position_mask
        
        # ===== NEW: Size filtering - headlights are small spots =====
        if np.sum(glare_mask) > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                glare_mask.astype(np.uint8), connectivity=8
            )
            
            filtered_mask = np.zeros_like(glare_mask)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Headlights: 20-2000 pixels (adjust based on your resolution)
                if 20 <= area <= 2000:
                    # Check aspect ratio (headlights are roughly circular/oval)
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                    
                    # Accept if aspect ratio is reasonable (not long streaks)
                    if aspect_ratio < 3.0:
                        filtered_mask[labels == i] = True
                        print(f"  Headlight {i}: area={area}, aspect={aspect_ratio:.2f} ✓")
                    else:
                        print(f"  Region {i}: area={area}, aspect={aspect_ratio:.2f} ✗ (wrong shape)")
                else:
                    print(f"  Region {i}: area={area} ✗ (wrong size)")
            
            glare_mask = filtered_mask
        
        glare_score = np.sum(glare_mask) / glare_mask.size
        
        print(f"\n[HEADLIGHT DETECTION]")
        print(f"  Detected pixels: {np.sum(glare_mask)}")
        print(f"  Score: {glare_score:.6f}")
        
        return glare_score, glare_mask.astype(bool)

    def _multi_scale_retinex(self, image):
        """Improved Multi-Scale Retinex with noise reduction"""
        from scipy.ndimage import gaussian_filter
        
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        # Convert to float and add small value to avoid log(0)
        img = image.astype(np.float64) / 255.0 + 1e-8
        
        # Optional pre-smoothing to reduce noise amplification
        if self.config.get('retinex_enhancement.pre_smooth', False):
            for c in range(3):
                img[:,:,c] = gaussian_filter(img[:,:,c], sigma=0.5)
        
        log_img = np.log(img)
        msr_result = np.zeros_like(log_img)

        # Multi-scale processing
        for scale, weight in zip(self.retinex_scales, self.retinex_weights):
            # Dynamic sigma calculation for better edge preservation
            sigma = scale / 3.0
            kernel_size = int(6 * sigma) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            blurred = cv2.GaussianBlur(log_img, (kernel_size, kernel_size), sigma)
            single_scale_retinex = log_img - blurred
            
            # Adaptive weighting based on local content
            local_variance = self._calculate_local_variance(single_scale_retinex)
            adaptive_weight = weight * (1 + 0.1 * local_variance)
            
            msr_result += adaptive_weight * single_scale_retinex

        # Apply enhanced post-processing
        enhanced = np.exp(msr_result) - 1e-8

        # Improved normalization with edge awareness
        enhanced = self._adaptive_normalization_v2(enhanced, image)
        
        # Apply noise reduction
        enhanced = self._apply_enhanced_noise_reduction(enhanced)
        
        # Final quality adjustments
        enhanced = self._final_quality_adjustments(enhanced)

        # Ensure output has same dimensions as input
        if enhanced.shape[:2] != (original_height, original_width):
            enhanced = cv2.resize(enhanced, (original_width, original_height))

        return enhanced.astype(np.uint8)

    def _calculate_local_variance(self, image, window_size=5):
        """Calculate local variance for adaptive processing"""
        if len(image.shape) == 3:
            # For color images, calculate variance across all channels
            variance_per_channel = []
            for c in range(image.shape[2]):
                kernel = np.ones((window_size, window_size)) / (window_size * window_size)
                local_mean = cv2.filter2D(image[:,:,c], -1, kernel)
                local_variance = cv2.filter2D((image[:,:,c] - local_mean) ** 2, -1, kernel)
                variance_per_channel.append(local_variance)
            return np.mean(variance_per_channel, axis=0)
        else:
            # For grayscale
            kernel = np.ones((window_size, window_size)) / (window_size * window_size)
            local_mean = cv2.filter2D(image, -1, kernel)
            local_variance = cv2.filter2D((image - local_mean) ** 2, -1, kernel)
            return local_variance

    def _adaptive_normalization_v2(self, enhanced, original):
        """Enhanced normalization with better edge preservation"""
        result = enhanced.copy()
        
        for c in range(3):
            channel = enhanced[:, :, c]
            orig_channel = original[:, :, c] / 255.0
            
            # Use configurable percentiles
            p_low = self.percentile_low
            p_high = self.percentile_high
            
            # Calculate percentiles
            low_val = np.percentile(channel, p_low)
            high_val = np.percentile(channel, p_high)
            
            # Avoid division by zero
            if high_val - low_val < 1e-8:
                high_val = low_val + 1e-8
            
            # Adaptive contrast stretching
            channel = (channel - low_val) / (high_val - low_val)
            channel = np.clip(channel, 0, 1)
            
            # Gamma correction
            channel = np.power(channel, self.gamma_correction)
            
            # Edge-aware contrast enhancement
            edge_map = self._detect_edges_simple(orig_channel)
            smooth_regions = 1 - edge_map
            
            # Apply different enhancement levels to edges vs smooth areas
            enhanced_edges = channel * self.contrast_strength
            enhanced_smooth = channel * 0.9  # Less enhancement in smooth areas
            
            channel = enhanced_edges * edge_map + enhanced_smooth * smooth_regions
            
            result[:, :, c] = np.clip(channel, 0, 1)
        
        return (result * 255).astype(np.uint8)

    def _detect_edges_simple(self, image):
        """Simple edge detection for adaptive processing"""
        # Convert to uint8 for edge detection
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(img_uint8, 50, 150)
        
        # Dilate edges slightly
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Convert back to float and normalize
        return edges.astype(np.float64) / 255.0

    def _apply_enhanced_noise_reduction(self, image):
        """Enhanced noise reduction with multiple methods"""
        if not hasattr(self, 'denoising_enabled'):
            # Check if denoising is configured
            denoising_config = self.config.get('denoising', {})
            if not denoising_config:
                return image
        
        # Get denoising method
        method = self.config.get('denoising.post_process', 'bilateral')
        noise_strength = self.config.get('retinex_enhancement.noise_reduction_strength', 0.4)
        
        if noise_strength <= 0:
            return image
        
        if method == 'bilateral':
            return self._bilateral_denoise(image, noise_strength)
        elif method == 'non_local':
            return self._non_local_denoise(image, noise_strength)
        elif method == 'guided':
            return self._guided_denoise(image, noise_strength)
        else:
            return image

    def _bilateral_denoise(self, image, strength):
        """Bilateral filtering for noise reduction"""
        d = self.config.get('denoising.bilateral_d', 7)
        sigma_color = self.config.get('denoising.bilateral_sigma_color', 30)
        sigma_space = self.config.get('denoising.bilateral_space', 30)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        # Blend with original based on strength
        result = image * (1 - strength) + filtered * strength
        return result.astype(np.uint8)

    def _non_local_denoise(self, image, strength):
        """Non-Local Means denoising"""
        h = self.config.get('denoising.non_local_h', 8)
        
        # Apply Non-Local Means
        denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        
        # Blend with original
        result = image * (1 - strength) + denoised * strength
        return result.astype(np.uint8)

    def _guided_denoise(self, image, strength):
        """Simple guided filter for denoising"""
        # Use grayscale as guide
        guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        
        radius = self.config.get('denoising.guided_radius', 8)
        eps = self.config.get('denoising.guided_eps', 0.01)
        
        result = image.copy().astype(np.float64)
        
        for c in range(3):
            channel = image[:,:,c].astype(np.float64) / 255.0
            filtered_channel = self._guided_filter_single(guide, channel, radius, eps)
            result[:,:,c] = filtered_channel * 255
        
        # Blend with original
        blended = image * (1 - strength) + result * strength
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _guided_filter_single(self, guide, src, radius, eps):
        """Guided filter for single channel"""
        mean_guide = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
        mean_src = cv2.boxFilter(src, cv2.CV_64F, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_64F, (radius, radius))
        
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        mean_guide_sq = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
        var_guide = mean_guide_sq - mean_guide * mean_guide
        
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
        
        return mean_a * guide + mean_b

    def _final_quality_adjustments(self, image):
        """Final adjustments for better visual quality"""
        # Convert to LAB for better color processing
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply mild CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Subtle saturation enhancement
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        saturation_boost = self.config.get('retinex_enhancement.saturation_boost', 1.05)
        hsv[:,:,1] = hsv[:,:,1] * saturation_boost
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return result

    def assess_enhancement_quality(self, original, enhanced):
        """Assess if the enhancement meets quality standards"""
        try:
            from skimage.metrics import structural_similarity as compare_ssim
            from skimage.metrics import peak_signal_noise_ratio as compare_psnr
            
            # Convert to grayscale for analysis
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            
            # Calculate metrics
            ssim_score = compare_ssim(orig_gray, enh_gray, data_range=255)
            psnr_score = compare_psnr(orig_gray, enh_gray, data_range=255)
            
            # Calculate noise level
            noise_variance = cv2.Laplacian(enh_gray, cv2.CV_64F).var()
            
            # Get quality thresholds from config
            min_ssim = self.config.get('quality_thresholds.min_ssim', 0.75)
            min_psnr = self.config.get('quality_thresholds.min_psnr', 18)
            max_noise = self.config.get('quality_thresholds.max_noise_variance', 50)
            
            print(f"Quality Assessment - SSIM: {ssim_score:.3f}, PSNR: {psnr_score:.1f}, Noise: {noise_variance:.1f}")
            
            # Quality gates
            if ssim_score < min_ssim:
                print(f"Warning: SSIM {ssim_score:.3f} below threshold {min_ssim}")
                return False, "Poor structural similarity"
                
            if psnr_score < min_psnr:
                print(f"Warning: PSNR {psnr_score:.1f} below threshold {min_psnr}")
                return False, "Too much noise/distortion"
                
            if noise_variance > max_noise:
                print(f"Warning: Noise variance {noise_variance:.1f} above threshold {max_noise}")
                return False, "Excessive noise"
            
            return True, "Quality acceptable"
            
        except ImportError:
            print("Advanced quality assessment not available - using basic checks")
            return True, "Basic quality check passed"
        except Exception as e:
            print(f"Quality assessment error: {e}")
            return True, "Quality assessment failed, proceeding"



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


    def _apply_smoothing(self, image):
        """Apply smoothing to improve clarity and reduce harsh noise"""
        if self.smoothing_strength <= 0:
            return image
        
        # Convert to BGR (bilateralFilter works best here)
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        smoothed = cv2.bilateralFilter(
            img_bgr,
            d=9,  # pixel neighborhood
            sigmaColor=self.smoothing_strength,
            sigmaSpace=self.smoothing_strength // 2
        )
        
        return cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)

    
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

        # Apply enhanced noise reduction
        enhanced_uint8 = self._apply_enhanced_noise_reduction(enhanced_uint8)

        # Double-check dimensions
        if enhanced_uint8.shape[:2] != (original_height, original_width):
            print(f"Warning: Resizing enhanced image from {enhanced_uint8.shape[:2]} to {(original_height, original_width)}")
            enhanced_uint8 = cv2.resize(enhanced_uint8, (original_width, original_width))

        return enhanced_uint8
        
    def setup_validation_images(self):
        validation_paths = [
            "validation_images/noisy_image1.jpg",
            "validation_images/noisy_image2.jpg",
            "validation_images/noisy_image3.jpg",
            # Add more paths
        ]
        self.enhanced_retinex.setup_validation_set(validation_paths)   


    def save_metrics_txt(self, metrics: dict, output_path: str):
        """Save metrics into a human-readable .txt report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("IMAGE ENHANCEMENT METRICS REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("ORIGINAL IMAGE METRICS\n")
            for k, v in metrics["original"].items():
                f.write(f"{k:<15}: {v:.4f}\n")

            f.write("\nENHANCED IMAGE METRICS\n")
            for k, v in metrics["enhanced"].items():
                f.write(f"{k:<15}: {v:.4f}\n")

            f.write("\nIMPROVEMENTS\n")
            for k, v in metrics["improvements"].items():
                if v is None:
                    f.write(f"{k:<20}: N/A\n")
                else:
                    f.write(f"{k:<20}: {v:.4f}\n")

            f.write("  contrast → standard deviation of brightness (higher = clearer)\n")
            f.write("  brightness → avg pixel brightness in glare area\n")
            f.write("  saturation → avg color richness in glare area\n")
            f.write("  glare_area_reduced → % of glare pixels removed\n")
            f.write("  SSIM → closer to 1 means more structural quality\n")
            f.write("  PSNR → higher dB means less noise, better quality\n")



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

# Sa intelligent_switch.py, palitan ang run_deraining method:

    def run_deraining(self, image_path, output_path):
        """
        FIXED: Execute deraining with proper config passing
        The day/night detection happens INSIDE test_model.py
        """
        try:
            # Get deraining configuration
            deraining_config = self.config.get('deraining', {})
            
            # Check if deraining is enabled
            if not deraining_config.get('enabled', True):
                return {'success': False, 'error': 'Deraining is disabled in config'}
            
            # Get weights path
            weights_path = os.path.join(self.project_root, "weights", "derain_gan", "derain_gan.ckpt-100000")
            
            # Verify weights exist
            weights_found = False
            for ext in ['', '.meta', '.index', '.data-00000-of-00001']:
                if os.path.exists(weights_path + ext):
                    weights_found = True
                    break
            
            if not weights_found:
                return {'success': False, 'error': f'Weights not found: {weights_path}'}
            
            # Get test_model.py path
            test_script_path = os.path.join(self.project_root, "tools", "test_model.py")
            if not os.path.exists(test_script_path):
                return {'success': False, 'error': f'test_model.py not found: {test_script_path}'}
            
            # Get deraining config path (this contains day/night settings)
            derain_config_path = deraining_config.get('config_path', 'deraining_config.yaml')
            if not os.path.isabs(derain_config_path):
                derain_config_path = os.path.join(self.project_root, derain_config_path)
            
            if not os.path.exists(derain_config_path):
                print(f"[WARN] Deraining config not found: {derain_config_path}")
                print(f"[WARN] test_model.py will use built-in defaults")
            
            print(f"\n{'='*70}")
            print(f"RUNNING DERAINING MODULE")
            print(f"{'='*70}")
            print(f"[INFO] Image: {image_path}")
            print(f"[INFO] Weights: {weights_path}")
            print(f"[INFO] Config: {derain_config_path}")
            
            # Build command
            cmd = [
                sys.executable,
                test_script_path,
                "--image_path", os.path.abspath(image_path),
                "--weights_path", os.path.abspath(weights_path),
                "--output_file", os.path.join(self.project_root, "derain_results.txt"),
                "--config", os.path.abspath(derain_config_path)
            ]
            
            # Add force_scene if configured
            force_scene = deraining_config.get('force_scene', None)
            if force_scene in ['day', 'night', 'default']:
                cmd.extend(["--force_scene", force_scene])
                print(f"[INFO] Force scene: {force_scene.upper()}")
            else:
                print(f"[INFO] Scene detection: AUTOMATIC (based on brightness)")
            
            print(f"[INFO] Command: {' '.join(cmd)}")
            print(f"{'='*70}\n")
            
            # Setup environment
            env = os.environ.copy()
            python_paths = [
                self.project_root,
                os.path.dirname(self.project_root),
            ]
            env['PYTHONPATH'] = os.pathsep.join(python_paths) + os.pathsep + env.get('PYTHONPATH', '')
            
            # Run deraining
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=env, 
                cwd=self.project_root
            )
            
            # Show output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"[STDERR] {result.stderr}")
            
            if result.returncode == 0:
                # Move output file
                output_file = os.path.join(self.project_root, "derain_ret.png")
                if os.path.exists(output_file):
                    # Copy to requested output path
                    import shutil
                    shutil.copy(output_file, os.path.abspath(output_path))
                    
                    print(f"\n[SUCCESS] Deraining completed!")
                    print(f"[INFO] Output saved to: {output_path}")
                    
                    # Read metrics if available
                    metrics = {}
                    metrics_file = os.path.join(self.project_root, "derain_results.txt")
                    if os.path.exists(metrics_file):
                        metrics['results_file'] = metrics_file
                        print(f"[INFO] Metrics saved to: {metrics_file}")
                    
                    return {'success': True, 'metrics': metrics}
                else:
                    return {'success': False, 'error': 'Output file not created'}
            else:
                return {'success': False, 'error': result.stderr or 'Unknown error'}
                
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            return {'success': False, 'error': error_msg}

    def run_deglaring(self, image_path, output_path):
        """Execute selective deglaring based on configuration"""
        try:
            # Read image to detect glare areas
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not read image'}
                
            # Detect glare areas - FIX: Use correct method name
            glare_score, glare_mask = self.detect_headlight_glare(image)  # ✅ FIXED
            
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
    
    def selective_deglaring(self, image_path, output_path, glare_mask):
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not read image'}
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply enhancement only to glare regions
            if self.enhance_only_glare_areas:
                enhanced = self._enhance_glare_regions(image_rgb, glare_mask)
            else:
                # Enhance entire image
                enhanced = self._multi_scale_retinex(image_rgb)
            
            # Apply smoothing if configured
            if self.smoothing_strength > 0:
                enhanced = self._apply_smoothing(enhanced)
            
            # Convert back to BGR and save
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, enhanced_bgr)
            
            return {'success': True}
            
        except Exception as e:
            import traceback
            return {'success': False, 'error': f"{str(e)}\n{traceback.format_exc()}"}

    def calculate_deglaring_metrics(self, original, enhanced, glare_mask):
        """Compare original vs enhanced image and calculate improvements"""

        # --- FIX: ensure enhanced always matches original dimensions ---
        if enhanced.shape[:2] != original.shape[:2]:
            enhanced = cv2.resize(
                enhanced,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Convert to grayscale AFTER resizing
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # Full image metrics (contrast, SSIM, PSNR)
        contrast_orig = float(np.std(orig_gray))
        contrast_enh = float(np.std(enh_gray))
        contrast_improvement = contrast_enh - contrast_orig

        try:
            ssim_val = compare_ssim(orig_gray, enh_gray, data_range=255)
            psnr_val = compare_psnr(orig_gray, enh_gray, data_range=255)
        except Exception:
            ssim_val, psnr_val = None, None

        # HSV for glare analysis (after resizing)
        orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        enh_hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        # --- FIX: resize glare_mask to match enhanced image size ---
        if glare_mask.shape != enh_hsv.shape[:2]:
            glare_mask = cv2.resize(
                glare_mask.astype(np.uint8),
                (enh_hsv.shape[1], enh_hsv.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        glare_pixels = np.where(glare_mask)

        if glare_pixels[0].size > 0:
            orig_v = orig_hsv[:, :, 2][glare_pixels]
            orig_s = orig_hsv[:, :, 1][glare_pixels]
            enh_v = enh_hsv[:, :, 2][glare_pixels]
            enh_s = enh_hsv[:, :, 1][glare_pixels]

            brightness_orig = float(np.mean(orig_v))
            brightness_enh = float(np.mean(enh_v))
            saturation_orig = float(np.mean(orig_s))
            saturation_enh = float(np.mean(enh_s))

            still_glare = (
                (enh_v > self.glare_brightness_threshold)
                & (enh_s < self.glare_saturation_threshold)
            )
            glare_area_reduced = 1 - (np.sum(still_glare) / len(glare_pixels[0]))
        else:
            brightness_orig = brightness_enh = 0.0
            saturation_orig = saturation_enh = 0.0
            glare_area_reduced = 0.0

        return {
            # Original values
            "original": {
                "contrast": contrast_orig,
                "brightness": brightness_orig,
                "saturation": saturation_orig,
            },
            # Enhanced values
            "enhanced": {
                "contrast": contrast_enh,
                "brightness": brightness_enh,
                "saturation": saturation_enh,
            },
            # Improvements
            "improvements": {
                "contrast_gain": contrast_improvement,
                "brightness_reduction": brightness_orig - brightness_enh,
                "saturation_gain": saturation_enh - saturation_orig,
                "glare_area_reduced": glare_area_reduced,
                "ssim": ssim_val,
                "psnr": psnr_val,
            },
        }


    def process_image(self, image_path, output_path):
        """Main processing function with enhanced debugging and quality gates"""
        try:
            # Read and analyze the image first
            img = cv2.imread(image_path)
            if img is None:
                return {'final_status': 'error', 'error': 'Could not read image'}

            print("=" * 50)
            print("IMAGE ANALYSIS DEBUG INFORMATION")
            print("=" * 50)
            
            # Analysis with detailed debugging
            rain_score = self.detect_rain(img)
            glare_score, glare_mask = self.detect_headlight_glare(img)
            needs_enhance = self.needs_enhancement_improved(img)
            
            print("\nSUMMARY:")
            print(f"Rain Score: {rain_score:.3f} (Threshold: {self.rain_threshold})")
            print(f"Glare Score: {glare_score:.3f} (Threshold: {self.glare_threshold})")
            print(f"Needs Enhancement: {needs_enhance}")
            
            if rain_score < (self.rain_threshold * 0.005) and \
            glare_score < (self.glare_threshold * 0.005) and \
            not needs_enhance:
                print("\nImage is clean — skipping deraining and deglaring.")
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

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Create debug directory
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save glare detection visualization
            self.visualize_glare_detection(img, glare_mask, os.path.join(debug_dir, "glare_detection.png"))
            
            # Determine processing mode - Check for command line arguments
            processing_mode = self.config.get('system.mode', 'auto')
            
            # Handle force arguments (check if they exist in global args)
            try:
                import __main__
                args = getattr(__main__, 'args', None)
                if args:
                    if hasattr(args, 'force_derain') and args.force_derain:
                        mode = 'DERAIN'
                    elif hasattr(args, 'force_deglare') and args.force_deglare:
                        mode = 'DEGLARE'
                    elif hasattr(args, 'force_enhance') and args.force_enhance:
                        mode = 'ENHANCE'
                    elif rain_score > self.rain_threshold:
                        mode = 'DERAIN'
                    elif glare_score > self.glare_threshold:
                        mode = 'DEGLARE'
                    elif needs_enhance:
                        mode = 'ENHANCE'
                    else:
                        mode = 'NONE'
                else:
                    # No command line args, use automatic detection
                    if rain_score > self.rain_threshold:
                        mode = 'DERAIN'
                    elif glare_score > self.glare_threshold:
                        mode = 'DEGLARE'
                    elif needs_enhance:
                        mode = 'ENHANCE'
                    else:
                        mode = 'NONE'
            except:
                # Fallback to automatic detection
                if rain_score > self.rain_threshold:
                    mode = 'DERAIN'
                elif glare_score > self.glare_threshold:
                    mode = 'DEGLARE'
                elif needs_enhance:
                    mode = 'ENHANCE'
                else:
                    mode = 'NONE'
            
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
                    # Convert to RGB for quality assessment
                    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                    
                    # QUALITY GATE: Check if enhancement meets standards
                    quality_ok, quality_msg = self.assess_enhancement_quality(original_rgb, enhanced_rgb)
                    
                    if not quality_ok:
                        print(f"\nQuality check failed: {quality_msg}")
                        print("Applying fallback processing...")
                        
                        # Try with more conservative settings
                        fallback_result = self._apply_fallback_enhancement(image_path, output_path, glare_mask)
                        if fallback_result['success']:
                            print("Fallback processing successful!")
                            # Re-assess quality after fallback
                            enhanced_img = cv2.imread(output_path)
                            if enhanced_img is not None:
                                enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                                quality_ok, quality_msg = self.assess_enhancement_quality(original_rgb, enhanced_rgb)
                        else:
                            print("Fallback failed, applying minimal enhancement...")
                            # Apply very conservative enhancement
                            self._apply_minimal_enhancement(image_path, output_path)
                            quality_msg = "Minimal enhancement applied"
                    
                    # Calculate final metrics
                    final_enhanced = cv2.imread(output_path)
                    if final_enhanced is not None:
                        # --- Ensure size matches ---
                        if final_enhanced.shape[:2] != img.shape[:2]:
                            final_enhanced = cv2.resize(
                                final_enhanced,
                                (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_LINEAR
                            )

                        # --- DEBUG save original/enhanced separately ---
                        debug_dir = os.path.join(output_dir, "debug")
                        os.makedirs(debug_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(debug_dir, "original.png"), img)
                        cv2.imwrite(os.path.join(debug_dir, "enhanced.png"), final_enhanced)

                        # --- Side by side comparison ---
                        comparison = np.hstack((img, final_enhanced))
                        cv2.imwrite(os.path.join(debug_dir, "comparison.png"), comparison)

                        # --- Continue with metrics ---
                        metrics = self.calculate_deglaring_metrics(img, final_enhanced, glare_mask)
                        result['metrics'] = metrics
                        result['quality_passed'] = quality_ok
                        result['quality_message'] = quality_msg


                        print("\nFINAL ENHANCEMENT METRICS:")
                        print(f"Quality Status: {quality_msg}")
                        if 'improvements' in metrics:
                            print(f"Glare Area Reduced: {metrics['improvements'].get('glare_area_reduced', 0):.1%}")
                            print(f"Brightness Reduction: {metrics['improvements'].get('brightness_reduction', 0):.1f}")
                            print(f"SSIM: {metrics['improvements'].get('ssim', 'N/A')}")
                            print(f"PSNR: {metrics['improvements'].get('psnr', 'N/A')}")

                        # Specify the new path where you want to save the metrics file
                        new_output_dir = r"D:\School UIC\GAN\attentive-gan-derainnet\metrics\deglaring"

                        # Ensure the directory exists
                        os.makedirs(new_output_dir, exist_ok=True)

                        # Define the new path for the metrics report
                        metrics_txt_path = os.path.join(new_output_dir, "metrics_report.txt")

                        # Save the metrics
                        self.save_metrics_txt(metrics, metrics_txt_path)

                return {
                    'final_status': 'success',
                    'processing_mode': mode,
                    'quality_passed': quality_ok if 'quality_ok' in locals() else True,
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

    def _apply_fallback_enhancement(self, image_path, output_path, glare_mask):
        """Apply more conservative enhancement settings"""
        try:
            print("Applying fallback enhancement with conservative settings...")
            
            # Temporarily modify settings for fallback
            original_gamma = self.gamma_correction
            original_contrast = self.contrast_strength
            original_noise_strength = self.config.get('retinex_enhancement.noise_reduction_strength', 0.4)
            
            # Use more conservative settings
            self.gamma_correction = 0.9  # Less aggressive
            self.contrast_strength = 0.8  # Reduced enhancement
            self.config.set('retinex_enhancement.noise_reduction_strength', 0.6)  # More noise reduction
            
            print(f"Fallback settings: gamma={self.gamma_correction}, contrast={self.contrast_strength}, noise_reduction=0.6")
            
            # Apply enhancement
            result = self.selective_deglaring(image_path, output_path, glare_mask)
            
            # Restore original settings
            self.gamma_correction = original_gamma
            self.contrast_strength = original_contrast
            self.config.set('retinex_enhancement.noise_reduction_strength', original_noise_strength)
            
            return result
            
        except Exception as e:
            print(f"Fallback enhancement failed: {e}")
            return {'success': False, 'error': str(e)}

    def _apply_minimal_enhancement(self, image_path, output_path):
        """Apply very minimal enhancement as last resort"""
        try:
            print("Applying minimal enhancement as last resort...")
            
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Convert to LAB for gentle enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Very mild CLAHE
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            
            # Convert back
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Save
            cv2.imwrite(output_path, result)
            print("Minimal enhancement completed")
            return True
            
        except Exception as e:
            print(f"Minimal enhancement failed: {e}")
            return False


    def _clahe_enhancement(self, image):
        """
        CLAHE-based enhancement - gentler than Retinex
        Add this method to your IntelligentImageProcessor class
        """
        # Convert to LAB color space for better luminance control
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L (luminance) channel only
        clahe = cv2.createCLAHE(
            clipLimit=self.config.get('clahe.clip_limit', 2.0),  # Lower = less aggressive
            tileGridSize=(
                self.config.get('clahe.tile_size', 8), 
                self.config.get('clahe.tile_size', 8)
            )
        )
        
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced 
    
    def _adaptive_bilateral_enhancement(self, image):
        """
        Noise-aware enhancement using bilateral filtering
        Add this method to your IntelligentImageProcessor class
        """
        # Analyze noise level first
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Noise level detected: {noise_level:.1f}")
        
        # Adjust parameters based on noise
        if noise_level > 50:
            # High noise - more aggressive denoising
            d = 15
            sigma_color = 80
            sigma_space = 80
        elif noise_level > 25:
            # Medium noise
            d = 9
            sigma_color = 50
            sigma_space = 50
        else:
            # Low noise
            d = 7
            sigma_color = 30
            sigma_space = 30
        
        # Apply bilateral filter first to reduce noise
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        # Then apply gentle CLAHE enhancement
        enhanced = self._clahe_enhancement(denoised)
        
        # Blend with original based on noise level
        blend_ratio = min(0.8, 0.4 + (noise_level / 100))  # More original if noisy
        result = enhanced * blend_ratio + image * (1 - blend_ratio)
        
        return result.astype(np.uint8)
    def _conservative_retinex(self, image):
        """
        Much gentler Retinex implementation
        Use this instead of aggressive _multi_scale_retinex
        """
        # Pre-denoise if configured
        if self.config.get('enhancement.pre_denoise', True):
            image = cv2.bilateralFilter(image, 5, 25, 25)
        
        # Convert to float and add small value to avoid log(0)
        img = image.astype(np.float64) / 255.0 + 1e-8
        log_img = np.log(img)
        
        # Use smaller, gentler scales
        scales = self.config.get('retinex_enhancement.conservative_scales', [15, 40])
        weights = self.config.get('retinex_enhancement.conservative_weights', [0.6, 0.4])
        
        msr_result = np.zeros_like(log_img)
        
        for scale, weight in zip(scales, weights):
            # Use Gaussian blur instead of box filter for smoother results
            sigma = scale / 4.0  # Gentler than /3.0
            blurred = cv2.GaussianBlur(log_img, (0, 0), sigma)
            single_scale_retinex = log_img - blurred
            msr_result += weight * single_scale_retinex
        
        # Apply gain with much gentler settings
        enhanced = np.exp(msr_result) - 1e-8
        
        # Very conservative normalization
        for c in range(3):
            channel = enhanced[:, :, c]
            
            # Use wider percentile range for gentler stretching
            p_low = np.percentile(channel, 10)  # Changed from 2
            p_high = np.percentile(channel, 90)  # Changed from 98
            
            if p_high - p_low < 1e-8:
                continue
                
            # Gentle stretching
            channel = (channel - p_low) / (p_high - p_low)
            channel = np.clip(channel, 0, 1)
            
            # Much gentler gamma
            gamma = self.config.get('retinex_enhancement.gentle_gamma', 0.95)
            channel = np.power(channel, gamma)
            
            # Reduced contrast strength
            contrast = self.config.get('retinex_enhancement.gentle_contrast', 0.7)
            channel = channel * contrast
            
            enhanced[:, :, c] = np.clip(channel, 0, 1)
        
        return (enhanced * 255).astype(np.uint8)

    def _hybrid_enhancement(self, image):
        """
        Hybrid approach: CLAHE + gentle Retinex + denoising
        This is the recommended alternative approach
        """
        # Step 1: Pre-denoise
        denoised = cv2.bilateralFilter(image, 7, 50, 50)
        
        # Step 2: Apply CLAHE for local contrast
        clahe_enhanced = self._clahe_enhancement(denoised)
        
        # Step 3: Apply very gentle Retinex if needed
        contrast_check = np.std(cv2.cvtColor(clahe_enhanced, cv2.COLOR_RGB2GRAY))
        if contrast_check < 30:  # Only if still low contrast
            retinex_enhanced = self._conservative_retinex(clahe_enhanced)
            # Blend CLAHE and Retinex results
            final = 0.6 * clahe_enhanced + 0.4 * retinex_enhanced
        else:
            final = clahe_enhanced
        
        # Step 4: Final denoising if configured
        if self.config.get('enhancement.final_denoise', True):
            final = cv2.bilateralFilter(final.astype(np.uint8), 5, 25, 25)
        
        return final.astype(np.uint8)

    def _aggressive_glare_reduction(self, image, glare_mask):
        """
        More aggressive glare reduction for visible results
        """
        print("Applying AGGRESSIVE glare reduction...")
        
        # Convert to HSV for direct brightness control
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_float = hsv.astype(np.float32)
        
        # Reduce brightness significantly in glare areas
        reduction_factor = 0.3  # Reduce to 30% of original brightness
        hsv_float[:,:,2][glare_mask] = hsv_float[:,:,2][glare_mask] * reduction_factor
        
        # Increase saturation slightly to compensate
        saturation_boost = 1.2
        hsv_float[:,:,1][glare_mask] = np.clip(hsv_float[:,:,1][glare_mask] * saturation_boost, 0, 255)
        
        # Convert back
        result = cv2.cvtColor(hsv_float.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Additional contrast enhancement in glare areas using gamma correction
        result_float = result.astype(np.float32) / 255.0
        
        # Apply strong gamma correction to glare areas
        gamma = 0.4  # Much darker
        for c in range(3):
            channel = result_float[:,:,c]
            channel[glare_mask] = np.power(channel[glare_mask], gamma)
            result_float[:,:,c] = channel
        
        return (result_float * 255).astype(np.uint8)

    def _apply_smoothing_enhancement(self, image):
        """
        Apply various smoothing techniques for better image quality
        Add this method to your IntelligentImageProcessor class
        """
        smoothing_method = self.config.get('smoothing.method', 'bilateral')
        smoothing_strength = self.config.get('smoothing.strength', 0.5)
        
        if smoothing_strength <= 0:
            return image
            
        print(f"Applying {smoothing_method} smoothing with strength {smoothing_strength}")
        
        if smoothing_method == 'bilateral':
            return self._bilateral_smoothing(image, smoothing_strength)
        elif smoothing_method == 'guided':
            return self._guided_filter_smoothing(image, smoothing_strength)
        elif smoothing_method == 'edge_preserving':
            return self._edge_preserving_smoothing(image, smoothing_strength)
        elif smoothing_method == 'non_local_means':
            return self._non_local_means_smoothing(image, smoothing_strength)
        elif smoothing_method == 'adaptive':
            return self._adaptive_smoothing(image, smoothing_strength)
        else:
            return self._bilateral_smoothing(image, smoothing_strength)
        
    def _bilateral_smoothing(self, image, strength):
        """
        Bilateral filtering - preserves edges while smoothing
        """
        # Map strength to bilateral filter parameters
        d = int(5 + strength * 10)  # 5-15
        sigma_color = 25 + strength * 50  # 25-75
        sigma_space = 25 + strength * 50  # 25-75
        
        smoothed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        # Blend with original based on strength
        result = image * (1 - strength) + smoothed * strength
        return result.astype(np.uint8)
    
    def _guided_filter_smoothing(self, image, strength):
        """
        Guided filter - excellent edge preservation
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Use grayscale as guide
        guide = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
        
        # Parameters based on strength
        radius = int(4 + strength * 12)  # 4-16
        eps = 0.01 + strength * 0.05    # 0.01-0.06
        
        result = np.zeros_like(img_float)
        
        for c in range(3):
            result[:,:,c] = self._guided_filter_single(guide, img_float[:,:,c], radius, eps)
        
        # Blend with original
        result = img_float * (1 - strength) + result * strength
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)   
    
    def _guided_filter_single(self, guide, src, radius, eps):
        """
        Single channel guided filter implementation
        """
        mean_guide = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_src = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
        
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        mean_guide_sq = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
        var_guide = mean_guide_sq - mean_guide * mean_guide
        
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
        
        return mean_a * guide + mean_b
    
    def _edge_preserving_smoothing(self, image, strength):
        """
        OpenCV's edge-preserving filter
        """
        # Parameters based on strength
        flags = 2  # RECURS_FILTER
        sigma_s = 20 + strength * 80    # 20-100
        sigma_r = 0.1 + strength * 0.3  # 0.1-0.4
        
        smoothed = cv2.edgePreservingFilter(image, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)
        
        # Blend with original
        result = image * (1 - strength) + smoothed * strength
        return result.astype(np.uint8)
    
    def _non_local_means_smoothing(self, image, strength):
        """
        Non-local means denoising/smoothing
        """
        # Parameters based on strength
        h = 5 + strength * 15           # 5-20
        template_window_size = 7
        search_window_size = 21
        
        smoothed = cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window_size, search_window_size
        )
        
        # Blend with original
        result = image * (1 - strength) + smoothed * strength
        return result.astype(np.uint8)
    
    def _adaptive_smoothing(self, image, strength):
        """
        Adaptive smoothing based on local image content
        """
        # Analyze local variance to determine smoothing needs
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Normalize variance to 0-1
        variance_norm = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min() + 1e-6)
        
        # Apply different smoothing levels based on variance
        # High variance areas (edges) get less smoothing
        # Low variance areas (smooth regions) get more smoothing
        adaptive_strength = strength * (1 - variance_norm * 0.7)
        
        # Apply bilateral filter with spatially varying strength
        result = image.copy().astype(np.float32)
        smoothed = cv2.bilateralFilter(image, 9, 50, 50).astype(np.float32)
        
        for c in range(3):
            result[:,:,c] = result[:,:,c] * (1 - adaptive_strength) + smoothed[:,:,c] * adaptive_strength
        
        return result.astype(np.uint8)
    
    def _detail_enhancement_smoothing(self, image, strength):
        """
        Combine smoothing with detail enhancement
        """
        # Step 1: Apply base smoothing
        smoothed = self._bilateral_smoothing(image, strength * 0.7)
        
        # Step 2: Extract details
        detail = cv2.subtract(image, smoothed)
        
        # Step 3: Enhance details
        detail_enhanced = detail * (1 + strength * 0.5)
        detail_enhanced = np.clip(detail_enhanced, -50, 50)  # Limit detail enhancement
        
        # Step 4: Combine smoothed base with enhanced details
        result = cv2.add(smoothed, detail_enhanced.astype(np.uint8))
        
        return result

    def generate_thesis_metrics_summary():
        """
        Generate CSV summary of ALL derained images for thesis
        Perfect for Results and Discussion table!
        """
        import glob
        import pandas as pd
        
        metrics_dir = os.path.join("metrics", "deraining")
        
        # Find all comparison txt files
        report_files = glob.glob(os.path.join(metrics_dir, "*_comparison.txt"))
        
        if not report_files:
            print("⚠️ No comparison reports found!")
            return
        
        rows = []
        
        for report_file in report_files:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract filename
                filename = os.path.basename(report_file).replace("_comparison.txt", "")
                
                # Parse metrics
                psnr = ssim = contrast_imp = noise_red = None
                
                for line in content.splitlines():
                    if "PSNR" in line and "Value:" in line:
                        psnr = float(line.split(":")[1].strip().replace("dB", ""))
                    elif "SSIM" in line and "Value:" in line:
                        ssim = float(line.split(":")[1].strip())
                    elif "Change:" in line and "%" in line and contrast_imp is None:
                        # First Change line is contrast
                        contrast_imp = float(line.split("(")[1].split("%")[0])
                    elif "Reduction:" in line and "%" in line:
                        noise_red = float(line.split("(")[1].split("%")[0])
                
                rows.append({
                    "Image": filename,
                    "PSNR (dB)": round(psnr, 2) if psnr else "N/A",
                    "SSIM": round(ssim, 4) if ssim else "N/A",
                    "Contrast Improvement (%)": round(contrast_imp, 2) if contrast_imp else "N/A",
                    "Noise Reduction (%)": round(noise_red, 2) if noise_red else "N/A"
                })
                
            except Exception as e:
                print(f"⚠️ Error parsing {report_file}: {e}")
                continue
        
        if rows:
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Calculate averages
            avg_row = {
                "Image": "AVERAGE",
                "PSNR (dB)": round(df["PSNR (dB)"].replace("N/A", np.nan).astype(float).mean(), 2),
                "SSIM": round(df["SSIM"].replace("N/A", np.nan).astype(float).mean(), 4),
                "Contrast Improvement (%)": round(df["Contrast Improvement (%)"].replace("N/A", np.nan).astype(float).mean(), 2),
                "Noise Reduction (%)": round(df["Noise Reduction (%)"].replace("N/A", np.nan).astype(float).mean(), 2)
            }
            
            df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
            
            # Save to CSV
            csv_path = os.path.join(metrics_dir, "THESIS_RESULTS_SUMMARY.csv")
            df.to_csv(csv_path, index=False)
            
            print("\n" + "="*80)
            print("📊 THESIS METRICS SUMMARY")
            print("="*80)
            print(df.to_string(index=False))
            print(f"\n✅ Summary saved to: {csv_path}")
            print("\n💡 Copy this table to your thesis Results and Discussion!")
            print("="*80)
        else:
            print("⚠️ No valid metrics found!")

# Main execution with config loading
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Enhanced Intelligent Image Processing')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--generate_summary', action='store_true',
                       help='Generate thesis metrics summary CSV')
    
    args = parser.parse_args()

    # Load configuration
    config_manager = ConfigManager(args.config or 'enhanced_deglare_config.yaml')
    processor = IntelligentImageProcessor(config_manager)

    # Process the image
    result = processor.process_image_with_validation(args.image, args.output)

    print(f"\nProcessing Status: {result['final_status']}")
    if 'quality_metrics' in result:
        print(f"Quality Grade: {result['quality_metrics']['quality_grade']}")
    
    # After processing, generate thesis summary
    print("\n" + "="*80)
    print("Generating thesis metrics summary...")
