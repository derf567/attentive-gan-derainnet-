
import sys
import os

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append('.')

import os.path as ops
import argparse
import datetime
import yaml

import glob
import pandas as pd
from intelligent_switch import IntelligentImageProcessor
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Import handling
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from attentive_gan_model import derain_drop_net
    print(" Successfully imported derain_drop_net")
except ImportError:
    try:
        import attentive_gan_model.derain_drop_net as derain_module
        derain_drop_net = derain_module
        print(" Successfully imported derain_drop_net (alternate method)")
    except ImportError as e:
        print(f"[FAIL] Failed to import derain_drop_net: {e}")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "derain_drop_net",
            os.path.join(parent_dir, "attentive_gan_model", "derain_drop_net.py")
        )
        derain_drop_net = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(derain_drop_net)
        print(" Successfully loaded derain_drop_net (manual loading)")

try:
    from config import global_config
    print(" Successfully imported global_config")
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "global_config",
        os.path.join(parent_dir, "config", "global_config.py")
    )
    global_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(global_config)
    print(" Successfully loaded global_config")

CFG = global_config.cfg


class DerainConfigManager:
    """FIXED: Configuration manager with proper day/night loading"""
    
    def __init__(self, config_path='deraining_config.yaml'):
        self.config = {}
        self.config_path = config_path
        self.loaded_from_file = False
        
        # FIX: Try to load from file first
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                    self.loaded_from_file = True
                print(f" Loaded config from: {config_path}")
                
                # CRITICAL FIX: Validate that day/night settings exist
                if 'day_settings' not in self.config:
                    print(f" 'day_settings' missing in config!")
                else:
                    print(f" 'day_settings' found in config")
                    
                if 'night_settings' not in self.config:
                    print(f" 'night_settings' missing in config!")
                else:
                    print(f" 'night_settings' found in config")
                    
            except Exception as e:
                print(f" Failed to load config: {e}")
                print(f" Using default config as fallback")
                self.config = self._get_default_config()
        else:
            print(f" Config file not found: {config_path}")
            print(f"[INFO] Using built-in default config")
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        """
        FIXED: This should only be used if config file doesn't exist
        This provides basic fallback settings
        """
        return {
            'scene_detection': {
                'enabled': True,
                'method': 'brightness',
                'brightness_threshold': 80
            },
            'default_settings': {
                'preprocessing': {
                    'resize_method': 'INTER_LINEAR',
                    'denoise_before': False,
                    'contrast_adjust': 1.0
                },
                'postprocessing': {
                    'apply_clahe': True,
                    'clahe_clip_limit': 3.5,
                    'clahe_tile_size': 8,
                    'gamma_correction': 1.2,
                    'sharpen': True,
                    'sharpen_strength': 0.5
                },
                'inference': {
                    'target_width': 512,
                    'target_height': 512   
                },
            },
            'system': {
                'gpu_memory_fraction': 0.8,
                'verbose': True
            }
        }
    
    def detect_scene_type(self, image):
        """
        FIXED: Detect if image is day or night based on brightness
        Returns: 'day', 'night', or 'default'
        """
        scene_config = self.config.get('scene_detection', {})
        
        # Check if scene detection is enabled
        if not scene_config.get('enabled', True):
            print("[INFO] Scene detection DISABLED, using DEFAULT settings")
            return 'default'
        
        # Analyze image brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = float(np.mean(gray))
        threshold = scene_config.get('brightness_threshold', 80)
        
        # FIX: Determine scene type with clear logic
        if avg_brightness < threshold:
            scene_type = 'night'
            print(f"[DETECT] Brightness {avg_brightness:.1f} < {threshold} -> NIGHT scene")
        else:
            scene_type = 'day'
            print(f"[DETECT] Brightness {avg_brightness:.1f} >= {threshold} -> DAY scene")
        
        # CRITICAL FIX: Check if the detected scene config exists
        settings_key = f'{scene_type}_settings'
        if settings_key not in self.config:
            print(f" '{settings_key}' not found in config!")
            print(f" Available keys: {list(self.config.keys())}")
            print(f" Falling back to 'default_settings'")
            return 'default'
        
        return scene_type
    
    def get_settings(self, scene_type='default'):
        """
        FIXED: Get appropriate settings based on scene type
        Now properly returns day/night settings when they exist
        """
        settings_key = f'{scene_type}_settings'
        
        print(f"[CONFIG] Requesting settings for: {scene_type}")
        
        # FIX: Try to get scene-specific settings first
        if settings_key in self.config:
            print(f"[CONFIG] ✓ Found '{settings_key}' in config")
            settings = self.config[settings_key]
            
            # Validate settings structure
            required_keys = ['preprocessing', 'inference', 'postprocessing']
            missing_keys = [k for k in required_keys if k not in settings]
            
            if missing_keys:
                print(f" Settings incomplete, missing: {missing_keys}")
                print(f" Falling back to default_settings")
                return self.config.get('default_settings', {})
            
            print(f"[CONFIG] ✓ Loaded {scene_type} settings successfully")
            return settings
        
        # Fallback to default
        print(f"[CONFIG] ✗ '{settings_key}' not found")
        print(f"[CONFIG] Using 'default_settings' as fallback")
        return self.config.get('default_settings', {})


def get_resize_method(method_name):
    """Convert string to cv2 interpolation constant"""
    methods = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4,
        'INTER_AREA': cv2.INTER_AREA
    }
    return methods.get(method_name, cv2.INTER_LINEAR)




def minmax_scale(input_arr):
    """Normalize array to 0-255 range"""
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)
    return output_arr


def save_results_to_file(image_path, weights_path, label_path, output_image, atte_maps, 
                        ssim_val=None, psnr_val=None, output_file='derain_results.txt',
                        scene_type='default', settings=None, full_config=None):
    """
    FIXED: Save results to BOTH locations:
    1. metrics/deraining/{image_name}_metrics.txt (detailed)
    2. Root directory derain_results.txt (as specified by --output_file)
    """
    
    # === LOCATION 1: Detailed metrics in organized folder ===
    metrics_root = os.path.join(os.getcwd(), "metrics", "deraining")
    os.makedirs(metrics_root, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    detailed_metrics_file = os.path.join(metrics_root, f"{base_name}_metrics.txt")
    
    # === LOCATION 2: Main output file (as specified by command line) ===
    main_output_file = output_file  # This is from --output_file argument
    
    # Create the content
    content = []
    content.append("=" * 70)
    content.append("DERAINING MODEL TEST RESULTS")
    content.append("=" * 70)
    content.append(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append(f"Scene Type: {scene_type.upper()}")
    content.append(f"Input Image: {image_path}")
    content.append(f"Weights Path: {weights_path}")
    content.append(f"Label Path: {label_path if label_path else 'None'}")
    content.append("-" * 70)
    content.append("")
    
    # Configuration used
    if settings:
        content.append("CONFIGURATION USED")
        content.append("=" * 70)
        
        sections = ['preprocessing', 'inference', 'attention', 'postprocessing', 'quality']
        
        for section in sections:
            if section in settings and settings[section]:
                content.append(f"\n{section.upper()}:")
                for k, v in settings[section].items():
                    content.append(f"  {k}: {v}")
        
        if full_config:
            if 'system' in full_config:
                content.append(f"\nSYSTEM:")
                for k, v in full_config['system'].items():
                    content.append(f"  {k}: {v}")
    
    # Metrics
    if ssim_val is not None and psnr_val is not None:
        content.append("\nPERFORMANCE METRICS")
        content.append("=" * 70)
        content.append(f"SSIM: {ssim_val:.5f}")
        content.append(f"PSNR: {psnr_val:.2f} dB")
        content.append("=" * 70)
        content.append("")
    
    content.append("Results saved successfully!")
    
    full_content = "\n".join(content)
    
    # === WRITE TO BOTH LOCATIONS ===
    
    # 1. Write detailed metrics to organized folder
    try:
        with open(detailed_metrics_file, 'w', encoding='utf-8') as f:
            f.write(full_content)
        print(f"[SAVE] Detailed metrics → {detailed_metrics_file}")
    except Exception as e:
        print(f" Failed to save detailed metrics: {e}")
    
    # 2. Write to main output file (command line specified location)
    try:
        with open(main_output_file, 'w', encoding='utf-8') as f:
            f.write(full_content)
        print(f"[SAVE] Main output file → {main_output_file}")
    except Exception as e:
        print(f" Failed to save main output: {e}")
    
    # === ALSO CREATE A SIMPLE SUMMARY VERSION ===
    if ssim_val is not None and psnr_val is not None:
        summary_file = os.path.join(metrics_root, f"{base_name}_summary.txt")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("QUICK SUMMARY\n")
                f.write("="*50 + "\n")
                f.write(f"Image: {base_name}\n")
                f.write(f"Scene: {scene_type.upper()}\n")
                f.write(f"PSNR: {psnr_val:.2f} dB\n")
                f.write(f"SSIM: {ssim_val:.5f}\n")
            print(f"[SAVE] Quick summary → {summary_file}")
        except Exception as e:
            print(f" Failed to save summary: {e}")
            
def test_model(image_path, weights_path, label_path=None, output_file='derain_results.txt',
               config_path='deraining_config.yaml', force_scene=None):
    """
    COMPLETE: Deraining with PROPER day/night settings application
    """
    assert ops.exists(image_path), f"Image not found: {image_path}"
    
    # Load configuration
    config_manager = DerainConfigManager(config_path)
    verbose = config_manager.config.get('system', {}).get('verbose', True)
    
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if verbose:
        print(f"\n{'='*70}")
        print(f"DERAINING WITH DAY/NIGHT SETTINGS")
        print(f"{'='*70}")
        print(f"Input: {image_path}")
        print(f"Size: {image.shape[1]}x{image.shape[0]}")
    
    # ================================================================
    # SCENE DETECTION
    # ================================================================
    if force_scene:
        scene_type = force_scene
        if verbose:
            print(f"\n[FORCE] Using FORCED scene type: {scene_type.upper()}")
    else:
        scene_type = config_manager.detect_scene_type(image)
        if verbose:
            print(f"\n[AUTO] Detected scene type: {scene_type.upper()}")
    
    # ================================================================
    # LOAD SETTINGS
    # ================================================================
    settings = config_manager.get_settings(scene_type)
    
    if not settings:
        print(f" Failed to load settings for scene: {scene_type}")
        settings = config_manager._get_default_config()['default_settings']
    
    # ================================================================
    # STEP 1: PREPROCESSING (Apply night/day settings!)
    # ================================================================
    if verbose:
        print(f"\n[STEP 1] Applying {scene_type.upper()} preprocessing...")
    
    prep_config = settings.get('preprocessing', {})
    
    # Apply denoising if configured
    if prep_config.get('denoise_before', False):
        strength = prep_config.get('denoise_strength', 5)
        image = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        if verbose:
            print(f"  ✓ Applied denoising (strength: {strength})")
    
    # Apply contrast adjustment
    contrast = prep_config.get('contrast_adjust', 1.0)
    if contrast != 1.0:
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        if verbose:
            print(f"  ✓ Adjusted contrast (factor: {contrast})")
    
    # ================================================================
    # RESIZE FOR MODEL
    # ================================================================
    inf_config = settings.get('inference', {})
    target_w = inf_config.get('target_width', 512)
    target_h = inf_config.get('target_height', 512)
    resize_method_name = prep_config.get('resize_method', 'INTER_LINEAR')
    resize_method = get_resize_method(resize_method_name)
    
    if verbose:
        print(f"\n[INFERENCE CONFIG]")
        print(f"  Target size: {target_w}x{target_h}")
        print(f"  Resize method: {resize_method_name}")
        print(f"  Scene type: {scene_type}")
    
    image_resized = cv2.resize(image, (target_w, target_h), interpolation=resize_method)
    image_normalized = np.divide(np.array(image_resized, np.float32), 127.5) - 1.0
    
    # ================================================================
    # STEP 2: BUILD MODEL AND RUN INFERENCE
    # ================================================================
    if verbose:
        print(f"\n[STEP 2] Running GAN model inference...")
    
    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[1, target_h, target_w, 3],
        name='input_tensor'
    )
    
    phase = tf.constant('test', tf.string)
    
    try:
        net = derain_drop_net.DeRainNet(phase=phase)
    except AttributeError:
        try:
            net = derain_drop_net.derain_drop_net(phase=phase)
        except AttributeError:
            print(" Could not find DeRainNet class")
            sys.exit(1)
    
    output, attention_maps = net.inference(input_tensor=input_tensor, name='derain_net')
    
    # Session config
    gpu_mem = config_manager.config.get('system', {}).get('gpu_memory_fraction', 0.8)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.allocator_type = 'BFC'
    
    sess = tf.Session(config=sess_config)
    saver = tf.train.Saver()
    
    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        
        # RUN INFERENCE
        output_image, atte_maps = sess.run(
            [output, attention_maps],
            feed_dict={input_tensor: np.expand_dims(image_normalized, 0)}
        )
        
        output_image = output_image[0]
        
        # Normalize output
        for i in range(output_image.shape[2]):
            output_image[:, :, i] = minmax_scale(output_image[:, :, i])
        
        output_image = np.array(output_image, np.uint8)
        
        # ================================================================
        # STEP 3: POST-PROCESSING (Apply night/day settings!)
        # ================================================================
        if verbose:
            print(f"\n[STEP 3] Applying {scene_type.upper()} post-processing...")
        
        post_config = settings.get('postprocessing', {})
        result = output_image.copy()
        
        # CLAHE Enhancement
        if post_config.get('apply_clahe', False):
            clip_limit = post_config.get('clahe_clip_limit', 2.0)
            tile_size = post_config.get('clahe_tile_size', 8)
            
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            if verbose:
                print(f"  ✓ Applied CLAHE (clip={clip_limit}, tile={tile_size})")
        
        # Bilateral Filter
        if post_config.get('bilateral_filter', False):
            d = post_config.get('bilateral_d', 9)
            sigma_color = post_config.get('bilateral_sigma_color', 75)
            sigma_space = post_config.get('bilateral_sigma_space', 75)
            
            result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
            if verbose:
                print(f"  ✓ Applied bilateral filter (d={d})")
        
        # Sharpening
        if post_config.get('sharpen', False):
            strength = post_config.get('sharpen_strength', 0.5)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * strength
            kernel[1,1] = 8 + strength
            result = cv2.filter2D(result, -1, kernel / kernel.sum())
            if verbose:
                print(f"  ✓ Applied sharpening (strength={strength})")
        
        # Gamma Correction
        gamma = post_config.get('gamma_correction', 1.0)
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in np.arange(0, 256)]).astype("uint8")
            result = cv2.LUT(result, table)
            if verbose:
                print(f"  ✓ Applied gamma correction (gamma={gamma})")
        
        # Saturation Boost
        sat_boost = post_config.get('saturation_boost', 1.0)
        if sat_boost != 1.0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_boost, 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if verbose:
                print(f"  ✓ Applied saturation boost (factor={sat_boost})")
        
        output_image = result
        
        # ================================================================
        # STEP 4: SAVE AND EVALUATE
        # ================================================================
        if verbose:
            print(f"\n[STEP 4] Saving output...")
        
        # Save output
        cv2.imwrite('derain_ret.png', output_image)
        
        # Calculate metrics
        print("\n[STEP 5] Evaluating deraining quality metrics...")
        ssim_val = None
        psnr_val = None
        
        try:
            result_metrics = IntelligentImageProcessor.compute_derain_metrics(
                image_path,
                "derain_ret.png"
            )
            if result_metrics:
                psnr_val = result_metrics["psnr"]
                ssim_val = result_metrics["ssim"]
                print(f"[METRICS] PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
        except Exception as e:
            import traceback
            print(f" Failed to compute metrics: {e}")
            print(traceback.format_exc())
        
        # Save results
        save_results_to_file(
            image_path, weights_path, label_path, output_image, atte_maps,
            ssim_val, psnr_val, output_file, 
            scene_type,
            settings,
            full_config=config_manager.config
        )
        
        # Save additional outputs
        cv2.imwrite('src_img.png', image_resized)
        cv2.imwrite('comparison.png', np.hstack([image_resized, output_image]))
        
        if verbose:
            print(f"\n[DONE] Processing complete!")
            print(f"  Scene: {scene_type.upper()}")
            print(f"  Applied preprocessing: {list(prep_config.keys())}")
            print(f"  Applied postprocessing: {list(post_config.keys())}")
            print(f"  Output: derain_ret.png")
            print(f"{'='*70}\n")
    
    return output_image, ssim_val, psnr_val

def summarize_deraining_metrics():
    """Collect all *_metrics.txt files from metrics/deraining and save summary CSV"""
    metrics_dir = os.path.join("metrics", "deraining")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_files = glob.glob(os.path.join(metrics_dir, "*_metrics.txt"))
    rows = []

    for file in metrics_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()

        psnr = ssim = None
        for line in content.splitlines():
            if "PSNR" in line:
                psnr = float(line.split(":")[1].strip().replace("dB", "").strip())
            elif "SSIM" in line:
                ssim = float(line.split(":")[1].strip())

        rows.append({
            "Filename": os.path.basename(file).replace("_metrics.txt", ""),
            "PSNR": psnr,
            "SSIM": ssim
        })

    if rows:
        df = pd.DataFrame(rows)
        summary_path = os.path.join(metrics_dir, "deraining_summary.csv")
        df.to_csv(summary_path, index=False)
        print("\n DERAINING METRICS SUMMARY")
        print(df)
        print(f"\nSummary saved to: {summary_path}")
    else:
        print("\n⚠ No metrics found in metrics/deraining directory.")


def init_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Brightness-based Deraining Model')
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--label_path', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='derain_results.txt')
    parser.add_argument('--config', type=str, default='deraining_config.yaml')
    parser.add_argument('--force_scene', type=str, choices=['day', 'night', 'default'])
    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()

    # 1️ Run the deraining model first
    test_model(
        args.image_path,
        args.weights_path,
        args.label_path,
        args.output_file,
        args.config,
        args.force_scene
    )

    # 2️ Then summarize all metrics files
    summarize_deraining_metrics()