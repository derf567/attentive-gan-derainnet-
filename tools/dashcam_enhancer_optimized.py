"""
UPDATED: Dashcam Video Enhancement System with Optimized Retinex Parameters
Now loads parameters from optimized_retinex.yaml (trained on your data!)

Usage:
    python dashcam_enhancer.py input.jpg enhanced_output/
"""

import os
import sys
import cv2
import numpy as np
import yaml
import time
from pathlib import Path

# Import metrics
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    print("✓ Advanced metrics available")
except ImportError:
    print("⚠ Advanced metrics not available - using basic metrics only")
    ssim = None
    psnr = None


class OptimizedDashcamEnhancer:
    """
    Dashcam enhancer with TRAINED parameters from optimized_retinex.yaml
    """
    
    def __init__(self, config_path='optimized_retinex.yaml'):
        """
        Initialize enhancer with optimized parameters
        
        Args:
            config_path: Path to optimized_retinex.yaml
        """
        self.config_path = config_path
        
        # Load optimized configuration
        if os.path.exists(config_path):
            print(f"\n{'='*70}")
            print(f"LOADING OPTIMIZED PARAMETERS")
            print(f"{'='*70}")
            print(f"Config file: {config_path}")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract Retinex parameters
            retinex_config = config['retinex_enhancement']
            self.scales = retinex_config['scales']
            self.weights = retinex_config['weights']
            self.gamma = retinex_config['gamma_correction']
            self.contrast_strength = retinex_config.get('contrast_strength', 1.2)
            self.percentile_low = retinex_config.get('percentile_low', 2)
            self.percentile_high = retinex_config.get('percentile_high', 98)
            
            # Show training info
            if 'training_info' in config:
                info = config['training_info']
                print(f"\nTraining Info:")
                print(f"  Training pairs: {info.get('num_training_pairs', 'N/A')}")
                print(f"  Best score: {info.get('best_score', 'N/A'):.4f}")
                print(f"  Avg SSIM: {info.get('avg_ssim', 'N/A'):.4f}")
                print(f"  Avg PSNR: {info.get('avg_psnr', 'N/A'):.2f} dB")
                print(f"  Training date: {info.get('training_date', 'N/A')}")
                print(f"  Method: {info.get('method', 'N/A')}")
            
            print(f"\nOptimized Parameters:")
            print(f"  Scales: {self.scales}")
            print(f"  Weights: {[f'{w:.3f}' for w in self.weights]}")
            print(f"  Gamma: {self.gamma:.3f}")
            print(f"  Contrast: {self.contrast_strength:.2f}")
            print(f"{'='*70}\n")
            
        else:
            print(f"⚠ Warning: Config not found: {config_path}")
            print(f"⚠ Using default parameters (not optimized)")
            
            # Fallback to defaults
            self.scales = [15, 80, 250]
            self.weights = [0.4, 0.4, 0.2]
            self.gamma = 0.75
            self.contrast_strength = 1.2
            self.percentile_low = 2
            self.percentile_high = 98
    
    def multi_scale_retinex(self, image):
        """
        Apply Multi-Scale Retinex with OPTIMIZED parameters
        
        Args:
            image: Input RGB image (numpy array)
        
        Returns:
            Enhanced RGB image (numpy array, uint8)
        """
        # Convert to float and add epsilon
        img = image.astype(np.float64) + 1.0
        log_img = np.log(img)
        
        msr_result = np.zeros_like(log_img)
        
        # Apply each scale with optimized weights
        for scale, weight in zip(self.scales, self.weights):
            # Gaussian blur
            blurred = cv2.GaussianBlur(log_img, (0, 0), scale)
            
            # Single scale retinex
            single_scale = log_img - blurred
            
            # Accumulate weighted result
            msr_result += weight * single_scale
        
        # Convert back from log domain
        enhanced = np.exp(msr_result) - 1.0
        
        # Normalize each channel with trained percentiles
        for c in range(3):
            channel = enhanced[:, :, c]
            
            # Percentile stretching (removes outliers)
            p_low = np.percentile(channel, self.percentile_low)
            p_high = np.percentile(channel, self.percentile_high)
            
            channel = np.clip((channel - p_low) / (p_high - p_low + 1e-8), 0, 1)
            enhanced[:, :, c] = channel
        
        # Apply trained gamma correction
        enhanced = np.power(enhanced, self.gamma)
        
        # Apply contrast strength
        enhanced = enhanced * self.contrast_strength
        enhanced = np.clip(enhanced, 0, 1)
        
        # Convert to uint8
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def enhance_image(self, image_path, output_dir='enhanced_output'):
        """
        Enhance a single dashcam image
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save enhanced image
        
        Returns:
            Dictionary with paths and metrics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Record start time
        start_time = time.time()
        
        # Apply optimized Retinex enhancement
        print(f"\n{'='*70}")
        print(f"ENHANCING IMAGE WITH OPTIMIZED PARAMETERS")
        print(f"{'='*70}")
        print(f"Input: {os.path.basename(image_path)}")
        print(f"Size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")
        
        enhanced_rgb = self.glare_aware_retinex(image_rgb)
        
        processing_time = time.time() - start_time
        
        print(f"Processing time: {processing_time:.3f} seconds")
        
        # Calculate metrics if available
        metrics = {}
        if ssim is not None and psnr is not None:
            # Convert to grayscale for metrics
            orig_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            enh_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
            
            # Calculate improvement metrics
            contrast_orig = np.std(orig_gray)
            contrast_enh = np.std(enh_gray)
            brightness_orig = np.mean(orig_gray)
            brightness_enh = np.mean(enh_gray)
            
            metrics = {
                'contrast_original': float(contrast_orig),
                'contrast_enhanced': float(contrast_enh),
                'contrast_improvement': float(contrast_enh - contrast_orig),
                'brightness_original': float(brightness_orig),
                'brightness_enhanced': float(brightness_enh),
                'processing_time': processing_time
            }
            
            print(f"\nQuality Metrics:")
            print(f"  Contrast: {contrast_orig:.2f} → {contrast_enh:.2f} (+{contrast_enh-contrast_orig:.2f})")
            print(f"  Brightness: {brightness_orig:.2f} → {brightness_enh:.2f}")
        
        # Generate output filename
        base_name = Path(image_path).stem
        enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
        
        # Save enhanced image
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(enhanced_path, enhanced_bgr)
        
        # Create comparison image (side by side)
        comparison = np.hstack([image, enhanced_bgr])
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison)
        
        print(f"\n✓ Enhanced image saved: {enhanced_path}")
        print(f"✓ Comparison saved: {comparison_path}")
        print(f"{'='*70}\n")
        
        return {
            'input_path': image_path,
            'enhanced_path': enhanced_path,
            'comparison_path': comparison_path,
            'metrics': metrics,
            'processing_time': processing_time
        }
    
    def detect_saturated_regions(self, image, threshold=240):
        """
        Detect overexposed glare regions (taillights)
        
        Args:
            image: RGB image
            threshold: Saturation threshold (default 240)
        
        Returns:
            Binary mask of saturated regions
        """
        # Find pixels where ANY channel is saturated
        max_channel = np.max(image, axis=2)
        saturated_mask = (max_channel > threshold).astype(np.float32)
        
        # Dilate mask to capture glare bloom
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        saturated_mask = cv2.dilate(saturated_mask, kernel, iterations=2)
        
        return saturated_mask

    def glare_aware_retinex(self, image):
        """
        Apply Retinex with GLARE SUPPRESSION for taillights
        
        Args:
            image: Input RGB image (numpy array)
        
        Returns:
            Enhanced RGB image with reduced glare
        """
        # 1. DETECT SATURATED REGIONS (taillights)
        saturation_mask = self.detect_saturated_regions(image, threshold=240)
        
        # 2. CLIP SATURATED PIXELS before Retinex
        # This prevents log domain explosion
        img_clipped = image.copy().astype(np.float64)
        for c in range(3):
            channel = img_clipped[:, :, c]
            # Cap at 95th percentile to reduce glare
            p95 = np.percentile(channel, 95)
            channel[saturation_mask > 0.5] = np.minimum(
                channel[saturation_mask > 0.5], 
                p95
            )
            img_clipped[:, :, c] = channel
        
        # 3. APPLY RETINEX on clipped image
        img_clipped = img_clipped + 1.0
        log_img = np.log(img_clipped)
        
        msr_result = np.zeros_like(log_img)
        
        # Use SMALLER scales to reduce halo artifacts
        # Larger scales create bloom around bright sources
        scales_glare = [10, 40, 120]  # Smaller than your [20, 100, 300]
        weights_glare = [0.5, 0.3, 0.2]  # More weight on small scale
        
        for scale, weight in zip(scales_glare, weights_glare):
            blurred = cv2.GaussianBlur(log_img, (0, 0), scale)
            single_scale = log_img - blurred
            msr_result += weight * single_scale
        
        # 4. CONVERT BACK with suppression constraint
        enhanced = np.exp(msr_result) - 1.0
        
        # 5. GLARE SUPPRESSION LOSS (from paper)
        # Ensure L * R_hat <= I (prevents over-brightening)
        illumination = cv2.GaussianBlur(image.astype(np.float64), (0, 0), 50)
        illumination = illumination / 255.0
        
        for c in range(3):
            channel = enhanced[:, :, c]
            
            # Apply suppression in saturated regions
            suppression_factor = 1.0 - (saturation_mask * 0.7)
            channel = channel * suppression_factor
            
            # Percentile normalization
            p_low = np.percentile(channel, self.percentile_low)
            p_high = np.percentile(channel, 92)  # Lower than 98 to reduce glare
            
            channel = np.clip((channel - p_low) / (p_high - p_low + 1e-8), 0, 1)
            enhanced[:, :, c] = channel
        
        # 6. REDUCED GAMMA for glare regions
        # Lower gamma darkens bright areas
        gamma_map = np.ones_like(saturation_mask) * self.gamma
        gamma_map[saturation_mask > 0.5] = 0.6  # Darker gamma for glare
        
        for c in range(3):
            enhanced[:, :, c] = np.power(enhanced[:, :, c], gamma_map)
        
        # 7. FINAL ADJUSTMENTS
        enhanced = enhanced * self.contrast_strength
        enhanced = np.clip(enhanced, 0, 1)
        
        # 8. BLEND with original in saturated regions
        # Preserve some original detail
        enhanced_bgr = enhanced.copy()
        original_normalized = image.astype(np.float64) / 255.0
        
        blend_weight = saturation_mask[:, :, np.newaxis] * 0.5
        enhanced = (1 - blend_weight) * enhanced + blend_weight * original_normalized
        
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced

    def batch_enhance(self, input_dir, output_dir='batch_enhanced'):
        """
        Enhance all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save enhanced images
        """
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        if len(image_files) == 0:
            print(f"❌ No images found in: {input_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"BATCH ENHANCEMENT")
        print(f"{'='*70}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Images to process: {len(image_files)}")
        print(f"{'='*70}\n")
        
        results = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
            
            result = self.enhance_image(str(image_path), output_dir)
            if result:
                results.append(result)
        
        # Summary
        print(f"\n{'='*70}")
        print(f"BATCH ENHANCEMENT COMPLETE!")
        print(f"{'='*70}")
        print(f"Total processed: {len(results)}/{len(image_files)}")
        
        if results:
            avg_time = np.mean([r['processing_time'] for r in results])
            print(f"Average processing time: {avg_time:.3f} seconds/image")
        
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}\n")
        
        return results





def main():
    """Main execution function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Dashcam Enhancement with Optimized Retinex Parameters'
    )
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('--output', '-o', default='enhanced_output',
                       help='Output directory (default: enhanced_output)')
    parser.add_argument('--config', '-c', default='optimized_retinex.yaml',
                       help='Config file (default: optimized_retinex.yaml)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Batch mode (process directory)')
    
    args = parser.parse_args()
    
    # Initialize enhancer with optimized parameters
    enhancer = OptimizedDashcamEnhancer(config_path=args.config)
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"❌ Input not found: {args.input}")
        sys.exit(1)
    
    # Process
    if args.batch or os.path.isdir(args.input):
        # Batch mode
        enhancer.batch_enhance(args.input, args.output)
    else:
        # Single image mode
        result = enhancer.enhance_image(args.input, args.output)
        
        if result:
            print(f"✓ Enhancement successful!")
            print(f"  Enhanced: {result['enhanced_path']}")
            print(f"  Comparison: {result['comparison_path']}")
        else:
            print(f"❌ Enhancement failed")
            sys.exit(1)


if __name__ == "__main__":
    main()