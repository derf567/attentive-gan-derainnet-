import sys
import os
import cv2
import numpy as np
import json
from enum import Enum
import subprocess
import importlib.util

class ProcessingMode(Enum):
    DERAIN = 1
    DEGLARE = 2
    ENHANCE = 3
    NONE = 4

class IntelligentImageProcessor:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.rain_threshold = 0.15
        self.glare_threshold = 0.25
        self.min_contrast = 40

    def detect_rain(self, image):
        """Detect rain patterns in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Additional texture analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        texture_score = np.mean(grad_mag) / 255.0
        
        # Combined score with weights
        rain_score = (edge_density * 0.7 + texture_score * 0.3)
        return min(rain_score * 1.2, 1.0)

    def detect_glare(self, image):
        """Detect glare/reflections in the image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        s = hsv[:,:,1]
        glare_mask = (v > 220) & (s < 30)
        return np.sum(glare_mask) / glare_mask.size

    def needs_enhancement(self, image):
        """Check if image needs general enhancement"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check for low contrast
        if np.std(gray) < self.min_contrast:
            return True
        
        # Check for poor exposure
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = hist / hist.sum()
        if hist[0] > 0.3 or hist[-1] > 0.3:
            return True
            
        return False

    def load_dashcam_enhancer(self):
        """Dynamically load the DashcamVideoEnhancer class"""
        try:
            dashcam_path = os.path.join(self.script_dir, "dashcam_enhancer.py")
            if not os.path.exists(dashcam_path):
                raise FileNotFoundError(f"dashcam_enhancer.py not found at {dashcam_path}")
            
            spec = importlib.util.spec_from_file_location(
                "dashcam_enhancer",
                dashcam_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.DashcamVideoEnhancer
        except Exception as e:
            print(f"Error loading dashcam_enhancer: {str(e)}")
            return None

    def run_deraining(self, image_path, output_path):
        """Execute Attentive GAN deraining model"""
        try:
            weights_path = os.path.join(self.project_root, "weights", "derain_gan", "derain_gan.ckpt-100000")
            cmd = [
                sys.executable,
                os.path.join(self.project_root, "tools", "test_model.py"),
                "--image_path", os.path.abspath(image_path),
                "--weights_path", os.path.abspath(weights_path),
                "--output_file", os.path.join(self.project_root, "derain_results.txt")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                output_file = os.path.join(self.project_root, "derain_ret.png")
                if os.path.exists(output_file):
                    os.replace(output_file, os.path.abspath(output_path))
                    return {'success': True}
            return {'success': False, 'error': result.stderr}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_deglaring(self, image_path, output_path):
        """Execute Retinex-based deglaring with proper path handling"""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            DashcamVideoEnhancer = self.load_dashcam_enhancer()
            if not DashcamVideoEnhancer:
                return {'success': False, 'error': 'Could not load DashcamVideoEnhancer'}
            
            enhancer = DashcamVideoEnhancer()
            if not enhancer.setup_model():
                return {'success': False, 'error': 'Failed to setup RetinexNet model'}
            
            # Process the image with temporary output directory
            temp_dir = os.path.join(self.script_dir, "temp_enhanced")
            os.makedirs(temp_dir, exist_ok=True)
            
            results = enhancer.process_image(image_path, temp_dir)
            
            if results and os.path.exists(results['enhanced_image']):
                # Move the enhanced image to the final output path
                os.replace(results['enhanced_image'], output_path)
                
                # Clean up temporary files
                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))
                os.rmdir(temp_dir)
                
                return {
                    'success': True,
                    'metrics': results.get('metrics', None)
                }
            
            return {'success': False, 'error': 'Enhancement returned no results'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def process_image(self, image_path, output_path):
        """Main processing function with enhanced error handling"""
        try:
            # Verify input image exists
            if not os.path.exists(image_path):
                return {'final_status': 'error', 'error': f'Input image not found: {image_path}'}
            
            img = cv2.imread(image_path)
            if img is None:
                return {'final_status': 'error', 'error': 'Could not read image file'}

            # Analysis
            rain_score = self.detect_rain(img)
            glare_score = self.detect_glare(img)
            needs_enhance = self.needs_enhancement(img)
            
            print(f"\nImage Analysis:")
            print(f"Rain Score: {rain_score:.2f} (Threshold: {self.rain_threshold})")
            print(f"Glare Score: {glare_score:.2f} (Threshold: {self.glare_threshold})")
            print(f"Needs Enhancement: {needs_enhance}")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Processing logic
            if rain_score > self.rain_threshold:
                print("\nExecuting rain removal...")
                result = self.run_deraining(image_path, output_path)
                mode = 'DERAIN'
            elif glare_score > self.glare_threshold:
                print("\nExecuting glare reduction...")
                result = self.run_deglaring(image_path, output_path)
                mode = 'DEGLARE'
            elif needs_enhance:
                print("\nExecuting general enhancement...")
                result = self.run_deglaring(image_path, output_path)
                mode = 'ENHANCE'
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
            print(f"\nError during processing: {str(e)}")
            return {'final_status': 'error', 'error': str(e)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--report', help='Optional report file path')
    args = parser.parse_args()

    processor = IntelligentImageProcessor()
    result = processor.process_image(args.image, args.output)

    print("\n=== PROCESSING SUMMARY ===")
    print(f"Input Image: {args.image}")
    print(f"Output Image: {args.output}")
    print(f"Final Status: {result['final_status']}")

    if result['final_status'] == 'error':
        print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print(f"Processing Mode: {result.get('processing_mode', 'NONE')}")
        if 'metrics' in result:
            print("\nMetrics:")
            print(f"Rain Score: {result['metrics'].get('rain_score', 0):.2f}")
            print(f"Glare Score: {result['metrics'].get('glare_score', 0):.2f}")
            print(f"Needed Enhancement: {result['metrics'].get('needed_enhancement', False)}")
            if 'brightness' in result['metrics']:
                print(f"\nBrightness Improvement: {result['metrics']['brightness']['improvement']:.1f}")
                print(f"Contrast Improvement: {result['metrics']['contrast']['improvement']:.1f}")
                if 'ssim' in result['metrics']:
                    print(f"SSIM: {result['metrics']['ssim']:.3f}")

    if args.report:
        try:
            with open(args.report, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nReport saved to: {args.report}")
        except Exception as e:
            print(f"\nFailed to save report: {str(e)}")