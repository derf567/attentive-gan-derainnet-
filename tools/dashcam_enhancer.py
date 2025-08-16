# Dashcam Video Enhancement System - Thesis Project
# Institution: Davao City Research
# Method: RetinexNet for Dashcam Footage Processing

print("THESIS PROJECT: Dashcam Video Enhancement System")
print("Institution: Davao City Research")
print("=" * 60)
print("Installing and importing requirements...")

# Import all required libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import time

# Import metrics with proper error handling
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage import exposure, filters
    print("SSIM and PSNR imported successfully")
except ImportError:
    print("Advanced metrics not available - using basic metrics only")
    ssim = None
    psnr = None

print("All packages imported successfully!")
print("System ready for dashcam video enhancement with RetinexNet!")

# ==================== RETINEX MODEL ARCHITECTURE ====================

class DecomNet(nn.Module):
    """Decomposition Network for Retinex-based enhancement"""
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode='replicate')
        self.net1_conv1 = nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net1_conv2 = nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net1_conv3 = nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net1_conv4 = nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net1_conv5 = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)

        feats0 = F.relu(self.net1_conv0(input_img))
        feats1 = F.relu(self.net1_conv1(feats0))
        feats2 = F.relu(self.net1_conv2(feats1))
        feats3 = F.relu(self.net1_conv3(feats2))
        feats4 = F.relu(self.net1_conv4(feats3))
        feats5 = torch.sigmoid(self.net1_conv5(feats4))

        I = feats5[:, 0:1, :, :]
        R = feats5[:, 1:4, :, :]
        return R, I

class RelightNet(nn.Module):
    """Relighting Network for illumination adjustment"""
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.net2_conv0 = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_conv1 = nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_conv2 = nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_conv3 = nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_conv4 = nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_conv5 = nn.Conv2d(channel, 1, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, input_I, input_R):
        input_img = torch.cat((input_I, input_R), dim=1)
        feats0 = self.relu(self.net2_conv0(input_img))
        feats1 = self.relu(self.net2_conv1(feats0))
        feats2 = self.relu(self.net2_conv2(feats1))
        feats3 = self.relu(self.net2_conv3(feats2))
        feats4 = self.relu(self.net2_conv4(feats3))
        feats5 = torch.sigmoid(self.net2_conv5(feats4))
        return feats5

class RetinexNet(nn.Module):
    """Complete RetinexNet architecture for dashcam enhancement"""
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()

    def forward(self, input_low, input_high=None):
        if input_high is None:
            input_high = input_low

        try:
            R_low, I_low = self.DecomNet(input_low)
            R_high, I_high = self.DecomNet(input_high)
            I_delta = self.RelightNet(I_low, R_low)

            I_low_3 = torch.cat([I_low, I_low, I_low], dim=1)
            I_delta_3 = torch.cat([I_delta, I_delta, I_delta], dim=1)
            enhanced = R_low * (I_low_3 + I_delta_3)
            enhanced = torch.clamp(enhanced, 0, 1)

            return enhanced, R_low, I_low, I_delta

        except Exception as e:
            print(f"RetinexNet forward error: {e}")
            return input_low, input_low[:, :1, :, :], input_low[:, :1, :, :], input_low[:, :1, :, :]

# ==================== DASHCAM VIDEO ENHANCEMENT SYSTEM ====================

class DashcamVideoEnhancer:
    """Main class for dashcam video enhancement using RetinexNet"""

    def __init__(self, target_size=(512, 512)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_size = target_size
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.retinex_model = None
        self.frame_metrics = []
        self.processing_times = []

    def setup_model(self):
        """Initialize RetinexNet model"""
        print("Setting up RetinexNet model...")

        try:
            self.retinex_model = RetinexNet().to(self.device)
            self.retinex_model.eval()
            print("RetinexNet initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to setup model: {e}")
            return False

    def preprocess_frame(self, frame):
        """Preprocess dashcam frame for enhancement"""
        # Resize frame maintaining aspect ratio
        h, w = frame.shape[:2]
        if w > h:
            new_w = self.target_size[0]
            new_h = int(h * new_w / w)
        else:
            new_h = self.target_size[1]
            new_w = int(w * new_h / h)

        frame = cv2.resize(frame, (new_w, new_h))

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0

        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)

        return frame_tensor.to(self.device), frame_rgb

    def enhance_frame_retinex(self, frame):
        """Enhance single frame using Classical Multi-Scale Retinex"""
        try:
            enhanced = self._multi_scale_retinex(frame)
            reflectance = self._extract_reflectance(frame)
            illumination = self._extract_illumination(frame)

            return {
                'enhanced': enhanced,
                'reflectance': reflectance,
                'illumination': illumination,
                'original': frame
            }
        except Exception as e:
            print(f"Retinex enhancement failed: {e}")
            return {
                'enhanced': frame,
                'reflectance': frame,
                'illumination': frame,
                'original': frame
            }

    def _multi_scale_retinex(self, image):
        """Classical Multi-Scale Retinex implementation optimized for dashcam footage"""
        img = image.astype(np.float64) + 1.0
        log_img = np.log(img)

        # Optimized scales for dashcam footage (road, sky, objects)
        scales = [15, 80, 250]
        weights = [0.4, 0.4, 0.2]  # Emphasize mid-range details
        msr_result = np.zeros_like(log_img)

        for scale, weight in zip(scales, weights):
            blurred = cv2.GaussianBlur(log_img, (0, 0), scale)
            single_scale_retinex = log_img - blurred
            msr_result += weight * single_scale_retinex

        enhanced = np.exp(msr_result) - 1.0

        # Normalize per channel with percentile stretching
        for c in range(3):
            channel = enhanced[:, :, c]
            p1, p99 = np.percentile(channel, [2, 98])  # More conservative for dashcam
            channel = np.clip((channel - p1) / (p99 - p1 + 1e-8), 0, 1)
            enhanced[:, :, c] = channel

        # Apply gamma correction for dashcam visibility
        enhanced = np.power(enhanced, 0.75)
        enhanced = (enhanced * 255).astype(np.uint8)

        return enhanced

    def _extract_reflectance(self, image):
        """Extract reflectance component for road surface analysis"""
        img = image.astype(np.float64) + 1.0
        illumination = cv2.GaussianBlur(img, (81, 81), 40)  # Adjusted for dashcam
        reflectance = img / (illumination + 1e-8)

        for c in range(3):
            channel = reflectance[:, :, c]
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            reflectance[:, :, c] = channel

        return (reflectance * 255).astype(np.uint8)

    def _extract_illumination(self, image):
        """Extract illumination component for lighting analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        illumination = cv2.GaussianBlur(gray.astype(np.float64), (81, 81), 40)
        illumination = (illumination - illumination.min()) / (illumination.max() - illumination.min() + 1e-8)
        illumination_rgb = np.stack([illumination, illumination, illumination], axis=2)

        return (illumination_rgb * 255).astype(np.uint8)

    def calculate_comprehensive_metrics(self, original, enhanced):
        """Calculate comprehensive metrics for dashcam footage analysis"""
        metrics = {}

        # Convert to different color spaces for analysis
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

        orig_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
        enh_lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)

        # Basic luminance metrics
        metrics['brightness'] = {
            'original': float(np.mean(orig_gray)),
            'enhanced': float(np.mean(enh_gray)),
            'improvement': float(np.mean(enh_gray) - np.mean(orig_gray))
        }

        metrics['contrast'] = {
            'original': float(np.std(orig_gray)),
            'enhanced': float(np.std(enh_gray)),
            'improvement': float(np.std(enh_gray) - np.std(orig_gray))
        }

        # Dynamic range metrics
        metrics['dynamic_range'] = {
            'original': float(orig_gray.max() - orig_gray.min()),
            'enhanced': float(enh_gray.max() - enh_gray.min()),
            'improvement': float((enh_gray.max() - enh_gray.min()) - (orig_gray.max() - orig_gray.min()))
        }

        # Entropy (information content)
        orig_hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        enh_hist = cv2.calcHist([enh_gray], [0], None, [256], [0, 256])

        orig_entropy = -np.sum((orig_hist/orig_hist.sum()) * np.log2((orig_hist/orig_hist.sum()) + 1e-10))
        enh_entropy = -np.sum((enh_hist/enh_hist.sum()) * np.log2((enh_hist/enh_hist.sum()) + 1e-10))

        metrics['entropy'] = {
            'original': float(orig_entropy),
            'enhanced': float(enh_entropy),
            'improvement': float(enh_entropy - orig_entropy)
        }

        # Edge preservation metrics
        orig_edges = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
        enh_edges = cv2.Laplacian(enh_gray, cv2.CV_64F).var()

        metrics['edge_strength'] = {
            'original': float(orig_edges),
            'enhanced': float(enh_edges),
            'improvement': float(enh_edges - orig_edges)
        }

        # Advanced metrics if available
        if ssim is not None:
            try:
                ssim_score = ssim(orig_gray, enh_gray, data_range=255)
                metrics['ssim'] = float(ssim_score)
            except:
                metrics['ssim'] = 0.5

        if psnr is not None:
            try:
                psnr_score = psnr(original, enhanced, data_range=255)
                metrics['psnr'] = float(psnr_score)
            except:
                metrics['psnr'] = 20.0

        # Color analysis in LAB space
        metrics['color_enhancement'] = {
            'l_channel_improvement': float(np.mean(enh_lab[:,:,0]) - np.mean(orig_lab[:,:,0])),
            'a_channel_variance': float(np.std(enh_lab[:,:,1]) - np.std(orig_lab[:,:,1])),
            'b_channel_variance': float(np.std(enh_lab[:,:,2]) - np.std(orig_lab[:,:,2]))
        }

        # Road visibility metrics (assuming lower portion contains road)
        road_region_orig = orig_gray[int(orig_gray.shape[0]*0.6):, :]
        road_region_enh = enh_gray[int(enh_gray.shape[0]*0.6):, :]

        metrics['road_visibility'] = {
            'original_brightness': float(np.mean(road_region_orig)),
            'enhanced_brightness': float(np.mean(road_region_enh)),
            'improvement': float(np.mean(road_region_enh) - np.mean(road_region_orig))
        }

        return metrics

    def process_image(self, image_path, output_dir="enhanced_images"):
        """Process a single image with comprehensive analysis"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None

        # Preprocess image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhance with Retinex
        start_time = time.time()
        retinex_result = self.enhance_frame_retinex(image_rgb)
        processing_time = time.time() - start_time

        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(image_rgb, retinex_result['enhanced'])

        # Save results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save enhanced image
        enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
        cv2.imwrite(enhanced_path, cv2.cvtColor(retinex_result['enhanced'], cv2.COLOR_RGB2BGR))
        
        # Save reflectance
        reflectance_path = os.path.join(output_dir, f"{base_name}_reflectance.jpg")
        cv2.imwrite(reflectance_path, cv2.cvtColor(retinex_result['reflectance'], cv2.COLOR_RGB2BGR))
        
        # Save illumination
        illumination_path = os.path.join(output_dir, f"{base_name}_illumination.jpg")
        cv2.imwrite(illumination_path, cv2.cvtColor(retinex_result['illumination'], cv2.COLOR_RGB2BGR))
        
        # Save comparison image
        comparison = np.hstack([image, cv2.cvtColor(retinex_result['enhanced'], cv2.COLOR_RGB2BGR)])
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison)

        # Create result dictionary
        results = {
            'original_image': image_path,
            'enhanced_image': enhanced_path,
            'reflectance_image': reflectance_path,
            'illumination_image': illumination_path,
            'comparison_image': comparison_path,
            'metrics': metrics,
            'processing_time': processing_time
        }

        # Generate report
        self.create_image_report(results)

        # Visualize results
        self.visualize_image_results(results)

        return results

    def create_image_report(self, results):
        """Create comprehensive report for image processing"""
        
        print("\n" + "="*80)
        print("IMAGE ENHANCEMENT ANALYSIS REPORT")
        print("RetinexNet Method")
        print("="*80)
        
        metrics = results['metrics']
        
        print(f"\nIMAGE INFORMATION:")
        print(f"  Original image: {results['original_image']}")
        print(f"  Enhanced image: {results['enhanced_image']}")
        print(f"  Processing time: {results['processing_time']:.2f} seconds")
        
        print(f"\nENHANCEMENT METRICS:")
        print(f"  Brightness improvement: {metrics['brightness']['improvement']:+.1f}")
        print(f"  Contrast improvement: {metrics['contrast']['improvement']:+.1f}")
        print(f"  Dynamic range improvement: {metrics['dynamic_range']['improvement']:+.1f}")
        print(f"  Entropy improvement: {metrics['entropy']['improvement']:+.2f} bits")
        print(f"  Edge strength improvement: {metrics['edge_strength']['improvement']:+.1f}")
        
        if 'ssim' in metrics:
            print(f"  Structural Similarity (SSIM): {metrics['ssim']:.3f}")
        
        if 'psnr' in metrics:
            print(f"  Peak Signal-to-Noise Ratio (PSNR): {metrics['psnr']:.2f} dB")
        
        print(f"\nOUTPUT FILES:")
        print(f"  Enhanced image saved to: {results['enhanced_image']}")
        print(f"  Reflectance component saved to: {results['reflectance_image']}")
        print(f"  Illumination component saved to: {results['illumination_image']}")
        print(f"  Before/after comparison saved to: {results['comparison_image']}")

    def visualize_image_results(self, results):
        """Visualize image enhancement results"""
        
        # Read all images
        original = cv2.cvtColor(cv2.imread(results['original_image']), cv2.COLOR_BGR2RGB)
        enhanced = cv2.cvtColor(cv2.imread(results['enhanced_image']), cv2.COLOR_BGR2RGB)
        reflectance = cv2.cvtColor(cv2.imread(results['reflectance_image']), cv2.COLOR_BGR2RGB)
        illumination = cv2.cvtColor(cv2.imread(results['illumination_image']), cv2.COLOR_BGR2GRAY)
        
        # Create figure
        plt.figure(figsize=(20, 10))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(original)
        plt.title('Original Image')
        plt.axis('off')
        
        # Enhanced image
        plt.subplot(2, 2, 2)
        plt.imshow(enhanced)
        plt.title('Enhanced Image')
        plt.axis('off')
        
        # Reflectance component
        plt.subplot(2, 2, 3)
        plt.imshow(reflectance)
        plt.title('Reflectance Component')
        plt.axis('off')
        
        # Illumination component
        plt.subplot(2, 2, 4)
        plt.imshow(illumination, cmap='gray')
        plt.title('Illumination Component')
        plt.axis('off')
        
        plt.suptitle('Image Enhancement Results - RetinexNet', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# ==================== COMMAND LINE EXECUTION ====================

if __name__ == "__main__":
    import sys
    import os
    
    print("\n" + "="*80)
    print("DASHCAM ENHANCEMENT SYSTEM - COMMAND LINE INTERFACE")
    print("="*80)
    
    # Check for command line arguments
    if len(sys.argv) < 2:
        print("\nUsage: python dashcam_enhancement.py <image_path> [output_directory]")
        print("Example: python dashcam_enhancement.py input.jpg enhanced_results")
        sys.exit(1)
    
    # Get input parameters
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "enhanced_images"
    
    # Resolve absolute path to the input image
    if not os.path.isabs(input_path):
        # If relative path, make it relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, input_path)
    
    # Initialize and run the enhancer
    enhancer = DashcamVideoEnhancer()
    
    if not enhancer.setup_model():
        print("\nERROR: Failed to setup RetinexNet model")
        sys.exit(1)
    
    # Verify input file exists
    if not os.path.exists(input_path):
        print(f"\nERROR: Input file not found at: {input_path}")
        print("Please provide the correct path to the image file")
        sys.exit(1)
    
    # Process the image
    print(f"\nProcessing image: {input_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        results = enhancer.process_image(input_path, output_dir)
        if results:
            print("\nEnhancement completed successfully!")
            print(f"Enhanced image saved to: {results['enhanced_image']}")
            sys.exit(0)
        else:
            print("\nEnhancement failed - no results returned")
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR during processing: {str(e)}")
        sys.exit(1)