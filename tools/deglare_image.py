import cv2
import numpy as np
import argparse
from dashcam_enhancer import DashcamVideoEnhancer

def main():
    parser = argparse.ArgumentParser(description='Deglare a single image using RetinexNet')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    args = parser.parse_args()

    # Initialize enhancer
    enhancer = DashcamVideoEnhancer(target_size=(640, 360))  # Adjust size as needed
    
    if not enhancer.setup_model():
        print("Failed to setup RetinexNet model")
        return

    # Read image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image {args.image}")
        return

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Enhance with RetinexNet
    result = enhancer.enhance_frame_retinex(image_rgb)
    
    # Save the enhanced image
    enhanced_bgr = cv2.cvtColor(result['enhanced'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, enhanced_bgr)
    print(f"Deglared image saved to {args.output}")

if __name__ == "__main__":
    main()