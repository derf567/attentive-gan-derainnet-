"""
CLEAN IMAGES BATCH TESTING - INTELLIGENT SWITCHING VALIDATION
==============================================================
Tests all images in Clean folder to validate:
1. Correct detection of clean/good quality images
2. System should output "No enhancement needed" (NONE mode)
3. Images should NOT be processed with DERAIN or DEGLARE
4. Validates efficiency - don't process what doesn't need processing

EXPECTED BEHAVIOR:
- Images in Clean folder SHOULD run NONE mode
- If they run DERAIN or DEGLARE, it's UNNECESSARY processing

Author: Thesis Project - Dashcam Enhancement System
Institution: Davao City Research
"""

import os
import sys
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import traceback

# Import your intelligent switch module
try:
    from intelligent_switch import IntelligentImageProcessor, ConfigManager
    print("✓ intelligent_switch module imported successfully")
except ImportError as e:
    print(f"✗ ERROR: Could not import intelligent_switch module")
    print(f"  Make sure intelligent_switch.py is in the same directory")
    print(f"  Error details: {e}")
    sys.exit(1)

class CleanFolderBatchTester:
    """
    Comprehensive batch tester for Clean folder images
    Validates that clean images are correctly identified and not processed unnecessarily
    """
    
    def __init__(self, clean_folder_path, output_folder="clean_test_results", config_path=None):
        """
        Initialize the batch tester
        
        Args:
            clean_folder_path: Path to folder containing clean images
            output_folder: Where to save test results
            config_path: Optional config file path
        """
        self.clean_folder = Path(clean_folder_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders for organized results
        self.enhanced_folder = self.output_folder / "enhanced_images"
        self.debug_folder = self.output_folder / "debug_info"
        self.reports_folder = self.output_folder / "reports"
        
        for folder in [self.enhanced_folder, self.debug_folder, self.reports_folder]:
            folder.mkdir(exist_ok=True)
        
        # Initialize processor with config
        if config_path and os.path.exists(config_path):
            self.config = ConfigManager(config_path)
            print(f"✓ Loaded configuration from {config_path}")
        else:
            # Try default config locations
            default_configs = [
                "enhanced_deglare_config.yaml",
                "config.yaml",
                "deglare_config.yaml"
            ]
            
            config_found = False
            for cfg in default_configs:
                if os.path.exists(cfg):
                    self.config = ConfigManager(cfg)
                    print(f"✓ Loaded configuration from {cfg}")
                    config_found = True
                    break
            
            if not config_found:
                print(f"⚠ Warning: No config file found, using defaults")
                self.config = ConfigManager()
        
        self.processor = IntelligentImageProcessor(self.config)
        
        # Display thresholds being used
        print(f"\n{'='*70}")
        print("DETECTION THRESHOLDS")
        print(f"{'='*70}")
        print(f"Rain threshold:  {self.processor.rain_threshold}")
        print(f"Glare threshold: {self.processor.glare_threshold}")
        print(f"Min contrast:    {self.processor.min_contrast}")
        print(f"{'='*70}\n")
        
        # Results storage
        self.test_results = []
        self.summary_stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'none_count': 0,           # CORRECT - no processing needed
            'deglare_count': 0,        # WRONG - unnecessary processing
            'derain_count': 0,         # WRONG - unnecessary processing
            'enhance_count': 0,        # MAYBE OK - depends on image quality
            'incorrect_mode_count': 0,
            'clean_unnecessarily_processed': 0,  # CRITICAL metric
        }
        
        # Timing statistics
        self.processing_times = []
        
    def get_image_files(self):
        """Get all image files from the clean folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.clean_folder.glob(f'*{ext}')))
            image_files.extend(list(self.clean_folder.glob(f'*{ext.upper()}')))
        
        return sorted(image_files)
    
    def analyze_single_image(self, image_path):
        """
        Analyze a single image and return detection scores
        WITHOUT processing it yet
        
        FIXED: Use detect_headlight_glare() instead of detect_glare()
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Get detection scores - FIXED METHOD NAMES
        rain_score = self.processor.detect_rain(img)
        glare_score, glare_mask = self.processor.detect_headlight_glare(img)  # ✅ FIXED
        
        # Check if needs enhancement using the improved method
        if hasattr(self.processor, 'needs_enhancement_improved'):
            needs_enhance = self.processor.needs_enhancement_improved(img)
        else:
            # Fallback to basic check
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            needs_enhance = np.std(gray) < self.processor.min_contrast
        
        # Additional quality metrics for clean detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        brightness = np.mean(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'rain_score': rain_score,
            'glare_score': glare_score,
            'needs_enhance': needs_enhance,
            'glare_pixel_count': np.sum(glare_mask),
            'glare_percentage': (np.sum(glare_mask) / glare_mask.size) * 100,
            'contrast': contrast,
            'brightness': brightness,
            'sharpness': sharpness
        }
    
    def process_single_image(self, image_path, image_index):
        """
        Process a single image and collect comprehensive results
        """
        image_name = image_path.name
        print(f"\n{'='*70}")
        print(f"Processing [{image_index}]: {image_name}")
        print(f"{'='*70}")
        
        result = {
            'index': image_index,
            'filename': image_name,
            'input_path': str(image_path),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Pre-analyze to get detection scores
            analysis = self.analyze_single_image(image_path)
            
            if analysis is None:
                result['status'] = 'FAILED'
                result['error'] = 'Could not read image'
                return result
            
            result['detection'] = analysis
            
            # Print detection results
            print(f"Rain Score:  {analysis['rain_score']:.4f} (threshold: {self.processor.rain_threshold})")
            print(f"Glare Score: {analysis['glare_score']:.4f} (threshold: {self.processor.glare_threshold})")
            print(f"Contrast:    {analysis['contrast']:.2f} (min: {self.processor.min_contrast})")
            print(f"Brightness:  {analysis['brightness']:.2f}")
            print(f"Sharpness:   {analysis['sharpness']:.2f}")
            print(f"Needs Enhance: {analysis['needs_enhance']}")
            
            # Predict expected mode - FOR CLEAN IMAGES: should be NONE
            if analysis['rain_score'] > self.processor.rain_threshold:
                expected_mode = 'DERAIN'
            elif analysis['glare_score'] > self.processor.glare_threshold:
                expected_mode = 'DEGLARE'
            elif analysis['needs_enhance']:
                expected_mode = 'ENHANCE'
            else:
                expected_mode = 'NONE'
            
            result['expected_mode'] = expected_mode
            print(f"Expected Mode: {expected_mode}")
            
            # Step 2: Process the image
            output_path = self.enhanced_folder / f"enhanced_{image_name}"
            
            import time
            start_time = time.time()
            
            processing_result = self.processor.process_image(
                str(image_path),
                str(output_path)
            )
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Step 3: Collect results
            result['processing_time'] = processing_time
            result['processing_result'] = processing_result
            result['status'] = processing_result.get('final_status', 'UNKNOWN')
            result['actual_mode'] = processing_result.get('processing_mode', 'UNKNOWN')
            result['output_path'] = str(output_path) if os.path.exists(output_path) else None
            
            # Check if mode matches expectation
            mode_match = (result['expected_mode'] == result['actual_mode'])
            result['mode_correct'] = mode_match
            
            print(f"Actual Mode: {result['actual_mode']}")
            print(f"Mode Match: {'✓ CORRECT' if mode_match else '✗ INCORRECT'}")
            print(f"Status: {result['status']}")
            print(f"Processing Time: {processing_time:.2f}s")
            
            # Update statistics
            if result['status'] == 'success':
                self.summary_stats['successful'] += 1
            else:
                self.summary_stats['failed'] += 1
            
            # Count modes
            actual_mode = result['actual_mode']
            if actual_mode == 'NONE':
                self.summary_stats['none_count'] += 1
            elif actual_mode == 'DEGLARE':
                self.summary_stats['deglare_count'] += 1
            elif actual_mode == 'DERAIN':
                self.summary_stats['derain_count'] += 1
            elif actual_mode == 'ENHANCE':
                self.summary_stats['enhance_count'] += 1
            
            # CRITICAL: Track clean images that were unnecessarily processed
            if actual_mode in ['DERAIN', 'DEGLARE']:
                self.summary_stats['clean_unnecessarily_processed'] += 1
                print(f"⚠⚠⚠ UNNECESSARY PROCESSING: Clean image ran {actual_mode}! ⚠⚠⚠")
            
            if not mode_match:
                self.summary_stats['incorrect_mode_count'] += 1
            
            # Copy metrics if available
            if 'metrics' in processing_result:
                result['metrics'] = processing_result['metrics']
            
            # Save individual debug info
            debug_file = self.debug_folder / f"{image_name}_debug.json"
            with open(debug_file, 'w') as f:
                json_safe_result = json.loads(
                    json.dumps(result, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
                )
                json.dump(json_safe_result, f, indent=2)
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            result['actual_mode'] = 'FAILED'
            self.summary_stats['failed'] += 1
            print(f"✗ ERROR: {e}")
            print(traceback.format_exc())
        
        return result
    
    def run_batch_test(self):
        """
        Main function to run batch test on all images
        """
        print("\n" + "="*80)
        print("CLEAN FOLDER BATCH TESTING - COMPLETE SYSTEM")
        print("="*80)
        print(f"Input folder:  {self.clean_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Config file:   {self.config.config_path}")
        print("="*80 + "\n")
        
        # Get all image files
        image_files = self.get_image_files()
        self.summary_stats['total_images'] = len(image_files)
        
        if len(image_files) == 0:
            print(f"✗ ERROR: No images found in {self.clean_folder}")
            return
        
        print(f"Found {len(image_files)} images to process\n")
        
        # Process each image with progress bar
        for idx, image_path in enumerate(tqdm(image_files, desc="Processing images"), start=1):
            result = self.process_single_image(image_path, idx)
            self.test_results.append(result)
        
        # Generate comprehensive reports
        self.generate_reports()
        
    def generate_reports(self):
        """Generate comprehensive reports after batch processing"""
        print("\n" + "="*80)
        print("GENERATING REPORTS")
        print("="*80)
        
        self._generate_summary_report()
        self._generate_csv_report()
        self._generate_incorrect_mode_report()
        self._generate_performance_report()
        self._generate_html_report()
        
        print("\n✓ All reports generated successfully!")
        print(f"📁 Reports location: {self.reports_folder}")
    
    def _generate_summary_report(self):
        """Generate text summary report"""
        summary_file = self.reports_folder / "summary_report.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CLEAN FOLDER BATCH TEST SUMMARY\n")
            f.write("THESIS PROJECT - DASHCAM ENHANCEMENT SYSTEM\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Folder: {self.clean_folder}\n")
            f.write(f"Output Folder: {self.output_folder}\n\n")
            
            f.write("PROCESSING STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Images Tested:     {self.summary_stats['total_images']}\n")
            f.write(f"Successfully Processed:  {self.summary_stats['successful']}\n")
            f.write(f"Failed:                  {self.summary_stats['failed']}\n\n")
            
            f.write("PROCESSING MODE DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"NONE Mode (Correct):     {self.summary_stats['none_count']} images ({self.summary_stats['none_count']/self.summary_stats['total_images']*100:.1f}%)\n")
            f.write(f"DERAIN (Unnecessary):    {self.summary_stats['derain_count']} images ({self.summary_stats['derain_count']/self.summary_stats['total_images']*100:.1f}%)\n")
            f.write(f"DEGLARE (Unnecessary):   {self.summary_stats['deglare_count']} images ({self.summary_stats['deglare_count']/self.summary_stats['total_images']*100:.1f}%)\n")
            f.write(f"ENHANCE (Maybe OK):      {self.summary_stats['enhance_count']} images ({self.summary_stats['enhance_count']/self.summary_stats['total_images']*100:.1f}%)\n\n")
            
            f.write("EFFICIENCY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            efficiency = (self.summary_stats['none_count'] / self.summary_stats['total_images']) * 100
            f.write(f"Efficiency Rate: {efficiency:.1f}%\n")
            f.write(f"  (Percentage of clean images correctly identified as not needing processing)\n\n")
            
            f.write("⚠ CRITICAL METRIC ⚠\n")
            f.write("-" * 80 + "\n")
            f.write(f"Unnecessary Processing: {self.summary_stats['clean_unnecessarily_processed']} images\n")
            
            if self.summary_stats['clean_unnecessarily_processed'] > 0:
                f.write(f"\n⚠ WARNING: {self.summary_stats['clean_unnecessarily_processed']} clean images were unnecessarily processed!\n")
                f.write(f"This wastes computational resources on images that don't need enhancement.\n")
            else:
                f.write(f"\n✓ EXCELLENT: All clean images correctly identified!\n")
                f.write(f"No unnecessary processing - maximum efficiency achieved.\n")
        
        print(f"✓ Summary report saved: {summary_file}")
        
        # Print to console
        print("\n" + "="*80)
        print("BATCH TEST RESULTS SUMMARY")
        print("="*80)
        print(f"Total:       {self.summary_stats['total_images']}")
        print(f"NONE Mode:   {self.summary_stats['none_count']} ({'✓ Correct' if self.summary_stats['none_count'] > self.summary_stats['total_images']*0.8 else '⚠ Low'})")
        print(f"Unnecessary: {self.summary_stats['clean_unnecessarily_processed']}")
        print(f"Efficiency:  {(self.summary_stats['none_count']/self.summary_stats['total_images']*100):.1f}%")
        print("="*80)
    
    def _generate_csv_report(self):
        """Generate detailed CSV report"""
        csv_file = self.reports_folder / "detailed_results.csv"
        
        rows = []
        for result in self.test_results:
            row = {
                'Index': result['index'],
                'Filename': result['filename'],
                'Status': result['status'],
                'Expected_Mode': result.get('expected_mode', 'N/A'),
                'Actual_Mode': result.get('actual_mode', 'N/A'),
                'Mode_Correct': result.get('mode_correct', False),
                'Processing_Time_s': result.get('processing_time', 0),
            }
            
            if 'detection' in result:
                row['Rain_Score'] = result['detection']['rain_score']
                row['Glare_Score'] = result['detection']['glare_score']
                row['Contrast'] = result['detection']['contrast']
                row['Brightness'] = result['detection']['brightness']
                row['Sharpness'] = result['detection']['sharpness']
            
            row['Unnecessary_Processing'] = result.get('actual_mode') in ['DERAIN', 'DEGLARE']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        
        print(f"✓ CSV report saved: {csv_file}")
    
    def _generate_incorrect_mode_report(self):
        """Generate report for unnecessary processing"""
        incorrect_file = self.reports_folder / "unnecessary_processing.txt"
        
        unnecessary = [r for r in self.test_results 
                      if r.get('actual_mode') in ['DERAIN', 'DEGLARE']]
        
        with open(incorrect_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("UNNECESSARY PROCESSING ANALYSIS - CLEAN FOLDER\n")
            f.write("="*80 + "\n\n")
            
            if len(unnecessary) == 0:
                f.write("✓✓✓ PERFECT EFFICIENCY ✓✓✓\n\n")
                f.write("All clean images correctly identified - no unnecessary processing!\n")
            else:
                f.write(f"Found {len(unnecessary)} images with unnecessary processing:\n\n")
                
                for idx, result in enumerate(unnecessary, 1):
                    f.write(f"\n{'─'*80}\n")
                    f.write(f"CASE #{idx}\n")
                    f.write(f"{'─'*80}\n")
                    f.write(f"File: {result['filename']}\n")
                    f.write(f"Mode: {result.get('actual_mode')} (should be NONE)\n\n")
                    
                    if 'detection' in result:
                        det = result['detection']
                        f.write(f"Detection Scores:\n")
                        f.write(f"  Rain:  {det['rain_score']:.6f}\n")
                        f.write(f"  Glare: {det['glare_score']:.6f}\n")
                        f.write(f"  Contrast: {det['contrast']:.2f}\n")
                        f.write(f"  Brightness: {det['brightness']:.2f}\n")
        
        print(f"✓ Unnecessary processing report saved: {incorrect_file}")
    
    def _generate_performance_report(self):
        """Generate performance stats"""
        perf_file = self.reports_folder / "performance_stats.txt"
        
        with open(perf_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PERFORMANCE STATISTICS - CLEAN FOLDER\n")
            f.write("="*80 + "\n\n")
            
            if len(self.processing_times) > 0:
                times = np.array(self.processing_times)
                f.write(f"Total Images:            {len(times)}\n")
                f.write(f"Mean Processing Time:    {np.mean(times):.3f}s\n")
                f.write(f"Total Time:              {np.sum(times):.2f}s\n")
        
        print(f"✓ Performance report saved: {perf_file}")
    
    def _generate_html_report(self):
        """Generate visual HTML report"""
        html_file = self.reports_folder / "visual_report.html"
        
        efficiency = (self.summary_stats['none_count'] / self.summary_stats['total_images']) * 100 if self.summary_stats['total_images'] > 0 else 0
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Clean Folder Batch Test Results</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                   color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; }}
        .stat-value.success {{ color: #27ae60; }}
        .stat-value.warning {{ color: #f39c12; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #11998e; color: white; }}
        .unnecessary {{ background-color: #fff3cd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin:0; color:white;">🎓 Clean Images Batch Test</h1>
        <p style="margin:10px 0 0 0;">Efficiency & Clean Detection Validation</p>
    </div>
    
    <div class="stat-grid">
        <div class="stat-card">
            <div style="color:#7f8c8d; font-size:14px;">Total Images</div>
            <div class="stat-value">{self.summary_stats['total_images']}</div>
        </div>
        <div class="stat-card">
            <div style="color:#7f8c8d; font-size:14px;">NONE Mode (Correct)</div>
            <div class="stat-value success">{self.summary_stats['none_count']}</div>
        </div>
        <div class="stat-card">
            <div style="color:#7f8c8d; font-size:14px;">Efficiency</div>
            <div class="stat-value {'success' if efficiency >= 80 else 'warning'}">{efficiency:.1f}%</div>
        </div>
        <div class="stat-card">
            <div style="color:#7f8c8d; font-size:14px;">Unnecessary Processing</div>
            <div class="stat-value warning">{self.summary_stats['clean_unnecessarily_processed']}</div>
        </div>
    </div>
    
    <div style="background:white; padding:20px; border-radius:8px; margin:20px 0;">
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Filename</th>
                <th>Actual Mode</th>
                <th>Rain Score</th>
                <th>Glare Score</th>
                <th>Contrast</th>
                <th>Time (s)</th>
            </tr>
"""
        
        for result in self.test_results:
            is_unnecessary = result.get('actual_mode') in ['DERAIN', 'DEGLARE']
            row_class = 'unnecessary' if is_unnecessary else ''
            
            det = result.get('detection', {})
            
            html_content += f"""
            <tr class="{row_class}">
                <td>{result['index']}</td>
                <td style="font-family:monospace; font-size:12px;">{result['filename']}</td>
                <td><strong>{result.get('actual_mode', 'N/A')}</strong></td>
                <td>{det.get('rain_score', 0):.4f}</td>
                <td>{det.get('glare_score', 0):.4f}</td>
                <td>{det.get('contrast', 0):.2f}</td>
                <td>{result.get('processing_time', 0):.2f}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved: {html_file}")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch test Clean folder images')
    parser.add_argument('--clean_folder', type=str, default='Clean',
                       help='Path to folder containing clean images')
    parser.add_argument('--output_folder', type=str, default='clean_test_results',
                       help='Output folder for results')
    parser.add_argument('--config', type=str, default='enhanced_deglare_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CLEAN FOLDER BATCH TESTER")
    print("Thesis Project - Efficiency Validation")
    print("="*80 + "\n")
    
    if not os.path.exists(args.clean_folder):
        print(f"✗ ERROR: Clean folder not found: {args.clean_folder}")
        sys.exit(1)
    
    try:
        tester = CleanFolderBatchTester(args.clean_folder, args.output_folder, args.config)
        tester.run_batch_test()
        
        print("\n" + "="*80)
        print("TESTING COMPLETED!")
        print("="*80)
        print(f"\n📊 Results:")
        print(f"   NONE Mode:   {tester.summary_stats['none_count']} / {tester.summary_stats['total_images']}")
        if tester.summary_stats['total_images'] > 0:
            print(f"   Efficiency:  {(tester.summary_stats['none_count']/tester.summary_stats['total_images']*100):.1f}%")
        print(f"   Unnecessary: {tester.summary_stats['clean_unnecessarily_processed']}")
        
        if tester.summary_stats['clean_unnecessarily_processed'] == 0:
            print(f"\n✓✓✓ PERFECT: No unnecessary processing!")
        else:
            print(f"\n⚠ {tester.summary_stats['clean_unnecessarily_processed']} images unnecessarily processed")
        
        print(f"\n📁 Reports: {args.output_folder}/reports/")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)