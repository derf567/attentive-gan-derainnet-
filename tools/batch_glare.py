"""
INTELLIGENT SWITCHING VALIDATION TEST - COMPLETE SYSTEM
========================================================
Tests all images in Glare folder to validate:
1. Correct glare detection
2. Proper processing mode selection (DEGLARE vs DERAIN)
3. Success rate of each processing mode
4. Detailed metrics for thesis presentation

Works with:
- intelligent_switch.py (main processing router)
- test_model.py (deraining module)
- enhanced_deglare_config.yaml (configuration)

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

class GlareFolderBatchTester:
    """
    Comprehensive batch tester for Glare folder images
    Validates intelligent switching behavior and generates detailed reports
    """
    
    def __init__(self, glare_folder_path, output_folder="batch_test_results", config_path=None):
        """
        Initialize the batch tester
        
        Args:
            glare_folder_path: Path to folder containing glare images
            output_folder: Where to save test results
            config_path: Optional config file path
        """
        self.glare_folder = Path(glare_folder_path)
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
        print(f"{'='*70}\n")
        
        # Results storage
        self.test_results = []
        self.summary_stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'deglare_count': 0,
            'derain_count': 0,
            'enhance_count': 0,
            'none_count': 0,
            'incorrect_mode_count': 0,
            'glare_ran_derain': 0,  # Critical: glare images that ran derain
        }
        
        # Timing statistics
        self.processing_times = []
        
    def get_image_files(self):
        """Get all image files from the glare folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.glare_folder.glob(f'*{ext}')))
            image_files.extend(list(self.glare_folder.glob(f'*{ext.upper()}')))
        
        return sorted(image_files)
    
    def analyze_single_image(self, image_path):
        """
        Analyze a single image and return detection scores
        WITHOUT processing it yet
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Get detection scores
        rain_score = self.processor.detect_rain(img)
        glare_score, glare_mask = self.processor.detect_glare(img)
        needs_enhance = self.processor.needs_enhancement(img)
        
        return {
            'rain_score': rain_score,
            'glare_score': glare_score,
            'needs_enhance': needs_enhance,
            'glare_pixel_count': np.sum(glare_mask),
            'glare_percentage': (np.sum(glare_mask) / glare_mask.size) * 100
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
            print(f"Glare Pixels: {analysis['glare_pixel_count']:,} ({analysis['glare_percentage']:.2f}%)")
            print(f"Needs Enhance: {analysis['needs_enhance']}")
            
            # Predict expected mode based on thresholds
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
            if actual_mode == 'DEGLARE':
                self.summary_stats['deglare_count'] += 1
            elif actual_mode == 'DERAIN':
                self.summary_stats['derain_count'] += 1
            elif actual_mode == 'ENHANCE':
                self.summary_stats['enhance_count'] += 1
            elif actual_mode == 'NONE':
                self.summary_stats['none_count'] += 1
            
            # CRITICAL: Track glare images that incorrectly ran derain
            if not mode_match:
                self.summary_stats['incorrect_mode_count'] += 1
                
                if expected_mode == 'DEGLARE' and actual_mode == 'DERAIN':
                    self.summary_stats['glare_ran_derain'] += 1
                    print(f"⚠⚠⚠ CRITICAL: GLARE IMAGE RAN DERAIN! ⚠⚠⚠")
            
            # Copy metrics if available
            if 'metrics' in processing_result:
                result['metrics'] = processing_result['metrics']
            
            # Check if deraining files were created
            derain_outputs = []
            for derain_file in ['derain_ret.png', 'src_img.png', 'comparison.png', 'derain_results.txt']:
                if os.path.exists(derain_file):
                    derain_outputs.append(derain_file)
            
            if derain_outputs:
                result['derain_outputs'] = derain_outputs
                print(f"Deraining output files found: {len(derain_outputs)}")
            
            # Save individual debug info
            debug_file = self.debug_folder / f"{image_name}_debug.json"
            with open(debug_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
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
        print("GLARE FOLDER BATCH TESTING - COMPLETE SYSTEM")
        print("="*80)
        print(f"Input folder:  {self.glare_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Config file:   {self.config.config_path}")
        print("="*80 + "\n")
        
        # Get all image files
        image_files = self.get_image_files()
        self.summary_stats['total_images'] = len(image_files)
        
        if len(image_files) == 0:
            print(f"✗ ERROR: No images found in {self.glare_folder}")
            return
        
        print(f"Found {len(image_files)} images to process\n")
        
        # Process each image with progress bar
        for idx, image_path in enumerate(tqdm(image_files, desc="Processing images"), start=1):
            result = self.process_single_image(image_path, idx)
            self.test_results.append(result)
        
        # Generate comprehensive reports
        self.generate_reports()
        
    def generate_reports(self):
        """
        Generate comprehensive reports after batch processing
        """
        print("\n" + "="*80)
        print("GENERATING REPORTS")
        print("="*80)
        
        # 1. Summary Statistics Report
        self._generate_summary_report()
        
        # 2. Detailed CSV Report
        self._generate_csv_report()
        
        # 3. Incorrect Mode Analysis (CRITICAL FOR THESIS)
        self._generate_incorrect_mode_report()
        
        # 4. Performance Statistics
        self._generate_performance_report()
        
        # 5. Visual HTML Report
        self._generate_html_report()
        
        print("\n✓ All reports generated successfully!")
        print(f"📁 Reports location: {self.reports_folder}")
    
    def _generate_summary_report(self):
        """Generate text summary report"""
        summary_file = self.reports_folder / "summary_report.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("INTELLIGENT SWITCHING BATCH TEST SUMMARY\n")
            f.write("THESIS PROJECT - DASHCAM ENHANCEMENT SYSTEM\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Folder: {self.glare_folder}\n")
            f.write(f"Output Folder: {self.output_folder}\n")
            f.write(f"Configuration: {self.config.config_path}\n\n")
            
            f.write("DETECTION THRESHOLDS USED\n")
            f.write("-" * 80 + "\n")
            f.write(f"Rain Detection Threshold:  {self.processor.rain_threshold}\n")
            f.write(f"Glare Detection Threshold: {self.processor.glare_threshold}\n\n")
            
            f.write("PROCESSING STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Images Tested:     {self.summary_stats['total_images']}\n")
            f.write(f"Successfully Processed:  {self.summary_stats['successful']} ({self.summary_stats['successful']/self.summary_stats['total_images']*100:.1f}%)\n")
            f.write(f"Failed:                  {self.summary_stats['failed']} ({self.summary_stats['failed']/self.summary_stats['total_images']*100:.1f}%)\n\n")
            
            f.write("PROCESSING MODE DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"DEGLARE Mode:            {self.summary_stats['deglare_count']} images ({self.summary_stats['deglare_count']/self.summary_stats['total_images']*100:.1f}%)\n")
            f.write(f"DERAIN Mode:             {self.summary_stats['derain_count']} images ({self.summary_stats['derain_count']/self.summary_stats['total_images']*100:.1f}%)\n")
            f.write(f"ENHANCE Mode:            {self.summary_stats['enhance_count']} images ({self.summary_stats['enhance_count']/self.summary_stats['total_images']*100:.1f}%)\n")
            f.write(f"NONE Mode:               {self.summary_stats['none_count']} images ({self.summary_stats['none_count']/self.summary_stats['total_images']*100:.1f}%)\n\n")
            
            f.write("MODE ACCURACY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            correct_count = self.summary_stats['total_images'] - self.summary_stats['incorrect_mode_count']
            accuracy = (correct_count / self.summary_stats['total_images']) * 100
            f.write(f"Correct Mode Selection:  {correct_count} / {self.summary_stats['total_images']} ({accuracy:.1f}%)\n")
            f.write(f"Incorrect Mode:          {self.summary_stats['incorrect_mode_count']} images ({self.summary_stats['incorrect_mode_count']/self.summary_stats['total_images']*100:.1f}%)\n\n")
            
            # CRITICAL SECTION FOR THESIS
            f.write("⚠ CRITICAL ISSUE DETECTION ⚠\n")
            f.write("-" * 80 + "\n")
            f.write(f"Glare Images that Ran DERAIN: {self.summary_stats['glare_ran_derain']}\n")
            
            if self.summary_stats['glare_ran_derain'] > 0:
                f.write(f"\n⚠⚠⚠ WARNING ⚠⚠⚠\n")
                f.write(f"{self.summary_stats['glare_ran_derain']} glare images incorrectly processed with DERAIN!\n")
                f.write(f"This indicates the intelligent switching is NOT working correctly.\n")
                f.write(f"These images should have been processed with DEGLARE mode.\n")
                f.write(f"Check 'incorrect_modes.txt' for detailed analysis.\n")
            else:
                f.write(f"\n✓ EXCELLENT: All glare images correctly processed with DEGLARE mode!\n")
            
            if len(self.processing_times) > 0:
                f.write("\n\nPERFORMANCE METRICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Average Processing Time: {np.mean(self.processing_times):.2f}s\n")
                f.write(f"Fastest:                 {np.min(self.processing_times):.2f}s\n")
                f.write(f"Slowest:                 {np.max(self.processing_times):.2f}s\n")
                f.write(f"Total Time:              {np.sum(self.processing_times):.2f}s ({np.sum(self.processing_times)/60:.1f} minutes)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Summary report saved: {summary_file}")
        
        # Also print critical stats to console
        print("\n" + "="*80)
        print("BATCH TEST RESULTS SUMMARY")
        print("="*80)
        print(f"Total:     {self.summary_stats['total_images']}")
        print(f"Success:   {self.summary_stats['successful']}")
        print(f"Failed:    {self.summary_stats['failed']}")
        print(f"DEGLARE:   {self.summary_stats['deglare_count']}")
        print(f"DERAIN:    {self.summary_stats['derain_count']}")
        print(f"Incorrect: {self.summary_stats['incorrect_mode_count']}")
        if self.summary_stats['glare_ran_derain'] > 0:
            print(f"\n⚠⚠⚠ CRITICAL: {self.summary_stats['glare_ran_derain']} glare images ran DERAIN!")
        else:
            print(f"\n✓ Perfect: All glare images ran DEGLARE correctly!")
        print("="*80)
    
    def _generate_csv_report(self):
        """Generate detailed CSV report for analysis"""
        csv_file = self.reports_folder / "detailed_results.csv"
        
        # Prepare data for DataFrame
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
            
            # Add detection metrics
            if 'detection' in result:
                row['Rain_Score'] = result['detection']['rain_score']
                row['Glare_Score'] = result['detection']['glare_score']
                row['Glare_Percentage'] = result['detection']['glare_percentage']
                row['Needs_Enhancement'] = result['detection']['needs_enhance']
            
            # Flag critical issues
            row['Critical_Issue'] = (
                result.get('expected_mode') == 'DEGLARE' and 
                result.get('actual_mode') == 'DERAIN'
            )
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        
        print(f"✓ CSV report saved: {csv_file}")
    
    def _generate_incorrect_mode_report(self):
        """Generate report specifically for incorrect mode selections - CRITICAL FOR THESIS"""
        incorrect_file = self.reports_folder / "incorrect_modes.txt"
        
        incorrect_results = [r for r in self.test_results if not r.get('mode_correct', True)]
        glare_ran_derain = [r for r in incorrect_results 
                           if r.get('expected_mode') == 'DEGLARE' and r.get('actual_mode') == 'DERAIN']
        
        with open(incorrect_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("INCORRECT MODE SELECTION ANALYSIS\n")
            f.write("CRITICAL REPORT FOR THESIS PRESENTATION\n")
            f.write("="*80 + "\n\n")
            
            if len(incorrect_results) == 0:
                f.write("✓✓✓ EXCELLENT RESULT ✓✓✓\n\n")
                f.write("All images processed with correct mode selection!\n")
                f.write("The intelligent switching system is working perfectly.\n")
            else:
                f.write(f"Found {len(incorrect_results)} images with incorrect mode:\n\n")
                
                # Highlight critical glare->derain cases
                if len(glare_ran_derain) > 0:
                    f.write("⚠⚠⚠ CRITICAL ISSUES ⚠⚠⚠\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"{len(glare_ran_derain)} GLARE IMAGES INCORRECTLY RAN DERAIN\n")
                    f.write("These are the most critical failures for your thesis:\n\n")
                    
                    for idx, result in enumerate(glare_ran_derain, 1):
                        f.write(f"\n{'─'*80}\n")
                        f.write(f"CRITICAL CASE #{idx}\n")
                        f.write(f"{'─'*80}\n")
                        f.write(f"File: {result['filename']}\n")
                        f.write(f"Expected: DEGLARE (correct for glare folder)\n")
                        f.write(f"Actual: DERAIN (WRONG!)\n\n")
                        
                        if 'detection' in result:
                            det = result['detection']
                            f.write(f"Detection Scores:\n")
                            f.write(f"  Rain Score:  {det['rain_score']:.6f} (threshold: {self.processor.rain_threshold})\n")
                            f.write(f"  Glare Score: {det['glare_score']:.6f} (threshold: {self.processor.glare_threshold})\n")
                            f.write(f"  Glare %:     {det['glare_percentage']:.2f}%\n\n")
                            
                            # Analysis
                            if det['rain_score'] > self.processor.rain_threshold:
                                f.write(f"Analysis: Rain score ({det['rain_score']:.4f}) exceeded threshold ({self.processor.rain_threshold})\n")
                                f.write(f"Recommendation: Increase rain threshold or improve rain detection algorithm\n")
                        
                        f.write("\n")
                
                # Other incorrect cases
                other_incorrect = [r for r in incorrect_results if r not in glare_ran_derain]
                if other_incorrect:
                    f.write(f"\n\nOTHER INCORRECT MODE CASES ({len(other_incorrect)})\n")
                    f.write("=" * 80 + "\n")
                    
                    for result in other_incorrect:
                        f.write("-" * 80 + "\n")
                        f.write(f"File: {result['filename']}\n")
                        f.write(f"Expected: {result.get('expected_mode', 'N/A')}\n")
                        f.write(f"Actual: {result.get('actual_mode', 'N/A')}\n")
                        
                        if 'detection' in result:
                            det = result['detection']
                            f.write(f"Rain Score: {det['rain_score']:.4f}\n")
                            f.write(f"Glare Score: {det['glare_score']:.4f}\n")
                            f.write(f"Glare %: {det['glare_percentage']:.2f}%\n")
                        
                        f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS FOR THESIS\n")
            f.write("="*80 + "\n")
            
            if len(glare_ran_derain) > 0:
                f.write("\n1. ADJUST THRESHOLDS:\n")
                f.write(f"   Current rain threshold: {self.processor.rain_threshold}\n")
                f.write(f"   Current glare threshold: {self.processor.glare_threshold}\n")
                f.write(f"   Suggestion: Increase rain threshold to reduce false rain detection\n\n")
                
                f.write("2. IMPROVE RAIN DETECTION:\n")
                f.write("   Analyze why glare images are being detected as rain\n")
                f.write("   Consider adding glare pre-check before rain detection\n\n")
                
                f.write("3. PRIORITY ORDER:\n")
                f.write("   Consider checking glare BEFORE rain in the processing pipeline\n")
            else:
                f.write("\n✓ System is working correctly!\n")
                f.write("  All glare images properly processed with DEGLARE mode.\n")
        
        print(f"✓ Incorrect modes report saved: {incorrect_file}")
    
    def _generate_performance_report(self):
        """Generate performance statistics report"""
        perf_file = self.reports_folder / "performance_stats.txt"
        
        with open(perf_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PERFORMANCE STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            if len(self.processing_times) > 0:
                times = np.array(self.processing_times)
                
                f.write(f"Total Images:            {len(times)}\n")
                f.write(f"Mean Processing Time:    {np.mean(times):.3f}s\n")
                f.write(f"Median Processing Time:  {np.median(times):.3f}s\n")
                f.write(f"Std Dev:                 {np.std(times):.3f}s\n")
                f.write(f"Min Time:                {np.min(times):.3f}s\n")
                f.write(f"Max Time:                {np.max(times):.3f}s\n")
                f.write(f"Total Time:              {np.sum(times):.2f}s ({np.sum(times)/60:.1f} min)\n")
                
                # Percentiles
                f.write(f"\nPercentiles:\n")
                f.write(f"  25th:                  {np.percentile(times, 25):.3f}s\n")
                f.write(f"  50th (median):         {np.percentile(times, 50):.3f}s\n")
                f.write(f"  75th:                  {np.percentile(times, 75):.3f}s\n")
                f.write(f"  95th:                  {np.percentile(times, 95):.3f}s\n")
                
                # Processing rate
                f.write(f"\nProcessing Rate:\n")
                f.write(f"  Images per minute:     {60/np.mean(times):.1f}\n")
                f.write(f"  Images per hour:       {3600/np.mean(times):.0f}\n")
        
        print(f"✓ Performance report saved: {perf_file}")
    
    def _generate_html_report(self):
        """Generate visual HTML report for thesis presentation"""
        html_file = self.reports_folder / "visual_report.html"
        
        # Calculate additional stats
        correct_count = self.summary_stats['total_images'] - self.summary_stats['incorrect_mode_count']
        accuracy = (correct_count / self.summary_stats['total_images']) * 100
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Batch Test Results - Intelligent Switching Validation</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .stats {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }}
        .stat-label {{ font-weight: bold; color: #7f8c8d; font-size: 14px; margin-bottom: 10px; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .stat-value.success {{ color: #27ae60; }}
        .stat-value.failed {{ color: #e74c3c; }}
        .stat-value.warning {{ color: #f39c12; }}
        .critical-alert {{ background: #ffe6e6; border-left: 5px solid #e74c3c; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .success-alert {{ background: #e8f8f5; border-left: 5px solid #27ae60; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; font-weight: bold; position: sticky; top: 0; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .failed {{ color: #e74c3c; font-weight: bold; }}
        .incorrect {{ background-color: #ffe6e6; }}
        .critical {{ background-color: #ffcccc; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #7f8c8d; }}
        .progress-bar {{ width: 100%; height: 30px; background: #ecf0f1; border-radius: 15px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #27ae60, #2ecc71); transition: width 0.3s; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin:0; border:none; color:white;">🎓 Thesis Project: Intelligent Switching Validation</h1>
        <p style="margin:10px 0 0 0; opacity:0.9;">Dashcam Enhancement System - Glare Folder Batch Test</p>
        <p style="margin:5px 0 0 0; opacity:0.8; font-size:14px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Total Images</div>
            <div class="stat-value">{self.summary_stats['total_images']}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Successfully Processed</div>
            <div class="stat-value success">{self.summary_stats['successful']}</div>
            <div style="font-size:12px; color:#7f8c8d; margin-top:5px;">
                {self.summary_stats['successful']/self.summary_stats['total_images']*100:.1f}%
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">DEGLARE Mode</div>
            <div class="stat-value" style="color:#3498db;">{self.summary_stats['deglare_count']}</div>
            <div style="font-size:12px; color:#7f8c8d; margin-top:5px;">
                {self.summary_stats['deglare_count']/self.summary_stats['total_images']*100:.1f}%
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">DERAIN Mode</div>
            <div class="stat-value warning">{self.summary_stats['derain_count']}</div>
            <div style="font-size:12px; color:#7f8c8d; margin-top:5px;">
                {self.summary_stats['derain_count']/self.summary_stats['total_images']*100:.1f}%
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Mode Accuracy</div>
            <div class="stat-value {'success' if accuracy >= 90 else 'warning'}">{accuracy:.1f}%</div>
            <div style="font-size:12px; color:#7f8c8d; margin-top:5px;">
                {correct_count} / {self.summary_stats['total_images']} correct
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Critical Issues</div>
            <div class="stat-value {'failed' if self.summary_stats['glare_ran_derain'] > 0 else 'success'}">
                {self.summary_stats['glare_ran_derain']}
            </div>
            <div style="font-size:12px; color:#7f8c8d; margin-top:5px;">
                Glare → Derain errors
            </div>
        </div>
    </div>
    
    {'<div class="critical-alert"><h3 style="margin-top:0; color:#e74c3c;">⚠ CRITICAL ISSUE DETECTED</h3><p><strong>' + str(self.summary_stats['glare_ran_derain']) + ' glare images incorrectly processed with DERAIN mode!</strong></p><p>These images from the Glare folder should have been processed with DEGLARE mode. This indicates the intelligent switching system needs adjustment.</p><p><strong>Recommendation:</strong> Check rain detection threshold and priority order in the processing pipeline.</p></div>' if self.summary_stats['glare_ran_derain'] > 0 else '<div class="success-alert"><h3 style="margin-top:0; color:#27ae60;">✓ EXCELLENT PERFORMANCE</h3><p><strong>All glare images correctly processed with DEGLARE mode!</strong></p><p>The intelligent switching system is working perfectly. All images were routed to the appropriate processing mode.</p></div>'}
    
    <div class="stats">
        <h2>Processing Accuracy Breakdown</h2>
        <div style="margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Correct Mode Selection</span>
                <span><strong>{accuracy:.1f}%</strong></span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {accuracy}%;"></div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 30px;">
            <div>
                <h3>Mode Distribution</h3>
                <ul style="list-style: none; padding: 0;">
                    <li style="padding: 8px 0; border-bottom: 1px solid #ecf0f1;">
                        <span style="color: #3498db;">●</span> <strong>DEGLARE:</strong> {self.summary_stats['deglare_count']} images
                    </li>
                    <li style="padding: 8px 0; border-bottom: 1px solid #ecf0f1;">
                        <span style="color: #f39c12;">●</span> <strong>DERAIN:</strong> {self.summary_stats['derain_count']} images
                    </li>
                    <li style="padding: 8px 0; border-bottom: 1px solid #ecf0f1;">
                        <span style="color: #9b59b6;">●</span> <strong>ENHANCE:</strong> {self.summary_stats['enhance_count']} images
                    </li>
                    <li style="padding: 8px 0;">
                        <span style="color: #95a5a6;">●</span> <strong>NONE:</strong> {self.summary_stats['none_count']} images
                    </li>
                </ul>
            </div>
            <div>
                <h3>Status Summary</h3>
                <ul style="list-style: none; padding: 0;">
                    <li style="padding: 8px 0; border-bottom: 1px solid #ecf0f1;">
                        <span style="color: #27ae60;">✓</span> <strong>Successful:</strong> {self.summary_stats['successful']} images
                    </li>
                    <li style="padding: 8px 0; border-bottom: 1px solid #ecf0f1;">
                        <span style="color: #e74c3c;">✗</span> <strong>Failed:</strong> {self.summary_stats['failed']} images
                    </li>
                    <li style="padding: 8px 0;">
                        <span style="color: #e74c3c;">⚠</span> <strong>Incorrect Mode:</strong> {self.summary_stats['incorrect_mode_count']} images
                    </li>
                </ul>
            </div>
        </div>
    </div>
    
    <div class="stats">
        <h2>Detailed Results Table</h2>
        <p style="color: #7f8c8d; font-size: 14px;">
            Red highlighted rows indicate images that ran incorrect processing mode
        </p>
        <div style="overflow-x: auto;">
            <table>
                <tr>
                    <th>#</th>
                    <th>Filename</th>
                    <th>Status</th>
                    <th>Expected Mode</th>
                    <th>Actual Mode</th>
                    <th>Match</th>
                    <th>Glare Score</th>
                    <th>Rain Score</th>
                    <th>Time (s)</th>
                </tr>
"""
        
        for result in self.test_results:
            mode_match = result.get('mode_correct', True)
            is_critical = (result.get('expected_mode') == 'DEGLARE' and 
                          result.get('actual_mode') == 'DERAIN')
            
            row_class = 'critical' if is_critical else ('incorrect' if not mode_match else '')
            match_symbol = '✓' if mode_match else '✗'
            status_class = 'success' if result['status'] == 'success' else 'failed'
            
            det = result.get('detection', {})
            
            html_content += f"""
                <tr class="{row_class}">
                    <td>{result['index']}</td>
                    <td style="font-family: monospace; font-size: 12px;">{result['filename']}</td>
                    <td class="{status_class}">{result['status']}</td>
                    <td><strong>{result.get('expected_mode', 'N/A')}</strong></td>
                    <td><strong>{result.get('actual_mode', 'N/A')}</strong></td>
                    <td style="font-size: 18px;">{'<span style="color:#27ae60;">✓</span>' if mode_match else '<span style="color:#e74c3c;">✗</span>'}</td>
                    <td>{det.get('glare_score', 0):.4f}</td>
                    <td>{det.get('rain_score', 0):.4f}</td>
                    <td>{result.get('processing_time', 0):.2f}</td>
                </tr>
"""
        
        html_content += """
            </table>
        </div>
    </div>
    
    <div class="stats">
        <h2>Configuration Used</h2>
        <div style="background: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 13px;">
"""
        
        html_content += f"""
            <strong>Config File:</strong> {self.config.config_path}<br>
            <strong>Rain Threshold:</strong> {self.processor.rain_threshold}<br>
            <strong>Glare Threshold:</strong> {self.processor.glare_threshold}<br>
            <strong>Glare Brightness Threshold:</strong> {self.processor.glare_brightness_threshold}<br>
            <strong>Glare Saturation Threshold:</strong> {self.processor.glare_saturation_threshold}
"""
        
        html_content += """
        </div>
    </div>
    
    <div class="footer">
        <p><strong>Thesis Project: Dashcam Enhancement System</strong></p>
        <p>Intelligent Switching Validation Report</p>
        <p style="font-size: 12px; margin-top: 10px;">
            This report validates the intelligent switching mechanism that routes images<br>
            to appropriate enhancement algorithms (DEGLARE, DERAIN, ENHANCE)
        </p>
    </div>
</body>
</html>
"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved: {html_file}")
        print(f"  Open in browser for visual presentation")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch test all images in Glare folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python glare_batch_tester.py
  python glare_batch_tester.py --glare_folder "path/to/Glare"
  python glare_batch_tester.py --config enhanced_deglare_config.yaml
  python glare_batch_tester.py --glare_folder Glare --output_folder my_test_results
        """
    )
    
    parser.add_argument('--glare_folder', type=str, default='Glare',
                       help='Path to folder containing glare images (default: Glare)')
    parser.add_argument('--output_folder', type=str, default='batch_test_results',
                       help='Output folder for results (default: batch_test_results)')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path (default: auto-detect)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("INTELLIGENT SWITCHING BATCH TESTER")
    print("Thesis Project - Dashcam Enhancement System")
    print("="*80 + "\n")
    
    # Verify glare folder exists
    if not os.path.exists(args.glare_folder):
        print(f"✗ ERROR: Glare folder not found: {args.glare_folder}")
        print("\nPlease ensure:")
        print("  1. The Glare folder exists in the current directory")
        print("  2. Or provide the correct path using --glare_folder argument")
        print("\nExample:")
        print(f"  python {sys.argv[0]} --glare_folder 'path/to/your/Glare'")
        sys.exit(1)
    
    # Initialize and run batch tester
    try:
        tester = GlareFolderBatchTester(
            args.glare_folder, 
            args.output_folder,
            args.config
        )
        tester.run_batch_test()
        
        print("\n" + "="*80)
        print("BATCH TESTING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📊 Results Summary:")
        print(f"   Total Images:    {tester.summary_stats['total_images']}")
        print(f"   Successful:      {tester.summary_stats['successful']}")
        print(f"   Failed:          {tester.summary_stats['failed']}")
        print(f"   DEGLARE Mode:    {tester.summary_stats['deglare_count']}")
        print(f"   DERAIN Mode:     {tester.summary_stats['derain_count']}")
        print(f"   Incorrect Mode:  {tester.summary_stats['incorrect_mode_count']}")
        
        if tester.summary_stats['glare_ran_derain'] > 0:
            print(f"\n⚠⚠⚠ CRITICAL ISSUE ⚠⚠⚠")
            print(f"   {tester.summary_stats['glare_ran_derain']} glare images incorrectly ran DERAIN!")
            print(f"   Check 'incorrect_modes.txt' for detailed analysis")
        else:
            print(f"\n✓✓✓ PERFECT PERFORMANCE ✓✓✓")
            print(f"   All glare images correctly ran DEGLARE mode!")
        
        print(f"\n📁 All reports saved to: {args.output_folder}/reports/")
        print("\n📋 Reports generated:")
        print("   • summary_report.txt      - Overall statistics")
        print("   • detailed_results.csv    - Full data for Excel analysis")
        print("   • incorrect_modes.txt     - Critical issues for thesis")
        print("   • performance_stats.txt   - Processing time analysis")
        print("   • visual_report.html      - Interactive visual report")
        
        print("\n💡 Next steps:")
        print("   1. Open visual_report.html in your browser")
        print("   2. Review incorrect_modes.txt for critical issues")
        print("   3. Import detailed_results.csv to Excel for charts")
        print("   4. Present summary_report.txt to your teacher")
        
        print("\n" + "="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR during batch testing:")
        print(f"  {str(e)}")
        traceback.print_exc()
        sys.exit(1)