import sys
sys.path.append('.')

import os.path as ops
import argparse
import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from attentive_gan_model import derain_drop_net
from config import global_config

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The input image path')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--label_path', type=str, default=None, help='The label image path')
    parser.add_argument('--output_file', type=str, default='results.txt', help='Output text file for results')

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def visualize_attention_map(attention_map):
    """
    The attention map is a matrix ranging from 0 to 1, where the greater the value,
    the greater attention it suggests
    :param attention_map:
    :return:
    """
    attention_map_color = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1], 3],
        dtype=np.uint8
    )

    red_color_map = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1]],
        dtype=np.uint8) + 255
    red_color_map = red_color_map * attention_map
    red_color_map = np.array(red_color_map, dtype=np.uint8)

    attention_map_color[:, :, 2] = red_color_map

    return attention_map_color


def save_results_to_file(image_path, weights_path, label_path, output_image, atte_maps, 
                        ssim_val=None, psnr_val=None, output_file='results.txt'):
    """
    Save results to a text file in matrix/table format
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 60 + "\n")
        f.write("DERAIN MODEL TEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Image: {image_path}\n")
        f.write(f"Weights Path: {weights_path}\n")
        f.write(f"Label Path: {label_path if label_path else 'None'}\n")
        f.write("-" * 60 + "\n\n")
        
        # Performance Metrics Table
        if ssim_val is not None and psnr_val is not None:
            f.write("PERFORMANCE METRICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Metric':<15} {'Value':<15} {'Range':<15} {'Quality':<15}\n")
            f.write("=" * 60 + "\n")
            
            # SSIM interpretation
            ssim_quality = "Excellent" if ssim_val > 0.9 else "Good" if ssim_val > 0.8 else "Fair" if ssim_val > 0.7 else "Poor"
            f.write(f"{'SSIM':<15} {ssim_val:<15.5f} {'[0 - 1]':<15} {ssim_quality:<15}\n")
            
            # PSNR interpretation
            psnr_quality = "Excellent" if psnr_val > 40 else "Good" if psnr_val > 30 else "Fair" if psnr_val > 20 else "Poor"
            f.write(f"{'PSNR (dB)':<15} {psnr_val:<15.5f} {'[0 - inf]':<15} {psnr_quality:<15}\n")
            
            f.write("=" * 60 + "\n")
            f.write("Quality Ratings:\n")
            f.write("  SSIM: >0.9=Excellent, >0.8=Good, >0.7=Fair, ≤0.7=Poor\n")
            f.write("  PSNR: >40dB=Excellent, >30dB=Good, >20dB=Fair, ≤20dB=Poor\n")
            f.write("-" * 60 + "\n\n")
        else:
            f.write("PERFORMANCE METRICS\n")
            f.write("=" * 60 + "\n")
            f.write("No label image provided - SSIM and PSNR metrics unavailable\n")
            f.write("To get performance metrics, provide --label_path parameter\n")
            f.write("-" * 60 + "\n\n")
        
        # Output Image Statistics
        f.write("OUTPUT IMAGE ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Property':<20} {'Red Channel':<15} {'Green Channel':<15} {'Blue Channel':<15}\n")
        f.write("=" * 80 + "\n")
        
        # Per-channel statistics
        red_stats = output_image[:, :, 0]
        green_stats = output_image[:, :, 1]
        blue_stats = output_image[:, :, 2]
        
        f.write(f"{'Min Value':<20} {np.min(red_stats):<15.2f} {np.min(green_stats):<15.2f} {np.min(blue_stats):<15.2f}\n")
        f.write(f"{'Max Value':<20} {np.max(red_stats):<15.2f} {np.max(green_stats):<15.2f} {np.max(blue_stats):<15.2f}\n")
        f.write(f"{'Mean Value':<20} {np.mean(red_stats):<15.2f} {np.mean(green_stats):<15.2f} {np.mean(blue_stats):<15.2f}\n")
        f.write(f"{'Std Deviation':<20} {np.std(red_stats):<15.2f} {np.std(green_stats):<15.2f} {np.std(blue_stats):<15.2f}\n")
        f.write(f"{'Median':<20} {np.median(red_stats):<15.2f} {np.median(green_stats):<15.2f} {np.median(blue_stats):<15.2f}\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Overall Image Shape: {output_image.shape}\n")
        f.write(f"Overall Mean: {np.mean(output_image):.2f}\n")
        f.write(f"Overall Std: {np.std(output_image):.2f}\n")
        f.write("-" * 60 + "\n\n")
        
        # Attention Maps Statistics
        f.write("ATTENTION MAPS ANALYSIS\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Map':<8} {'Shape':<15} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8} {'Focus':<12} {'Coverage':<12}\n")
        f.write("=" * 90 + "\n")
        
        for i, att_map in enumerate(atte_maps):
            map_data = att_map[0, :, :, 0]  # Extract the first batch and channel
            
            # Calculate additional metrics
            focus_score = np.max(map_data) - np.mean(map_data)  # How focused the attention is
            coverage = np.sum(map_data > 0.5) / map_data.size * 100  # Percentage of high attention areas
            
            f.write(f"{'Map_' + str(i+1):<8} {str(map_data.shape):<15} "
                   f"{np.min(map_data):<8.3f} {np.max(map_data):<8.3f} "
                   f"{np.mean(map_data):<8.3f} {np.std(map_data):<8.3f} "
                   f"{focus_score:<12.3f} {coverage:<12.1f}%\n")
        
        f.write("=" * 90 + "\n")
        f.write("Focus Score: Higher values indicate more focused attention\n")
        f.write("Coverage: Percentage of areas with attention > 0.5\n")
        f.write("-" * 60 + "\n\n")
        
        # Attention Maps Value Matrix (Sample)
        f.write("ATTENTION MAPS SAMPLE VALUES (Top-left 10x10 region)\n")
        f.write("=" * 60 + "\n")
        
        for i, att_map in enumerate(atte_maps):
            map_data = att_map[0, :, :, 0]
            f.write(f"\nAttention Map {i+1}:\n")
            f.write("-" * 30 + "\n")
            
            # Show top-left 10x10 region
            sample_size = min(10, map_data.shape[0], map_data.shape[1])
            sample_data = map_data[:sample_size, :sample_size]
            
            for row in sample_data:
                f.write(" ".join([f"{val:6.3f}" for val in row]) + "\n")
            f.write("\n")
        
        # Output Image Sample Values
        f.write("OUTPUT IMAGE SAMPLE VALUES (Top-left 10x10 region, RGB channels)\n")
        f.write("=" * 60 + "\n")
        
        sample_size = min(10, output_image.shape[0], output_image.shape[1])
        
        for channel in range(3):
            channel_names = ['Red', 'Green', 'Blue']
            f.write(f"\n{channel_names[channel]} Channel:\n")
            f.write("-" * 30 + "\n")
            
            sample_data = output_image[:sample_size, :sample_size, channel]
            for row in sample_data:
                f.write(" ".join([f"{val:3d}" for val in row]) + "\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("Results saved successfully!\n")
        f.write("Generated files: src_img.png, derain_ret.png, atte_map_*.png\n")
        f.write("=" * 60 + "\n")


def test_model(image_path, weights_path, label_path=None, output_file='results.txt'):
    """

    :param image_path:
    :param weights_path:
    :param label_path:
    :param output_file:
    :return:
    """
    assert ops.exists(image_path)

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TEST.BATCH_SIZE, CFG.TEST.IMG_HEIGHT, CFG.TEST.IMG_WIDTH, 3],
                                  name='input_tensor'
                                  )

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    image_vis = image
    image = np.divide(np.array(image, np.float32), 127.5) - 1.0

    label_image_vis = None
    if label_path is not None:
        label_image = cv2.imread(label_path, cv2.IMREAD_COLOR)
        label_image_vis = cv2.resize(
            label_image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )

    phase = tf.constant('test', tf.string)

    net = derain_drop_net.DeRainNet(phase=phase)
    output, attention_maps = net.inference(input_tensor=input_tensor, name='derain_net')

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        output_image, atte_maps = sess.run(
            [output, attention_maps],
            feed_dict={input_tensor: np.expand_dims(image, 0)})

        output_image = output_image[0]
        for i in range(output_image.shape[2]):
            output_image[:, :, i] = minmax_scale(output_image[:, :, i])

        output_image = np.array(output_image, np.uint8)

        ssim_val = None
        psnr_val = None
        
        if label_path is not None:
            label_image_vis_gray = cv2.cvtColor(label_image_vis, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
            output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
            psnr_val = compare_psnr(label_image_vis_gray, output_image_gray)
            ssim_val = compare_ssim(label_image_vis_gray, output_image_gray)

            print('SSIM: {:.5f}'.format(ssim_val))
            print('PSNR: {:.5f}'.format(psnr_val))

        # Save results to text file
        save_results_to_file(image_path, weights_path, label_path, output_image, 
                           atte_maps, ssim_val, psnr_val, output_file)
        
        print(f"Results saved to: {output_file}")

        # 保存并可视化结果
        cv2.imwrite('src_img.png', image_vis)
        cv2.imwrite('derain_ret.png', output_image)

        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('derain_ret')
        plt.imshow(output_image[:, :, (2, 1, 0)])
        plt.figure('atte_map_1')
        plt.imshow(atte_maps[0][0, :, :, 0], cmap='jet')
        plt.savefig('atte_map_1.png')
        plt.figure('atte_map_2')
        plt.imshow(atte_maps[1][0, :, :, 0], cmap='jet')
        plt.savefig('atte_map_2.png')
        plt.figure('atte_map_3')
        plt.imshow(atte_maps[2][0, :, :, 0], cmap='jet')
        plt.savefig('atte_map_3.png')
        plt.figure('atte_map_4')
        plt.imshow(atte_maps[3][0, :, :, 0], cmap='jet')
        plt.savefig('atte_map_4.png')
        plt.show()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # test model
    test_model(args.image_path, args.weights_path, args.label_path, args.output_file)