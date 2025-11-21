# run_derain.py - Wrapper for deraining
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run
from tools.test_model import test_model, init_args

if __name__ == '__main__':
    args = init_args()
    test_model(
        args.image_path,
        args.weights_path,
        args.label_path,
        args.output_file,
        args.config
    )