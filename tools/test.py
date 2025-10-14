import os
import subprocess

subprocess.run([
    'python', 'tools/test.py',
    'local_configs/armformer_config.py',
    'work_dirs/best_mIoU_iter_160000.pth',
    '--out', 'results/test_images',
    '--work-dir', 'work_dirs'
], check=True)

subprocess.run([
    'python', 'tools/analysis_tools/benchmark.py',
    'local_configs/armformer_config.py',
    'work_dirs/best_mIoU_iter_160000.pth'
], check=True)

subprocess.run([
    'python', 'tools/analysis_tools/get_flops.py',
    'local_configs/armformer_config.py',
    '--shape', '512', '512'
], check=True)

