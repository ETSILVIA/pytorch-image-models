# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python test_yolov5_single_prune.py
"""

import os
import sys
from pathlib import Path
import yaml

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_file, check_yaml, colorstr

from npruner.utils.compress_utils import compression


def test_yolov5_coco_single_prune():
    with open('data/hyps/hyp.scratch.yaml', errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    model = attempt_load('./weight/yolov5s6.pt', map_location='cpu')
    model.training = True

    test_path = '/home/SharedDatasets/coco/images/val2017'
    testloader = create_dataloader(path=test_path, imgsz=640, batch_size=16, stride=64, single_cls=False,
                                   hyp=hyp, cache=False, rect=False, rank=-1,
                                   workers=8, pad=0.5, image_weights=True,
                                   prefix=colorstr('val: '))[0]

    # STEP.2 Automatic Prune
    print('start model pruning...')

    checkpoints_dir = './checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    dummy_input = [torch.randn([1, 12, 320, 320]).to("cpu")]
    exclude_layers = ['model.33.m.0', 'model.33.m.1', 'model.33.m.2', 'model.33.m.3']

    model = compression(model=model, val_func=val.run, val_loader=testloader, dummy_input=dummy_input, ori_metric=0.367,
                        metric_thres=0.03, exclude_layers=exclude_layers, single_process_mode=False)

    pruned_model_path = os.path.join(checkpoints_dir,
                                     'pruned_{}_{}_{}.pth'.format('yolov5', 'coco', 'l2'))
    torch.save(model, pruned_model_path)
    from thop import profile
    macs, params = profile(model, inputs=dummy_input, verbose=False)
    print("MACs: {} G, Params: {} M".format(macs / 1000000000, params / 1000000))


if __name__ == "__main__":
    test_yolov5_coco_single_prune()
