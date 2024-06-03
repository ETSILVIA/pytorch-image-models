#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
from email.policy import default
import os
import csv
import glob
import json
import time
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress
from Triplet_dataset import glass_Dataset_Triplet
from my_dataset import MyDataset
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models, set_fast_norm
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser,\
    decay_batch_step, check_batch_size_retry
from my_eval_index import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from torchvision import models
from Triplet_model import *
from torchlcnet_whb import TorchLCNet
from vgnet import *
has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')
torch.distributed.init_process_group(backend='nccl',init_method='tcp://localhost:23400', world_size=1, rank=0)

import torch.distributed as dist
# dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)


# checkpoint:output/train/20221020-191206-mobilenetv3_small_100-128/model_best.pth.tar
#./output/train/20221121-155758-mobilenetv3_small_100-128/checkpoint-192.pth.tar 
# ./output/train/20221124-115053-mobilenetv3_small_100-128/checkpoint-89.pth.tar结果最好
parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
# parser.add_argument('data', metavar='DIR',default='/dataset',
#                     help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='ImageFolder',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='mobilenetv3_small_100',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=(3,128,128), nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='./output/train/20231026-132748-mobilenetv3_small_large-128/checkpoint-15.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                    help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    # if args.checkpoint:
        
    # model=torch.load('/workspace/eye_net/single_eye_state/model/VGNET_hardswish_256linear_20240201_acc9789_checkpoint_81.pth')
    model=torch.load('/workspace/eye_net/single_eye_state/output/train/20240515-204305-vgnetg-128/checkpoint-131.pth')
    print(model)
    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)
    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()
    
    
    
    val_data= pd.read_csv('/workspace/eye_net/single_eye_state/label/single_eye_val_20240513_new.csv')
    
    data_path = '/data/cifs/f/Dataset/eyes_state/00_Singleye_Dataset_0513/val/'
    dataset= glass_Dataset_Triplet(val_data,data_path)
   

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    glass_loss_m = AverageMeter()
    eye_loss_m = AverageMeter()
    glass_top1 = AverageMeter()
    eye_top1 = AverageMeter()
    
    
    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)
        pred_eyes=[]
        pred_glasses=[]
        label_glass=[]
        label_eye=[]
        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):   
            glass=target[:,0]
            eye=target[:,1] 
            label_eye.append(eye.tolist())
            label_glass.append(glass.tolist())
            if args.no_prefetcher:
                
                glass=glass.cuda()
                eye=eye.cuda()
                input = input.cuda()
                label_eye.append(eye.tolist())
                label_glass.append(glass.tolist())
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)
                output_glass=output[0]
                output_eye=output[1]
                glass_score, pred_glass = torch.max(output_glass, 1)
                eye_score, pred_eye = torch.max(output_eye, 1)
                pred_glasses.append(pred_glass.tolist())
                pred_eyes.append(pred_eye.tolist())

           
                
            glass_loss=criterion(output_glass, glass)
            eye_loss=criterion(output_eye, eye)
        

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            glass_acc1, _ = accuracy(output_glass.detach(), glass, topk=(1, 5))
            eye_acc1, _ = accuracy(output_eye.detach(), eye, topk=(1, 5))
            glass_loss_m.update(glass_loss.item(),input.size(0))
            eye_loss_m.update(eye_loss.item(),input.size(0))
            glass_top1.update(glass_acc1.item(), input.size(0))
            eye_top1.update(eye_acc1.item(), input.size(0))
       
            glass_confusion =ConfusionMatrix(num_classes=3, labels=['no_glass','glass','sunglass'], true_labels=glass.cpu(),preds =pred_glass.cpu())
            
          
            glass_confusion.summary() 
            glass_confusion.sum_index()
            eye_confusion = ConfusionMatrix(num_classes=3, labels=['open','close','invisible'], true_labels=eye.cpu(),preds=pred_eye.cpu())
      
            eye_confusion.summary() 
            eye_confusion.sum_index()
      
            batch_time.update(time.time() - end)
            end = time.time()
        
            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss_glass: {Loss_glass.val:>7.4f} ({Loss_glass.avg:>6.4f}) '
                    'Loss_eye: {Loss_eye.val:>7.4f} ({Loss_eye.avg:>6.4f}) '
                    'glass_Acc@1: {glass_top1.val:>7.3f} ({glass_top1.avg:>7.3f}) '
                    'eye_Acc@1: {eye_top1.val:>7.3f} ({eye_top1.avg:>7.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        Loss_glass=glass_loss_m,Loss_eye=eye_loss_m,glass_top1=glass_top1, eye_top1=eye_top1))
                
   
    eye_confusion1 = ConfusionMatrix(num_classes=3, labels=['open','close','invisible'],preds=sum(pred_eyes,[]), true_labels=sum(label_eye,[]))
            # eye_confusion.plot()
    eye_confusion1.summary() 
    eye_confusion1.sum_index()
    glass_confusion1 = ConfusionMatrix(num_classes=3, labels=['no_glass','glass','sunglass'],preds=sum(pred_glasses,[]), true_labels=sum(label_glass,[]))
            # eye_confusion.plot()
    glass_confusion1.summary() 
    glass_confusion1.sum_index()
    if real_labels is not None:
        # real labels mode replaces topk values at the end
        # top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
        glass_top1a, _ = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
        eye_top1a, _ = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        # top1a, top5a = top1.avg, top5.avg
        glass_top1a, eye_top1a = glass_top1.avg, eye_top1.avg
    results = OrderedDict(
        model=args.model,
        glass_top1=round(glass_top1a, 4), glass_top1_err=round(100 - glass_top1a, 4),
        eye_top1=round(eye_top1a, 4), eye_top1_err=round(100 - eye_top1a, 4),
        # top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        crop_pct=crop_pct,
        interpolation=data_config['interpolation'])

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['glass_top1'], results['eye_top1'], results['glass_top1_err'], results['eye_top1_err']))

    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = 'Unknown'
    while batch_size:
        args.batch_size = batch_size * args.num_gpu  # multiply by num-gpu for DataParallel case
        try:
            torch.cuda.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            _logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        _logger.warning(f'Reducing batch size to {batch_size} for retry.')
    results['error'] = error_str
    _logger.error(f'{args.model} failed to validate ({error_str}).')
    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k', '*_dino'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            initial_batch_size = args.batch_size
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                r = _try_run(args, initial_batch_size)
                if 'error' in r:
                    continue
                if args.checkpoint:
                    r['checkpoint'] = args.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        if args.retry:
            results = _try_run(args, args.batch_size)
        else:
            results = validate(args)
    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
