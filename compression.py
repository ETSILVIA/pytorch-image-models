import logging
import os
import shutil
import time
# from thop import profile
from collections import OrderedDict
from contextlib import suppress
import torch
from qua_mobilenetv3 import mobilenet_v3_large 
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from Triplet_dataset import TinyImageNetDataset_Triplet,glass_Dataset_Triplet
from network_utils_whb import focal_loss,MultiClassFocalLossWithAlpha
import yaml
import gc
import torch.functional as F
from copy import deepcopy
import numpy as np
from Triplet_model import mbn ,tri_mbnv3,incept
from torchvision import models
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torchvision.transforms import transforms
from timm import utils
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, convert_sync_batchnorm, model_parameters, set_fast_norm
from copy import deepcopy
from nova_slim.pruning.one_shot import L1FilterPruner, L2FilterPruner
from nova_slim.utils.compress_utils import execute_trace
from nova_slim.utils.infer_registry import *
from nova_slim.speedup import inference
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

logging.basicConfig(level=logging.DEBUG)

prune_config = {
    'l2filter': {
        'dataset_name': 'eye_state',
        'model_name': 'torchlcnet',
        'pruner_class': L2FilterPruner,
        'input_shape': [1, 3, 64, 64],
        'prune_type': 'l2',
        'config_list': [],
    }
}

def get_data_loaders(batch_size=128):
    train_transform = transforms.Compose([transforms.Resize((64,64)),
                                          # transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          #    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                          ])

    val_transform = transforms.Compose([transforms.Resize((64, 64)),

                                        transforms.ToTensor(),
                                        ])

    

    train_data= pd.read_csv('label/train_single_eye_20231113.csv')
    val_data= pd.read_csv('label/val_single_eye_20231114_new.csv')
    train_data_path = '/data/cifs/f/Dataset/2022_8_23_data_label/wang_hongbo/20230220/process/eye/single_eye/train/'
    val_data_path = '/data/cifs/f/Dataset/2022_8_23_data_label/wang_hongbo/20230220/process/eye/single_eye/val/'
  
    dataset_train =  glass_Dataset_Triplet(train_data,train_data_path,transform=train_transform)
    dataset_eval=  glass_Dataset_Triplet(val_data,val_data_path,transform=val_transform)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def train(model, train_loader):
    model.train()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    for batch_idx, (data, target) in enumerate(train_loader):
        glass = target[0]
        eye = target[1]
        data, glass,eye = data.cuda(), glass.cuda(),eye.cuda()
        optimizer.zero_grad()
        output = model(data)
        x_glass = output[0]
        x_eye = output[1]

        glass_loss = F.cross_entropy(x_glass, glass)
        eye_loss = F.cross_entropy(x_eye, eye)
        loss=glass_loss+eye_loss
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logging.info('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))
    return model

# def test(model, test_loader, **kwargs):
#     print(kwargs, "nothing happened to these args.")
#     model.eval()
#     model.cuda()
#     test_loss = 0
#     glass_correct = 0
#     eye_correct = 0
#     # acc = 0
#     iter = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             glass = target[0]
#             eye = target[1]
#             data, glass,eye= data.cuda(), glass.cuda(),eye.cuda()

#             output = model(data)
#             x_glass = output[0]
#             x_eye = output[1]
#             glass_loss = F.cross_entropy(x_glass, glass)
#             eye_loss = F.cross_entropy(x_eye, eye)
#             loss = glass_loss + eye_loss
#             test_loss+=loss.item()
#             # test_loss += F.cross_entropy(output, target, reduction='sum').item()
#             glass_pred = x_glass.argmax(dim=1, keepdim=True)
#             eye_pred = x_eye.argmax(dim=1, keepdim=True)
#             glass_correct += glass_pred.eq(glass.view_as(glass_pred)).sum().item()
#             eye_correct += eye_pred.eq(eye.view_as(eye_pred)).sum().item()

#             # correct += pred.eq(target.view_as(pred)).sum().item()
#             logging.info('Iteration {}/{}'.format(iter, int(len(test_loader.dataset) / args.batch_size)))
#             iter += 1
#     test_loss /= len(test_loader.dataset)
#     glass_acc = glass_correct / len(test_loader.dataset)
#     eye_acc = eye_correct / len(test_loader.dataset)
#     acc=(glass_acc+eye_acc)/2
#     logging.info('Loss: {}  Accuracy: {})\n'.format(test_loss, acc))
#     return acc
def test(model, loader):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    glass_top1_m = utils.AverageMeter()
    eye_top1_m = utils.AverageMeter()
    loss_fn = nn.CrossEntropyLoss().cuda()
   
    model.cuda()
    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
       
        for batch_idx, (input, target) in enumerate(loader):
         
            glass=target[0]
       
            eye=target[1]   
            
            last_batch = batch_idx == last_idx
            
            input = input.cuda()
            glass=glass.cuda()
        
            eye=eye.cuda()
   
            output = model(input)
            x_glass=output[0]
            x_eye=output[1] 

            glass_loss=loss_fn(x_glass, glass)
            eye_loss=loss_fn(x_eye,eye)
            loss= glass_loss + eye_loss
         
            glass_acc1, glass_acc5 = utils.accuracy(x_glass, glass, topk=(1, 5))
      
            eye_acc1, eye_acc5 = utils.accuracy(x_eye, eye, topk=(1, 5))
        
            reduced_loss = loss.data
            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))

            glass_top1_m.update(glass_acc1.item(), x_glass.size(0))
            eye_top1_m.update(eye_acc1.item(), x_eye.size(0))
            

    logging.info('Loss: {}  glass_Accuracy: {} eye_Accuracy: {})\n'.format(losses_m.avg,glass_top1_m.avg, eye_top1_m.avg))
    return (glass_top1_m.avg+eye_top1_m.avg)/200

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    model_name = prune_config[args.pruner_name]['model_name']
    dataset_name = prune_config[args.pruner_name]['dataset_name']
    train_loader, test_loader = get_data_loaders(args.batch_size)

    # model = MobileNetV3_Small(6).cuda()
    # mbn_model = models.mobilenet_v3_small()
    # model = tri_mbnv3(mbn_model)
    # model1=create_model('inception_v3')
    # model=incept(model1)
    # print(model)
    # model=mobilenet_v3_large()
    # total=sum([param.nelement() for param in model.parameters()])
    # print('total params:',sum([param.nelement() for param in model.parameters()]))
    model=torch.load('./output/train/20231115-141438-torchlcnet_normalize_size64-64/checkpoint-223.pth')
    # for i in range(10):
    #     train(model,train_loader)
    #     test(model,test_loader)
    # torch.load('compression/20221105_124630/checkpoints/pruned_mobilenetv3_small_100_eye_state_l2filter.pth')
    # for i in range(10):
    #     train(model,train_loader)
    #     test(model,test_loader)
   
            
    # def convert_hard_sigmoid(model):
    #     for child_name, child in model.named_children():
    #         if isinstance(child, nn.Hardsigmoid):
    #             setattr(model, child_name, nn.Sigmoid)
    #         else:
    #             convert_hard_sigmoid(child)
                
    # convert_hard_sigmoid(model)
    # print(model)
    # model.cuda()

    logging.info('Step 1: training or resume from pre-trained model.')
    if args.resume_from is not None and os.path.exists(args.resume_from):
        logging.info('loading checkpoint {} ...'.format(args.resume_from))
        # model.load_state_dict(torch.load(args.resume_from)['state_dict'])
        # total=sum([param.nelement() for param in model.parameters()])
        # print("number of parameter: %.2fM" %(total/1e6))
        # resume_checkpoint(model,
        #                   '/workspace/timm_classifer/output/train/20221030-131606-mobilenetv3_small_100-128/model_best.pth.tar')
        # test(model, test_loader)
    else:
        # STEP.1 Train from scratch 从头开始训练
        if args.multi_gpu and torch.cuda.device_count():
            model = nn.DataParallel(model)
        logging.info('start training')
        pretrain_model_path = os.path.join(
            args.checkpoints_dir, 'pretrain_{}_{}_{}.pth'.format(model_name, dataset_name, args.pruner_name))
        for epoch in range(args.pretrain_epochs):
            train(model, train_loader)
            test(model, test_loader)
        torch.save(model.state_dict(), pretrain_model_path)

    # STEP.2 Automatic compression
    logging.info('Step 2: Automatic compression.')
    logging.info('start model pruning...')
    # pruner needs to be initialized from a model not wrapped by DataParallel
    if isinstance(model, nn.DataParallel):
        logging.info('Model is instance of DataParallel.')
        model = model.module

    # STEP.2.1 Sensitivity analysis
    logging.info('Step 2.1: Sensitivity analysis.')
    outdir = 'sens_analysis/2023_11_16/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    csv_file_path = os.path.join(outdir, 'sens.csv')
    
    exclude_layers = []
    # exclude_layers : (list)包含不需要执⾏敏感度分析的卷积层名字，这些卷积层不会被压缩
    if os.path.exists(csv_file_path):
        logging.info('Sensitivity CSV file exists, resume from {}'.format(csv_file_path))
        from nova_slim.utils.sensitivity_analysis import load_csv
        sensitivity = load_csv(csv_file_path)
    else:
        logging.info('Will generate {}, then start sensitivity analysis.'.format(csv_file_path))
        from nova_slim.utils.sensitivity_analysis import SensitivityAnalysis
        # naive model
        # When the early_stop_mode is 'dropped', the values of early_stop_value is equal to metric_thres.
        s_analyzer = SensitivityAnalysis(model=model, val_func=test, val_loader=test_loader,
                                         sparsities_dict=None, prune_type='l2', apply_async=True,
                                         single_process_mode=False, result_dir=outdir,
                                         early_stop_mode='dropped', early_stop_value=0.05)
        sensitivity = s_analyzer.analysis(exclude_layers=exclude_layers)
    logging.info("Sensitivity analysis finished.")
    
    # '''
    # STEP.2.2 Compress
    logging.info('Step 2.2: Compress model by sensitivity.')
 
    from nova_slim.pruning.apply_compression import compress
    dummy_input = [torch.randn(prune_config[args.pruner_name]['input_shape']).to("cpu")]
    model = model.to("cpu")
    with torch.onnx.select_model_mode_for_export(model, False):
        torch._C._jit_set_inline_everything_mode(True)
        trace = torch.jit.trace(model, dummy_input)
        torch._C._jit_pass_inline(trace.graph)
    # trace = execute_trace(model, dummy_input, forward_name="forward")
    print(trace)
    # exclude_layers = ['blocks.2.0.se.conv_expand', 'blocks.2.0.conv_pwl', 'blocks.2.2.se.conv_expand', \
        # 'blocks.3.0.se.conv_expand', 'blocks.4.0.conv_pwl']


    # methods = {
    #     "infer_from_inshape": inference.batchnorm2d_inshape,
    # }
    #
    # methods = {k: v for k, v in methods.items() if v is not None}
    # Register.regist_module_cls(nn.BatchNorm1d, **methods)
    model, fixed_mask = compress(model, dummy_input, prune_config[args.pruner_name]['pruner_class'],
                                 ori_metric=0.99, metric_thres=0.6, sensitivity=sensitivity, trace=trace,
                                 exclude_layers=exclude_layers,channel_alignment = 8)

    pruned_model_path = os.path.join(args.checkpoints_dir,
                                     'pruned_{}_{}_{}.pth'.format(model_name, dataset_name, args.pruner_name))
    mask_path = os.path.join(args.checkpoints_dir,
                             'mask_{}_{}_{}.pth'.format(model_name, dataset_name, args.pruner_name))
    
    torch.save(fixed_mask, mask_path)
    
    torch.save(model, pruned_model_path)
    
    torch.save(model.state_dict(),'1.pth')
    
    from thop import profile
    
    model.eval()
    model.cuda()
    # if args.export_onnx :
        
    #             dummy_input = torch.randn(1, 3, 128, 128).to('cuda')
    #             # model.model.set_swish(memory_efficient=False)
    #             torch.onnx.export(model, dummy_input, "compression/20221027_203038/checkpoints/eye_statenet-lite2_2_1.onnx" , verbose=False, opset_version=10)
                
    _forward = model.forward
    model.forward = getattr(model, "forward", model.forward)
    macs, params = profile(deepcopy(model).to('cpu'), inputs=dummy_input, verbose=False)
    model.forward = _forward
    logging.info(
        "MACs and Params after compression: MACs: {} G, Params: {} M".format(macs / 1000000000, params / 1000000))
    model = model.to(device)
    
'''
    # STEP.3 Finetune
    logging.info('start finetuning')
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    for epoch in range(args.finetune_epochs):
        # pruner.update_epoch(epoch)
        logging.info('# Epoch {} #'.format(epoch))
        train(model, train_loader)
        test(model, test_loader)
    # 微调之后不保存模型？？？
'''

if __name__ == '__main__':
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser()
    process_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # process_time = "20221103_165848"
    if not os.path.exists(os.path.join("compression", process_time)):
        os.makedirs(os.path.join("compression", process_time))
        
    parser.add_argument("--process_time", type=str, default=process_time, help="format process time")
    parser.add_argument("--pruner_name", type=str, default="l2filter", help="pruner name")
    parser.add_argument("--export_onnx", type=str, default=True, help="if transport onnx")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pretrain_epochs", type=int, default=1, help="training epochs before model pruning")
    parser.add_argument("--finetune_epochs", type=int, default=4, help="finetuning epochs after model pruning")
    parser.add_argument("--checkpoints_dir", type=str, default="compression/{}/checkpoints".format(process_time), help="checkpoints directory")
    parser.add_argument("--resume_from", type=str, default='./output/train/20231109-105801-mobilenetv2_qua-64/checkpoint-35.pth', help="pretrained model weights")
    parser.add_argument("--multi_gpu", default=False, help="Use multiple GPUs for training")

    args = parser.parse_args()
    main(args)
    
# 2022/11/9以前:output/train/20221105-094901-mobilenetv3_small_100-128/model_best.pth.tar