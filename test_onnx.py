# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import torch
# from PPLCNet_whb import *
# # torch.distributed.init_process_group(backend='nccl',init_method='tcp://localhost:23456', world_size=1, rank=0)
# # from torchlcnet import TorchLCNet
# model=torch.load('/workspace/eye_net/single_eye_state/20231213_lr0.0001_model_epoch211_acc0.967.pth')
# print(model)
# dummy_input = torch.randn(1, 3, 64, 64).cuda()
# dynamic_axes = {
#     'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}
#     #             # model.model.set_swish(memory_efficient=False)
# torch.onnx.export(model, dummy_input, "20231213_lr0.0001_model_epoch211_acc0.967.onnx" , verbose=False, opset_version=10,dynamic_axes=dynamic_axes)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
# torch.distributed.init_process_group(backend='nccl',init_method='tcp://localhost:23456', world_size=1, rank=0)
# from torchlcnet import TorchLCNet
model=torch.load('/workspace/eye_net/single_eye_state/output/train/20231218-203811-pplcnet_normalize_size64-128/checkpoint-85.pth')
# torch.save(model.moudle,'model/modify__moudle_triplet_checkpoint-207.pth')
# print(model)
dummy_input = torch.randn(2, 3, 128, 128).cuda()
dynamic_axes = {
    'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}
    #             # model.model.set_swish(memory_efficient=False)
torch.onnx.export(model, dummy_input, "modified_single_eye_pplcnet_focal_triplet_size128_epoch85_acc975_20231219_batch2.onnx" , verbose=False, opset_version=10,dynamic_axes=dynamic_axes)
# torch.onnx.export(model, dummy_input, "dbht_mbnv3_large_epoch_110_no_normalize_top957_20230630.onnx" , verbose=False, opset_version=10)