import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
torch.distributed.init_process_group(backend='nccl',init_method='tcp://localhost:23456', world_size=1, rank=0)
# from torchlcnet import TorchLCNet
model=torch.load("/workspace/eye_net/single_eye_state/output/train/20240515-204305-vgnetg-128/checkpoint-131.pth")
print(model)
# print(model)
# model=model.module
# print(model)
# torch.save(model.module,'/workspace/eye_net/single_eye_state/model/vgnetg_1_0mp_se_hardswish_256linear_20240201_modifytriplet256_acc977_checkpoint_165.pth')
# print(model)
dummy_input = torch.randn(1, 3, 128, 128).cuda()
# dynamic_axes = {
#     'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}
#     #             # model.model.set_swish(memory_efficient=False)
# # torch.onnx.export(model, dummy_input, "/workspace/eye_net/single_eye_state/model/vgnetg_checkpoint-135_2batch.onnx" , verbose=False, opset_version=10,dynamic_axes=dynamic_axes)
torch.onnx.export(model, dummy_input, "/workspace/eye_net/single_eye_state/model/vgnet_20240515_acc971_checkpoint_131_batch1_opset9.onnx" , verbose=False, opset_version=9)