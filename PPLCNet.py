import torch.nn as nn
import torch


__all__ = ['lcnet_baseline']

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=num_groups,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels) # Here
        self.hardswish = nn.Hardswish()        # And Here

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x

class StemConv_simple(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=num_groups,
            bias=False)
        # self.bn = nn.BatchNorm2d(out_channels) # Here
        # self.hardswish = nn.Hardswish()        # And Here

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        # x = self.hardswish(x)
        return x

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        # self.hardsigmoid = nn.Hardsigmoid()
        self.hardswish = nn.Hardswish()
        self.upsample=nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.hardsigmoid(x)
        x = self.hardswish(x)
        x= self.upsample(x)
        out = identity * x
        # x2=torch.cat((x,x),2)
        
        # print("**************x",x.size())
        # print("**************identity",identity.size())
        return out

class DepthSepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dw_kernel_size, use_se=False, pw_kernel_size=1):
        super().__init__()
        self.use_se = use_se
        # self.dw_conv = StemConv(
        #     in_channels, in_channels, kernel_size=dw_kernel_size,
        #     stride=stride, num_groups=in_channels)
        self.dw_conv = StemConv_simple(
            in_channels, in_channels, kernel_size=dw_kernel_size,
            stride=stride, num_groups=in_channels)
        if self.use_se:
            self.se = SEModule(in_channels)
        self.pw_conv = StemConv(in_channels, out_channels, kernel_size=pw_kernel_size, stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x

class LCNet(nn.Module):
    def __init__(self, cfgs, block, num_classes=1000, dropout=0.2, scale=1.0, class_expand=1280):
        super(LCNet, self).__init__()
        self.cfgs = cfgs
        self.class_expand = class_expand
        self.block = block

        self.conv1 = StemConv(
            in_channels=3,
            kernel_size=3,
            out_channels=make_divisible(16 * scale),
            stride=2)
        stages = []
        for cfg in self.cfgs:
            layers = []
            for k, inplanes, planes, stride, use_se in cfg:
                in_channel = make_divisible(inplanes * scale)
                out_channel = make_divisible(planes * scale)
                layers.append(block(in_channel, out_channel, stride=stride, dw_kernel_size=k, use_se=use_se))

            stages.append(nn.Sequential(*layers))

        self.blocks = nn.Sequential(*stages)

        # out_channel = make_divisible(out_channel * scale)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(
            in_channels=out_channel,
            out_channels=self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.triple_eye=nn.Linear(self.class_expand,64)
        self.triple_glass=nn.Linear(self.class_expand,64)
        self.glass=nn.Linear(64,3)
        self.eye=nn.Linear(64,3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.avg_pool(x)
        x = self.last_conv(x)
        feature_maps = x
        x = self.hardswish(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x2=self.triple_glass(x)
        x3=self.triple_eye(x)
        triplets_glass = self.glass(x2)
        triplets_eye = self.eye(x3)
        return triplets_glass,triplets_eye,x2,x3
        # x = self.f(x)
        # return x

lcnet_cfgs = [
    # kernel, in_channels, out_channels, stride, use_se
    # stage1
    [[3, 16, 32, 1, False]],
    # stage2
    [[3, 32, 64, 2, False]],
    [[3, 64, 64, 1, False]],
    # stage3
    [[3, 64, 128, 2, False]],
    [[3, 128, 128, 1, False]],
    # stage4
    [[3, 128, 256, 2, False]],
    [[5, 256, 256, 1, False]],
    # stage5
    [[5, 256, 256, 1, False]],
    [[5, 256, 256, 1, False]],
    [[5, 256, 256, 1, False]],
    [[5, 256, 256, 1, False]],
    [[5, 256, 512, 2, True]],
    [[5, 512, 512, 1, True]]]
        
def lcnet_baseline(**kwargs):
    cfgs = lcnet_cfgs
    return LCNet(cfgs, DepthSepConvBlock, scale=0.2, **kwargs)

def pplcnet_x050_e512():
    cfgs = lcnet_cfgs
    return LCNet(cfgs, DepthSepConvBlock, num_classes=6, dropout=0.2, scale=0.5, class_expand=512)

def pplcnet_x075_e512():
    cfgs = lcnet_cfgs
    return LCNet(cfgs, DepthSepConvBlock, num_classes=6, dropout=0.2, scale=0.75, class_expand=512)

def pplcnet_x100_e512():
    cfgs = lcnet_cfgs
    return LCNet(cfgs, DepthSepConvBlock, num_classes=6, dropout=0.2, scale=1, class_expand=512)

def pplcnet_x050_e1k():
    cfgs = lcnet_cfgs
    return LCNet(cfgs, DepthSepConvBlock, num_classes=6, dropout=0.2, scale=0.5, class_expand=1024)

def pplcnet_x075_e1k():
    cfgs = lcnet_cfgs
    return LCNet(cfgs, DepthSepConvBlock, num_classes=6, dropout=0.2, scale=0.75, class_expand=1024)

def pplcnet_x100_e1k():
    cfgs = lcnet_cfgs
    return LCNet(cfgs, DepthSepConvBlock, num_classes=6, dropout=0.2, scale=1, class_expand=1024)

if __name__ == '__main__':
    from thop import profile
    # model=pplcnet_x075_e512()
    # dummy_input = torch.randn(1, 3, 128, 128)
    # flops, params = profile(model, inputs=(dummy_input,))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
#     import torch
    model = pplcnet_x075_e512()
    dummy_input = torch.randn(1, 3, 128, 128)
    model=model
    # print(model)
    out=model(dummy_input)
    print(out)
#     # model = pplcnet_x075_e1k()
#     model = pplcnet_x075_e512()
#     # model = pplcnet_x100_e512()
#     # model = pplcnet_x100_e1k()
#     # model = lcnet_baseline()
#     # sd = torch.load('model/pplcnet/20231027-141920-lcnetopt100/checkpoint-102.pth.tar', map_location='cpu')
#     model.load_state_dict(torch.load('model/pplcnet/20231027-141920-lcnetopt100/checkpoint-143.pth.tar', map_location='cpu')['state_dict'])
#     model.eval()
#     dummy_input = torch.randn(1, 3, 256, 256)
#     # torch.save(model, 'model_zoo/pplcnet0.5.pth.tar')
#     torch.onnx.export(
#         model, 
#         dummy_input, 
#         "model/pplcnet/20231027-141920-lcnetopt100/checkpoint-143.onnx" , 
#         verbose=False, 
#         opset_version=10
#         )
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import torch
# # torch.distributed.init_process_group(backend='nccl',init_method='tcp://localhost:23456', world_size=1, rank=0)
# # from torchlcnet import TorchLCNet
# model=pplcnet_x075_e512()
# model.load_state_dict(torch.load('/workspace/eye_net/single_eye_state/model/modify_moudle_checkpoint_149_center_loss_20240116.pth').state_dict())
# model=model.cuda()
# print(model)
# # print(model)
# # model=model.module
# # print(model)
# # torch.save(model.module,'/workspace/eye_net/single_eye_state/model/vgnetg_1_0mp_se_hardswish_256linear_20240201_modifytriplet256_acc977_checkpoint_165.pth')
# # print(model)
# dummy_input = torch.randn(1, 3, 128, 128).cuda()
# dynamic_axes = {
#     'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}
# #     #             # model.model.set_swish(memory_efficient=False)
# # # torch.onnx.export(model, dummy_input, "/workspace/eye_net/single_eye_state/model/vgnetg_checkpoint-135_2batch.onnx" , verbose=False, opset_version=10,dynamic_axes=dynamic_axes)
# torch.onnx.export(model, dummy_input, "/workspace/eye_net/single_eye_state/model/checkpoint_149_center_loss_20240116.onnx" , verbose=False, opset_version=9)