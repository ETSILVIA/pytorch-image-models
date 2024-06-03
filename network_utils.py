import torch
import torch.nn as nn
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet
from efficientnet_lite_pytorch import EfficientNet    #2021_7_12  change
from efficientnet_lite2_pytorch_model import EfficientnetLite2ModelFile



import torchvision
import torch.cuda.amp as amp
from torchlcnet import TorchLCNet
from efficientnet_lite import efficientnet_lite_params, build_efficientnet_lite


from effnet_lite import effnet_lite0, effnet_lite1, effnet_lite2, effnet_lite3, effnet_lite4

##
# version 1: use torch.autograd
class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        Usage is like this:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
            >>> loss = criteria(logits, lbs)
        '''

        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


##
# version 2: user derived grad computation
class FocalSigmoidLossFuncV2(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha

        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1. - probs) ** gamma

        ctx.vars = (coeff, probs, log_probs, log_1_probs, probs_gamma,
                probs_1_gamma, label, gamma)

        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        (coeff, probs, log_probs, log_1_probs, probs_gamma,
                probs_1_gamma, label, gamma) = ctx.vars

        term1 = (1. - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1. - probs) * log_1_probs).mul_(probs_gamma)

        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        return grads, None, None, None


class FocalLossV2(nn.Module):
    '''
    This use better formula to compute the gradient, which has better numeric stability
    Usage is like this:
        >>> criteria = FocalLossV2()
        >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
        >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
        >>> loss = criteria(logits, lbs)
    '''
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

# focal_loss func, L = -α(1-yi)**γ *ce_loss(xi,yi)
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss�?(1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class NormCost(object):

    def __init__(self, loss_kw, alpha=0.25, gamma=2):
        if loss_kw in ['normal', 'focal']:
            self.loss_kw = loss_kw
            if loss_kw == 'focal':
                self.alpha = alpha
                self.gamma = gamma
        else:
            raise KeyError

    def __call__(self, logits, levels):
        if self.loss_kw == 'normal':
            val = (-torch.sum((F.logsigmoid(logits)*levels + (F.logsigmoid(logits)-logits)*(1-levels)), dim=1))
            return torch.mean(val)
        else:
            total_sigmoid = torch.sigmoid(logits)
            one_logsigmoid = - torch.log(total_sigmoid+1e-7) * levels
            zero_logsigmoid = - (torch.log(1 - total_sigmoid+1e-7)) * (1 - levels)
            add_logsigmoid = (one_logsigmoid*((1-total_sigmoid)**self.gamma) + zero_logsigmoid*(total_sigmoid**self.gamma)) * self.alpha 
            return torch.mean(torch.sum(add_logsigmoid, dim=1))

class StnModule(nn.Module):

    def __init__(self, img_size):
        super(StnModule, self).__init__()
        self.img_size = img_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=img_size*img_size*3, out_features=1000),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(in_features=1000, out_features=20),
            nn.ReLU(True),
            nn.Linear(in_features=20, out_features=6),
        )
        bias = torch.Tensor([1, 0, 0, 0, 1, 0])

        nn.init.constant_(self.fc[5].weight, 0)
        self.fc[5].bias.data.copy_(bias)

    def forward(self, img):
        batch_size = img.size(0)
        theta = self.fc(img.view(batch_size, -1))
        theta[:, [0, 1, 3, 4]] = F.tanh(theta[:, [0, 1, 3, 4]])
        theta = theta.view(batch_size, 2, 3)

        grid = F.affine_grid(theta, torch.Size((batch_size, 3, self.img_size, self.img_size)))
        img_transform = F.grid_sample(img, grid)

        return img_transform

class MainModel(nn.Module):
    def __init__(self, backbone, num_classes, pretrain=False):
        super(MainModel, self).__init__()
        self.backbone = backbone
        if backbone == 'E':
            self.model = TorchLCNet(scale=1.0, class_num=81, dropout_prob=0.2, class_expand=1280)
            #self.model = effnet_lite3()
            #self.model = build_efficientnet_lite('efficientnet_lite4', num_classes)

            # if pretrain == True:
            #     weights_path = EfficientnetLite2ModelFile.get_model_file_path()
            #     self.model = EfficientNet.from_pretrained('efficientnet-lite2', weights_path=weights_path)
            # else:
            #     weights_path = EfficientnetLite2ModelFile.get_model_file_path()
            #     self.model = EfficientNet.from_pretrained('efficientnet-lite2', weights_path=weights_path)
            #     self.model = EfficientNet.from_name('efficientnet-lite2')

            # ##self.model.set_swish(False)   #delete
            # self.model._fc = nn.Linear(self.model._fc.in_features, num_classes - 1, bias=False)
            # self.last_bias = nn.Parameter(torch.zeros(num_classes - 1).float())
            # self.fc_gender = nn.Linear(self.model._fc.in_features, 2, bias=False)

            # efficient_lite3/efficient_lite4
            # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes - 1, bias=False)
            # self.last_bias = nn.Parameter(torch.zeros(num_classes - 1).float())
            # self.fc_gender = nn.Linear(self.model.fc.in_features, 2, bias=False)

            # pplcnet
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes-1, bias=False)
            self.last_bias = nn.Parameter(torch.zeros(num_classes-1).float())
            self.fc_gender = nn.Linear(self.model.fc.in_features, 2,  bias=False)

        # elif backbone == 'R':
        #     if pretrain == True:
        #         self.model = torchvision.models.resnet101(pretrained=True, num_classes=num_classes-1)
        #     else:
        #         self.model = torchvision.models.resnet101(num_classes=num_classes-1)
        # else:
        #     raise KeyError

    # 多分支
    def forward(self, x):
        x_1,x_2 = self.model(x)
        gender = self.fc_gender(x_2)
        #gender_value = torch.softmax(gender, dim=1)

        if self.backbone == 'E':
            age = x_1 + self.last_bias
            #age_value = torch.sigmoid(age)

        if True:  # pth_model to onnx_model, True
            # age_value = torch.sum(torch.sigmoid(age) > 0.5, dim=1) + 1  # not support Greater
            age_value = torch.sigmoid(age)
            gender_value = torch.softmax(gender, dim=1)
        return age_value, gender_value

        # return age, gender

    # # 单分支
    # def forward(self, x):
    #     x = self.model(x)
    #     #gender_value = torch.softmax(gender, dim=1)

    #     if self.backbone == 'E':
    #         age = x + self.last_bias
    #         # age_value = torch.sigmoid(age)

    #     if False:  # pth_model to onnx_model, True
    #         # age_value = torch.sum(torch.sigmoid(age) > 0.5, dim=1) + 1  # not support Greater
    #         age_value = torch.sigmoid(age)
    #         return age_value

    #     # return age,gender
    #     return age

