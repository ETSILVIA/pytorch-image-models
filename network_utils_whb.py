import torch
import torch.nn as nn
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet
import torchvision



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


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
   
class TripletLoss1(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss1, self).__init__()
        self.margin = margin
    def calc_euclidean(self, x1, x2):
        return (x1- x2).pow(2).sum(1)
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu (distance_positive-distance_negative + self.margin)
        return losses.mean()
    
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=1.5, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        alpha = alpha.to(pred.device)
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
    
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
        alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
class hardTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):#三元组的阈值margin
        super(hardTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)#三元组损失函数
        #ap an margin y:倍率   Relu(ap - anxy + margin)这个relu就起到和0比较的作用

    def forward(self, inputs, targets):
        """
        Args:
            inputs: visualization_feature_map matrix with shape (batch_size, feat_dim)#32x2048
            targets: ground truth labels with shape (num_classes)#tensor([32])[1,1,1,1,2,3,2,,,,2]32个数，一个数代表ID的真实标签
        """
        n = inputs.size(0)#取出输入的batch
        # Compute pairwise distance, replace by the official when merged
        #计算距离矩阵，其实就是计算两个2048维之间的距离平方(a-b)**2=a^2+b^2-2ab
        #[1,2,3]*[1,2,3]=[1,4,9].sum()=14  点乘

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())#生成距离矩阵32x32，.t()表示转置
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability#clamp(min=1e-12)加这个防止矩阵中有0，对梯度下降不好
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())#利用target标签的expand，并eq，获得mask的范围，由0，1组成，，红色1表示是同一个人，绿色0表示不是同一个人
        dist_ap, dist_an = [], []#用来存放ap，an
        for i in range(n):#i表示行
            # dist[i][mask[i]],,i=0时，取mask的第一行，取距离矩阵的第一行，然后得到tensor([1.0000e-06, 1.0000e-06, 1.0000e-06, 1.0000e-06])
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))#取某一行中，红色区域的最大值，mask前4个是1，与dist相乘
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))#取某一行，绿色区域的最小值,加一个.unsqueeze(0)将其变成带有维度的tensor
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)#y是个权重，长度像dist-an
        loss = self.ranking_loss(dist_an, dist_ap, y) #ID损失：交叉商输入的是32xf f.shape=分类数,然后loss用于计算损失
                                                      #度量三元组：输入的是dist_an（从距离矩阵中，挑出一行（即一个ID）的最大距离），dist_ap
                                                     #ranking_loss输入 an ap margin y:倍率  loss： Relu(ap - anxy + margin)这个relu就起到和0比较的作用
        # from IPython import embed
        # embed()
        return loss
# class focal_loss_1(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
#         super(focal_loss_1, self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha, list):
#             assert len(alpha) == num_classes
#             self.alpha = torch.Tensor(alpha)
#         else:
#             assert alpha < 1
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1 - alpha)

#         self.gamma = gamma

#     def forward(self, preds, labels):
#         # assert preds.dim()==2 and labels.dim()==1
#         preds = preds.view(-1, preds.size(-1))
#         # print('preds',preds)
#         self.alpha = self.alpha.to(preds.device)
#         preds_softmax = F.softmax(preds, dim=1)
#         preds_logsoft = torch.log(preds_softmax)

#         # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
#         preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
#         preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
#         self.alpha = self.alpha.gather(0, labels.view(-1))
#         # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
#         loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

#         loss = torch.mul(self.alpha, loss.t())
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss
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
            if pretrain == True:
                self.model = EfficientNet.from_pretrained('efficientnet-b1')
            else:
                self.model = EfficientNet.from_name('efficientnet-b1')
            self.last_bias = nn.Parameter(torch.zeros(num_classes-1).float())
            self.model._fc = nn.Linear(self.model._fc.in_features, num_classes-1, bias=False)
            self.last_bias = nn.Parameter(torch.zeros(num_classes-1).float())
            self.fc_gender = nn.Linear(self.model._fc.in_features,2,bias=False)






        elif backbone == 'R':
            if pretrain == True:
                self.model = torchvision.models.resnet101(pretrained=True, num_classes=num_classes-1)
            else:
                self.model = torchvision.models.resnet101(num_classes=num_classes-1)
        else:
            raise KeyError

    def forward(self, x):
        x_1,x_2 = self.model(x)
        gender = self.fc_gender(x_2)
        if self.backbone == 'E':
            # x_1 = x_1.squeeze(3).squeeze(2)
            age = x_1 + self.last_bias
        if False: #export onnx
            # age_value = torch.sum(torch.sigmoid(age) > 0.5, dim=1) + 1  # not support Greater
            age_value = torch.sigmoid(age)
            gender_value = torch.softmax(gender,dim=1)
            return age_value,gender_value
        return age,gender

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin=0.1, hardest=False, squared=False,device='cuda'):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared
        self.device=device

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + 0.1)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels,self.device).float()
            triplet_loss = loss * mask

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ True

    return mask


def _get_triplet_mask(labels,device):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ True)

    mask = distinct_indices * valid_labels   # Combine the two masks

    return mask