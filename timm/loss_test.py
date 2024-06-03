import torch
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def calc_euclidean(self, x1, x2):
        return (x1- x2).pow(2).sum(1)
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu (distance_positive-distance_negative + self.margin)
        return losses.mean()
x1=torch.tensor([3.76452,1.34567],dtype=torch.float)
p=torch.tensor([5.76452,2.34567],dtype=torch.float)
n=torch.tensor([0.76452,4.34567],dtype=torch.float)
loss=TripletLoss()
d=loss(x1,p,n)
print(d)