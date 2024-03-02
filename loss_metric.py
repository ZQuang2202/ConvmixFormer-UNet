import torch
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss
import torch.nn as nn

EPSILON = 1e-32


class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        # y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target, weight=self.weight,
                             ignore_index=self.ignore_index)


def classwise_iou(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    #dims = (0, *range(2, len(output.shape)))
    #gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    output = torch.argmax(output, dim=1)
    intersection = output*gt
    union = output + gt - intersection
    #classwise_iou = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)
    classwise_iou = (intersection.sum().float() + EPSILON) / (union.sum() + EPSILON)
    return classwise_iou


def classwise_f1(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1) #(n_bath, image.shape)
    #print(output)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    true_positives = true_positives[1].item()
    selected = ((output == 1)).sum().float()
    relevant = ((gt == 1)).sum().float()
    #selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    #relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()
    #print("relevant:",relevant)
    #print("selected:",selected)

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    #print(precision)
    #print(recall)
    classwise_f1 = 2 * (precision * recall + EPSILON) / (precision + recall + EPSILON)

    return classwise_f1
def classwise_dicescore(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    #print(output)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    true_positives = true_positives[1].item()
    selected = ((output == 1)).sum().float()
    relevant = ((gt == 1)).sum().float()
    dice_score = 2 * true_positives / (selected + relevant)
    return dice_score

def make_weighted_metric(classwise_metric):
    """
    Args:
        classwise_metric: classwise metric like classwise_IOU or classwise_F1
    """

    def weighted_metric(output, gt, weights=None):

        # dimensions to sum over
        dims = (0, *range(2, len(output.shape)))

        # default weights
        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            # creating tensor if needed
            if len(weights) != output.shape[1]:
                raise ValueError("The number of weights must match with the number of classes")
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            # normalizing weights
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu()

        return classwise_scores

    return weighted_metric


# Implement Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, size_average=True,):
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1-alpha])
        if isinstance(alpha, (list)) :
            self.alpha = torch.tensor(alpha)
        self.size_average = size_average
        #self.alpha = self.alpha.to(device)
    def forward(self, inputs, targets):
        """
        Inputs:
        targets : shape (N, 1, H, W), dtype = long
        inputs : shape (N, C, H, W) - has propability for each class

        Returns:
        Focal loss between groundtruth and predict
        """
        if inputs.dim() > 2:
            B, C, H, W = inputs.shape
            inputs = inputs.contiguous().permute(0,2,3,1) # shape (B, H, W, C)
            inputs = inputs.contiguous().reshape(B*H*W,C)
        targets = targets.reshape(-1, 1) # shape (N*H*W, 1)

        logpt = F.log_softmax(inputs, dim = 1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1) # shape (N*H*W)
        pt = logpt.exp()
        #print(targets.device)
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.to(inputs.dtype)
            self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets.view(-1))
            logpt = logpt * at
        loss = -1. * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# Implement Focal Loss
class DiceLoss(nn.Module):
    def __init__(self, size_average=True,):
        super().__init__()
    def forward(self, inputs, targets):
        """
        Inputs:
        targets : shape (N, 1, H, W), dtype = long
        inputs : shape (N, C, H, W) - has propability for each class

        Returns:
        Focal loss between groundtruth and predict
        """
        if inputs.dim() > 2:
            B, C, H, W = inputs.shape
            inputs = inputs.contiguous().permute(0,2,3,1) # shape (B, H, W, C)
            inputs = inputs.contiguous().reshape(B*H*W,C)
        targets = targets.reshape(-1, 1) # shape (N*H*W, 1)

        logpt = F.log_softmax(inputs, dim = 1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1) # shape (N*H*W)
        pt = logpt.exp()
        #print(targets.device)
        pt = pt.view(-1) # shape (N*H*W)
        intersection = (pt * targets.view(-1)).sum()
        #print(targets.device)
        dice = (2. * intersection + 1e-32) / (pt.sum() + targets.sum() + 1e-32)
        return 1 - dice

class CEDiceloss(nn.Module):
    def __init__(self, alpha = 0.5):
        super().__init__()
        self.alpha = alpha
    def forward(self, inputs, targets):
        criterion1 = DiceLoss()
        criterion2 = nn.CrossEntropyLoss()
        return self.alpha * criterion1(inputs, targets) + (1 - self.alpha) * criterion2(inputs, targets)