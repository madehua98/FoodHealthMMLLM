import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union


class LabelAwareSmoothing(nn.Module):
    def __init__(self, cls_num_list, smooth_head, smooth_tail, shape='concave', power=None):
        super(LabelAwareSmoothing, self).__init__()

        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin(
                (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(
                1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power(
                (np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        # if torch.cuda.is_available():
        # self.smooth = self.smooth.cuda()

    def forward(self, x, target):
        smoothing = self.smooth[target].to(x.device)
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()


def get_batch_distribution(inputs: torch.Tensor, num_classes: int) -> torch.Tensor:
    """[function to get classes distribution within a batch]

    Args:
        inputs (torch.Tensor): [class labels]
        num_classes (int): [number of classes]

    Returns:
        torch.Tensor: [distribution of classes]
    """
    distribution = torch.zeros(num_classes)
    for c in range(num_classes):
        distribution[c] = torch.sum(inputs == c)

    return distribution


class ClassFocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
                 reduction: Optional[str] = 'none') -> None:
        """[Focal Loss Class. Hacked from: https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py]

        Args:
            alpha (float): [alpha hyperparameter for focal loss]
            gamma (Optional[float], optional): [gamma hyperparams]. Defaults to 2.0.
            reduction (Optional[str], optional): [reduction type]. Defaults to 'none'.
        """
        super(ClassFocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            weight: torch.Tensor = None) -> torch.Tensor:
        """[focal loss forward pass]

        Args:
            input (torch.Tensor): [input tensor. Dims: (batch_size, num_classes)]
            target (torch.Tensor): [target tensor. Dims: (batch_size, num_classes)]
            weight (torch.Tensor, optional): [weigths for each class. Dims: (num_classes)]. Defaults to None.

        Returns:
            torch.Tensor: [computed loss value]
        """
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 2:
            raise ValueError("Invalid input shape, we expect BxD. Got: {}"
                             .format(input.shape))
        if not input.shape[0] == target.shape[0]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1) + self.eps

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input.shape[1])

        # compute the actual focal loss
        fl_weight = torch.pow(1. - input_soft, self.gamma)
        focal = -self.alpha * fl_weight * torch.log(input_soft)
        loss_tmp = target_one_hot * focal

        # weighting the loss
        if weight is not None:
            assert loss_tmp.shape[1] == weight.shape[0]
            loss_tmp = torch.mul(loss_tmp, weight)

        loss_tmp = torch.sum(loss_tmp, dim=1)

        loss = -1
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss


class ClassBalancedLoss(nn.Module):
    def __init__(self,
                 beta: float = 0.9,
                 num_classes: int = 7,
                 **kwargs):
        """[Class Balanced Loss Module]

        Args:
            loss (dict, optional): [loss function configuration]. Defaults to None.
            beta (float, optional): [beta hyperparam]. Defaults to 0.9.
            num_classes (int, optional): [number of classes]. Defaults to 7.
        """
        super().__init__()

        self.beta = torch.tensor(beta)
        self.loss = ClassFocalLoss(gamma=0.0, alpha=0.5, reduction='mean')
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.dataset_distribution = torch.zeros(self.num_classes)
        self.num_iter = 0
        self.epsilon = 1e-6

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """[pass forward]

        Args:
            inputs (torch.Tensor): [inputs feed to loss.]
            targets (torch.Tensor): [targets feed to loss.]

        Returns:
            [torch.Tensor]: [loss value]
        """
        loss = torch.tensor(0.0).type_as(inputs)
        distribution = get_batch_distribution(targets, self.num_classes) + self.epsilon
        weight = ((1.0 - self.beta) / (1.0 - self.beta ** distribution)).type_as(inputs)
        loss += self.loss(inputs, targets, weight=weight, **self.kwargs)

        self.num_iter += 1
        self.dataset_distribution += distribution

        return loss
