import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class DualFocalLoss(nn.Module):
    def __init__(self, gamma=1.0, hard_ratio=None):
        super(DualFocalLoss, self).__init__()
        self.gamma = gamma
        self.hard_ratio = hard_ratio

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        loss = -1 * (1 - p_k + p_j) ** self.gamma * logp_k
        if self.hard_ratio is None:
            return loss.mean()
        else:
            OHEM, _ = loss.topk(k=int(self.hard_ratio * [*loss.shape][0]))
            return OHEM.mean()


class SmoothedCE(nn.Module):
    '''
        CELoss for mixup, label smoothing, etc. Labels are mixed one hot
    '''
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets):
        loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))
        return loss


def loss_gls(logits, labels, smooth_rate=0.1, hard_ratio=None): # To Smooth or Not? When Label Smoothing Meets Noisy Labels
    # logits: model prediction logits before the soft-max, with size [batch_size, classes]
    # labels: the (noisy) labels for evaluation, with size [batch_size]
    # smooth_rate: could go either positive or negative,
    # smooth_rate candidates we adopted in the paper: [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0].
    confidence = 1. - smooth_rate
    logprobs = F.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smooth_rate * smooth_loss
    if hard_ratio is None:
        return torch.sum(loss)/loss.shape[0]
    else:
        OHEM, _ = loss.topk(k=int(hard_ratio * [*loss.shape][0]))
        return OHEM.mean()

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


class SmoothCE(nn.Module):
    def __init__(self, smoothing=0.1, hard_ratio=None):
        super(SmoothCE, self).__init__()
        self.smoothing = smoothing
        self.hard_ratio = hard_ratio

    def forward(self, pred, gold):
        n_class = pred.size(1)

        one_hot = torch.full_like(pred, fill_value=self.smoothing / (n_class - 1))
        one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - self.smoothing)
        log_prob = F.log_softmax(pred, dim=1)

        loss = F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
        if self.hard_ratio is None:
            return torch.sum(loss) / loss.shape[0]
        else:
            OHEM, _ = loss.topk(k=int(self.hard_ratio * [*loss.shape][0]))
            return OHEM.mean()


class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):
        # cross_entropy = F.cross_entropy(output, target)
        # cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


class FocalOHEMLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25, percent=0.5, num_classes=2):
        super(FocalOHEMLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param
        self.OHEM_percent = percent
        self.num_classes = num_classes

    def forward(self, output, target):
        if not output.shape[-1] == self.num_classes:
            self.num_classes = output.shape[-1]
        target = torch.nn.functional.one_hot(target, self.num_classes).type_as(target)
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)

        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()
        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-output * (target * 2 - 1))
        focal_loss = self.balance_param * (invprobs * self.focusing_param).exp() * loss

        # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM, _ = focal_loss.topk(k=int(self.OHEM_percent * [*focal_loss.shape][0]))
        return OHEM.mean()


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes=2, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()
        normalizor = 1 / 4 * (self.num_classes - 1)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * normalizor * rce.mean()


class NormalizedReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes=2, scale=1.0):
        super(NormalizedReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        normalizor = 1 / 4 * (self.num_classes - 1)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * normalizor * rce.mean()


class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes=2, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).type_as(labels).float()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()


class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes=2, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).type_as(labels).float()
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()


class NormalizedGeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes=2, scale=1.0, q=0.7):
        super(NormalizedGeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).type_as(labels).float()
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()


class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes=2, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).type_as(labels).float()
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * mae.mean()


class NormalizedMeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes=2, scale=1.0):
        super(NormalizedMeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        normalizor = 1 / (2 * (self.num_classes - 1))
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * normalizor * mae.mean()


class NCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)


class NCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)


class GCEandMAE(torch.nn.Module):
    def __init__(self, num_classes, q=0.7, beta=1.0):
        super(GCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.mae(pred, labels)


class GCEandRCE(torch.nn.Module):
    def __init__(self, num_classes, beta=1.0, q=0.7):
        super(GCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.rce(pred, labels)


class GCEandNCE(torch.nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.nce = NormalizedCrossEntropy(num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.nce(pred, labels)


class NGCEandNCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.nce(pred, labels)


class NGCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.mae(pred, labels)


class NGCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.rce(pred, labels)


class MAEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(MAEandRCE, self).__init__()
        self.num_classes = num_classes
        self.mae = MeanAbsoluteError(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.mae(pred, labels) + self.rce(pred, labels)


class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).to(pred.device).random_(1,
                                                                                           self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes - 1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels) * self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight.to(pred.device)[labels].unsqueeze(-1).expand(s_neg.size()).to(pred.device)
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float(
            (labels_neg >= 0).sum())
        loss = ((loss + loss_neg) / (float((labels >= 0).sum()) + float((labels_neg[:, 0] >= 0).sum())))
        return loss


# class FocalLoss(torch.nn.Module):
#     '''
#         https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
#     '''
#
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.Tensor([alpha, 1-alpha])
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)
#
#         logpt = F.log_softmax(input, dim=1)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = torch.autograd.Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * torch.autograd.Variable(at)
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()


class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class NFLandNCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandNCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.nce(pred, labels)


class NFLandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


class NFLandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


class DMILoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = y_onehot @ outputs
        return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)


def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class SmoothCEOHEMBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super(SmoothCEOHEMBCE, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred_, gold_):
        clean_mask = gold_ > 0
        dirty_mask = gold_ < 1
        clean_mask_sum = clean_mask.sum()
        dirty_mask_sum = dirty_mask.sum()
        select_item = max(1, min(clean_mask_sum, dirty_mask_sum))
        loss = 0
        if clean_mask_sum > 0:
            mask = clean_mask
            pred, gold = pred_[mask], gold_[mask]
            n_class = pred.size(1)
            one_hot = torch.full_like(pred, fill_value=self.smoothing / (n_class - 1))
            one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - self.smoothing)
            log_prob = F.log_softmax(pred, dim=1)
            loss_, _ = F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).topk(k=select_item)
            loss += loss_.mean()

        if dirty_mask_sum > 0:
            mask = dirty_mask
            pred, gold = pred_[mask], gold_[mask]
            n_class = pred.size(1)
            one_hot = torch.full_like(pred, fill_value=self.smoothing / (n_class - 1))
            one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - self.smoothing)
            log_prob = F.log_softmax(pred, dim=1)
            loss_, _ = F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).topk(k=select_item)
            loss += loss_.mean()

        return loss


if __name__ == '__main__':
    criterion = SmoothCE(hard_ratio=0.3)
    loss = criterion(torch.randn(20, 10), torch.zeros(20).long())