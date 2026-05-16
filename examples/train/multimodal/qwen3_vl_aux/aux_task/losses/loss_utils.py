import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        target = target.to(device=input.device)
        weights = weights.to(device=input.device, dtype=input.dtype)
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
            torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        target = target.to(device=input.device, dtype=input.dtype)
        weights = weights.to(device=input.device, dtype=input.dtype)
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.register_buffer('code_weights', None)
        if code_weights is not None:
            code_weights = np.array(code_weights, dtype=np.float32)
            self.register_buffer('code_weights', torch.from_numpy(code_weights))

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = target.to(device=input.device, dtype=input.dtype)
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            code_weights = self.code_weights.to(device=diff.device, dtype=diff.dtype)
            diff = diff * code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            weights = weights.to(device=loss.device, dtype=loss.dtype)
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss

class MaskedRegressionLoss(torch.nn.Module):
    def __init__(self):
        super(MaskedRegressionLoss, self).__init__()
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='none')

    def forward(self, regress_results, gt, mask, weights=None):
        gt = gt.to(device=regress_results.device, dtype=regress_results.dtype)
        mask = mask.to(device=regress_results.device, dtype=regress_results.dtype)
        loss = self.loss_fn(regress_results, gt)
        if weights is not None:
            weights = weights.to(device=regress_results.device, dtype=regress_results.dtype)
            loss *= weights
        if len(loss.shape) == 4 and len(mask.shape) == 3:
            loss = torch.mean(loss, 1)
        loss *= mask
        loss = torch.sum(loss) / (1 + torch.sum(mask))
        return loss

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, gt, mask):
        gt = gt.to(device=pred.device, dtype=pred.dtype)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        out = torch.sum(((pred - gt) * mask)**2.0)  / (torch.sum(mask) + 1 )
        return out

class MultiClassFocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha, gamma=2, size_average=True, ignore_class=255):
        super(MultiClassFocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6
        self.ignore_class = ignore_class
        self.alpha = torch.Tensor(alpha)

    def forward(self, pred, target, loss_weight=None):
        ignore_idx = (target == self.ignore_class)
        target[ignore_idx] = 0
        ignore_idx = ignore_idx.view(-1)

        logit = F.softmax(pred, dim=1)
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.contiguous().view(-1, 1)

        pt = logit.gather(1, target).view(-1) + self.eps
        logpt = torch.log(pt)
        alpha = self.alpha.to(device=logpt.device, dtype=logpt.dtype)

        alpha_class = alpha.gather(0, target.view(-1))
        logpt = alpha_class * logpt

        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt
        try:
            loss[ignore_idx] = 0
        except:
            print("-------------------------")
            print("target shape: ", target.size())
            print("loss shape: ", loss.size())
            print(" ")
        # loss[ignore_idx.view(-1)] = 0

        if loss_weight is not None:
            loss_weight_flat = loss_weight.view(-1)

            loss *= loss_weight_flat

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, outputs, labels, mask):
        mask = mask.to(device=outputs.device, dtype=outputs.dtype)
        loss = self.loss(outputs, labels) * mask
        mask_loss = torch.sum(loss) / (torch.sum(mask) + 1)
        return mask_loss

class MaskedDiceLoss(nn.Module):
    def __init__(self, num_classes=5, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred, target, mask):
        """
        Args:
            pred:   (N, C, H, W)  # 未归一化的 logits
            target: (N, H, W)      # 类别标签（0 ~ num_classes-1）
            mask:   (N, H, W)      # 二值掩码（0或1，1的区域参与计算）
        """
        target = target.to(device=pred.device)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        # 计算 masked 交集和并集
        pred = F.softmax(pred, dim=1) * mask  # 掩码作用在预测结果
        target = target * mask   # 掩码作用在真实标签

        intersection = (pred * target).sum(dim=(2, 3))  # (N, C)
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # (N, C)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # 平均所有类和样本的 Dice Loss

class MaskedBCELoss(nn.Module):
    """
    binary cross entropy loss with element-wise mask
    """
    def __init__(self):
        super().__init__()
        self.BCE_loss = nn.BCELoss(reduction="none")

    def forward(self, pred, labels, weight):
        labels = labels.to(device=pred.device, dtype=pred.dtype)
        weight = weight.to(device=pred.device, dtype=pred.dtype)
        loss = self.BCE_loss(pred, labels) * weight
        return loss

class MaskedBCEWithLogitsLoss(nn.Module):
    """
    binary cross entropy loss with element-wise mask
    """
    def __init__(self):
        super().__init__()
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, labels, weight):
        labels = labels.to(device=pred.device, dtype=pred.dtype)
        weight = weight.to(device=pred.device, dtype=pred.dtype)
        loss = self.BCE_loss(pred, labels) * weight
        return loss

class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if alpha is None:
            alpha_tensor = torch.ones(self.num_class, dtype=torch.float32)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == self.num_class
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32).view(self.num_class)
            alpha_tensor = alpha_tensor  # / alpha_tensor.sum()
        elif isinstance(alpha, float):
            alpha_tensor = torch.ones(self.num_class, dtype=torch.float32)
            alpha_tensor = alpha_tensor * (1 - alpha)
            alpha_tensor[balance_index] = alpha
        else:
            raise TypeError('Not support alpha type')
        self.register_buffer('alpha', alpha_tensor)

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target, weight=None, dustbin_index=None):
        logit = F.softmax(input, dim=1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha.to(device=input.device, dtype=input.dtype).clone()

        idx = target.long()
        # one_hot_key = torch.FloatTensor(target.shape[0], self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key = F.one_hot(idx.to(torch.int64), self.num_class)
        t_perm_idx = list(range(len(one_hot_key.shape)-1))
        t_perm_idx.insert(1, len(one_hot_key.shape)-1)
        one_hot_key = one_hot_key.permute(t_perm_idx).contiguous()
        one_hot_key = one_hot_key.to(device=logit.device, dtype=logit.dtype)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (self.num_class - 1), 1.0 - self.smooth)

        # make it more confident since weights are usually smaller than 1
        # The weights are related to central position of voxel.
        # if weight is not None:
        #     one_hot_key = weight.unsqueeze(1) * one_hot_key
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        if dustbin_index is not None:
            alpha[int(dustbin_index)] *= 0.2

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        
        if weight is not None:
            weight = weight.to(device=loss.device, dtype=loss.dtype)
            loss = (loss * weight)
            loss = loss.sum() / ((weight > 0).to(loss.dtype).sum() + 0.001)
        
        return loss
