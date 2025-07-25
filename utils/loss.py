import torch
import torch.nn as nn
import torch.nn.functional as F


class Focal_MultiLabel_Loss(nn.Module):
    def __init__(self, gamma=2.0):
      super(Focal_MultiLabel_Loss, self).__init__()
      self.gamma = gamma
      self.bceloss = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets): 
      bce = F.cross_entropy(inputs, targets, reduction='none')
      bce_exp = torch.exp(-bce)
      focal_loss = (1-bce_exp)**self.gamma * bce
      return focal_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=1.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        # Weighting for class 1
        F_loss = F_loss * (self.pos_weight * targets + (1 - targets))

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class DiceLoss(nn.Module):
    # Dice loss is commonly used in medical image segmentation tasks. It mitigates class imbalance by maximizing the average of prediction probabilities for each class.
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).float()

        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)

        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_coeff.mean()


class TverskyLoss(nn.Module):
    # A generalized form of Dice Loss that introduces hyperparameters to control the trade-off between False Positives and False Negatives. This enables better control over class imbalance.
    # This function can be used for binary classification problems, but caution is needed when applying to multi-class classification problems
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.nn.functional.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        
        true_positive = (inputs * targets).sum(dim=1)
        false_positive = ((1 - targets) * inputs).sum(dim=1)
        false_negative = (targets * (1 - inputs)).sum(dim=1)

        tversky_index = true_positive / (true_positive + self.alpha * false_negative + self.beta * false_positive + self.smooth)
        return 1 - tversky_index.mean()


class FBetaLoss(nn.Module):
    # A generalized version of the F1 score, which is the harmonic mean of Precision and Recall. By adjusting the beta parameter, you can control whether to emphasize Recall or Precision.
    def __init__(self, beta=1., smooth=1e-5):
        super(FBetaLoss, self).__init__()
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.nn.functional.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).float()

        true_positive = (inputs * targets).sum(dim=1)
        precision = true_positive / (inputs.sum(dim=1) + self.smooth)
        recall = true_positive / (targets.sum(dim=1) + self.smooth)

        fbeta = (1 + self.beta**2) * precision * recall / ((self.beta**2 * precision) + recall + self.smooth)
        return 1 - fbeta.mean()


class CrossEntropyFBetaLoss(nn.Module):
    def __init__(self, beta=1., eps=1e-7, weight_ce=0.5, weight_fbeta=0.5, weight=None):
        # beta < 1: emphasizes precision more. That is, it focuses more on suppressing the number of False Positives (cases that are actually Negative but predicted as Positive).
        # beta > 1: emphasizes recall more. That is, it focuses more on suppressing the number of False Negatives (cases that are actually Positive but predicted as Negative).
        # eps: a small value to prevent division by zero during calculation. Default is 1e-7.
        # weight_ce: weight for cross-entropy loss (ce_loss). Adjusts the influence of ce_loss in the total loss. Default is 0.5.
        # weight_fbeta: weight for FBeta loss (fbeta_loss). Adjusts the influence of fbeta_loss in the total loss. Default is 0.5.
        super(CrossEntropyFBetaLoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.weight_ce = weight_ce
        self.weight_fbeta = weight_fbeta
        self.weight = weight

    def forward(self, inputs, targets):
        print(f'inputs.shape: {inputs.shape} | targets.shape: {targets.shape}')
        # Compute CrossEntropyLoss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        
        # Compute FBetaLoss
        inputs = torch.sigmoid(inputs)
        true_positive = (inputs * targets_one_hot).sum(dim=0)
        false_positive = (inputs * (1 - targets_one_hot)).sum(dim=0)
        false_negative = ((1 - inputs) * targets_one_hot).sum(dim=0)
        fbeta_loss = 1 - ((1 + self.beta ** 2) * true_positive + self.eps) / ((1 + self.beta ** 2) * true_positive + self.beta ** 2 * false_negative + false_positive + self.eps)
        fbeta_loss = fbeta_loss.mean()

        # Combine losses
        loss = self.weight_ce * ce_loss + self.weight_fbeta * fbeta_loss
        return loss
