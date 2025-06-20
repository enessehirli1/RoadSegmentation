import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Softmax uygula
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        dice_loss = 0
        for i in range(self.num_classes):
            input_flat = inputs[:, i, :, :].contiguous().view(-1)
            target_flat = targets_one_hot[:, i, :, :].contiguous().view(-1)
            
            intersection = (input_flat * target_flat).sum()
            dice = (2 * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
            dice_loss += 1 - dice
        
        return dice_loss / self.num_classes

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=2, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.alpha * dice + (1 - self.alpha) * ce

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, num_classes=2, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # False Positive penalty
        self.beta = beta    # False Negative penalty
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        tversky_loss = 0
        for i in range(self.num_classes):
            input_flat = inputs[:, i, :, :].contiguous().view(-1)
            target_flat = targets_one_hot[:, i, :, :].contiguous().view(-1)
            
            TP = (input_flat * target_flat).sum()
            FP = ((1 - target_flat) * input_flat).sum()
            FN = (target_flat * (1 - input_flat)).sum()
            
            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            tversky_loss += 1 - tversky
        
        return tversky_loss / self.num_classes