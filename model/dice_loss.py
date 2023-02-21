import torch
import torch.nn as nn
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self, multiclass: bool = False):
        super(DiceLoss, self).__init__()
        self.multiclass = multiclass

    def dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        assert input.dim() == 3 or not reduce_batch_first

        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()

    def multiclass_dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all classes
        return self.dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

    def forward(self, input: Tensor, target: Tensor):
        # Dice loss (objective to minimize) between 0 and 1
        if self.multiclass:
            return 1 - self.multiclass_dice_coeff(input, target, reduce_batch_first=True)
        else:
            return 1 - self.dice_coeff(input, target, reduce_batch_first=True)
