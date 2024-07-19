import torch

class Loss:
    def compute(self, input, target, weight):
        raise NotImplementedError("Subclasses should implement this!")

class WeightedMSELoss(Loss):
    def compute(self, input, target, weight):
        assert input.shape == target.shape == weight.shape, f'Shapes of input {input.shape}, target {target.shape}, and weight {weight.shape} must match'
        loss = torch.mean(weight * (input - target) ** 2)
        return loss
