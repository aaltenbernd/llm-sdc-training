import torch


def preprocess(single_gpu, model):
    if single_gpu:
        return model
    else:
        return model.module


def compute_parameter_difference(model1, model2):
    total_norm = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        total_norm += torch.norm(p1.data - p2.data).item() ** 2
    return total_norm ** 0.5


def compute_gradient_norm(model):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm ** 2
    return total_norm ** 0.5


def compute_weight_norm(model):
    total_norm = 0.0
    for name, param in model.named_parameters():
        param_norm = torch.norm(param, p=2)
        total_norm += param_norm ** 2
    return total_norm ** 0.5