# Adopted from https://github.com/P2333/Bag-of-Tricks-for-AT
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

# mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
# std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = 1.
lower_limit = 0.


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.1)


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

        
def normalize(X, mu=torch.tensor(cifar10_mean).view(3,1,1).cuda(), std=torch.tensor(cifar10_std).view(3,1,1).cuda()):
    return (X - mu)/std

def unnormalize(X, mu=torch.tensor(cifar10_mean).view(3,1,1).cuda(), std=torch.tensor(cifar10_std).view(3,1,1).cuda()):
    return (X *std)+ mu

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def f_6(outputs, y):

    label_mask = torch.nn.functional.one_hot(y, 10).to(torch.bool)
    label_logit = outputs[label_mask]
    others = outputs[~label_mask].reshape(-1, 9)
    top_other_logit, _ = torch.max(others, dim=1)

    margin_w = torch.mean(top_other_logit - label_logit)

    return margin_w


def l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5 
    return norms

def l2_norm_batch2(v):
    norms = (v ** 2).sum([1]) ** 0.5 
    return norms
 





def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


import random
def atta_aug(input_tensor, rst):
    batch_size = input_tensor.shape[0]
    x = torch.zeros(batch_size)
    y = torch.zeros(batch_size)
    flip = [False] * batch_size

    for i in range(batch_size):
        flip_t = bool(random.getrandbits(1))
        x_t = random.randint(0, 8)
        y_t = random.randint(0, 8)

        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
        flip[i] = flip_t
        x[i] = x_t
        y[i] = y_t

    return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def attack_trade(model, x_natural, epsilon, 
               step_size, attack_iters,
               mu, std):
    batch_size = len(x_natural)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(attack_iters):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = F.kl_div(F.log_softmax(model(normalize(x_adv, mu, std)), dim=1),
                                F.softmax(model(normalize(x_natural, mu, std)), dim=1),
                                reduction='sum')
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv - x_natural

def attack_pgd(model, X, y, epsilon, 
               alpha, attack_iters, restarts,
               mu, std, use_CWloss=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta, mu, std))
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            if use_CWloss:
                loss = CW_loss(output, y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(normalize(X+delta, mu, std)), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def pgd_attack(model, X, y, epsilon, 
               alpha, attack_iters, restarts,
               mu, std, use_CWloss=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta, mu, std))
            if use_CWloss:
                loss = CW_loss(output, y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta
            g = grad
            d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X, upper_limit - X)
            delta.data = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(normalize(X+delta, mu, std)), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def cw(outputs, y):

    outputs = F.softmax(outputs, 1) 

    label_mask = torch.nn.functional.one_hot(y, 10).to(torch.bool)
    label_logit = outputs[label_mask]
    others = outputs[~label_mask].reshape(-1, 9)
    top_other_logit, _ = torch.max(others, dim=1)

    margin_w = (top_other_logit - label_logit)

    return margin_w


def evaluate_standard(test_loader, model, mu, std, val=None):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(normalize(X, mu, std))
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            if val and i == val - 1:
                break
    return test_loss/n, test_acc/n

def evaluate_pgd(test_loader, model, mu, std, attack_iters, restarts=1, eps=8, step=2, 
                 val=None, use_CWloss=False):
    epsilon = (eps / 255.)
    if attack_iters == 1:
        alpha = epsilon
    else:
        alpha = (step / 255.)
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    results = torch.zeros((10))
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, mu, std, use_CWloss=use_CWloss)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta, mu, std))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            for cc in range(10):
                if (y==cc).sum()>0:
                    if val:
                        results[cc] += ((output[y==cc]).max(1)[1]== (y[y==cc])).sum().item()*(10/(val*128))
                    else:
                        results[cc] += ((output[y==cc]).max(1)[1]== (y[y==cc])).sum().item()*0.001 

            n += y.size(0)
        if val and i == val - 1:
            break
    return pgd_loss/n, pgd_acc/n,results

def in_class(predict, label):

    probs = torch.zeros(10)
    for i in range(10):
        # in_class_id = torch.tensor(label == i, dtype= torch.float)
        in_class_id = (label == i).clone().detach().float()
        # correct_predict = torch.tensor(predict == label, dtype= torch.float)
        correct_predict = (predict == label).clone().detach().float()
        in_class_correct_predict = (correct_predict) * (in_class_id)
        acc = torch.sum(in_class_correct_predict).item() / torch.sum(in_class_id).item()
        probs[i] = acc

    return probs

def evaluate_pgd_fair(test_loader, model, mu, std, attack_iters, restarts=1, eps=8, step=2, 
                 val=None, use_CWloss=False):
    epsilon = (eps / 255.)
    if attack_iters == 1:
        alpha = epsilon
    else:
        alpha = (step / 255.)
    pgd_loss = 0
    pgd_acc = 0
    n = 0

    correct = 0
    correct_adv = 0

    all_label = []
    all_pred = []
    all_pred_adv = []

    model.eval()
    results = torch.zeros((10))
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        all_label.append(y)

        output = model(normalize(X, mu, std))
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add = pred.eq(y.view_as(pred)).sum().item()
        correct += add
        model.zero_grad()
        all_pred.append(pred)


        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, mu, std, use_CWloss=use_CWloss)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta, mu, std))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            for cc in range(10):
                if (y==cc).sum()>0:
                    results[cc] += ((output[y==cc]).max(1)[1]== (y[y==cc])).sum().item()*0.001
            n += y.size(0)

            pred1 = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            add1 = pred1.eq(y.view_as(pred1)).sum().item()
            correct_adv += add1
            all_pred_adv.append(pred1)
        

        

        if val and i == val - 1:
            break

    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()

    acc = in_class(all_pred, all_label)
    acc_adv = in_class(all_pred_adv, all_label)

    total_clean_error = 1- correct / len(test_loader.dataset)
    total_bndy_error = correct / len(test_loader.dataset) - correct_adv / len(test_loader.dataset)

    class_clean_error = 1 - acc
    class_bndy_error = acc - acc_adv

    return class_clean_error, class_bndy_error, total_clean_error, total_bndy_error,pgd_loss/n, pgd_acc/n,results


    # return pgd_loss/n, pgd_acc/n,results




def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])

    loss_value = -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
        1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    return loss_value.mean()

def dlr_loss_targeted(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    loss_value = 0
    for target_class in range(2, 11):
        y_target = x.sort(dim=1)[1][:, -target_class]

        loss_value+= (-(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
        x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)).mean()

    return loss_value/9


def weight_average(model, new_model, decay_rate, init=False):
    model.eval()
    new_model.eval()
    state_dict = model.state_dict()
    new_dict = new_model.state_dict()
    if init:
        decay_rate = 0
    for key in state_dict:
        new_dict[key] = (state_dict[key]*decay_rate + new_dict[key]*(1-decay_rate)).clone().detach()
    model.load_state_dict(new_dict)
    return model 