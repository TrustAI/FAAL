import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20
from utils import normalize


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


def cw(outputs, y):

    outputs = F.softmax(outputs, 1) 

    label_mask = torch.nn.functional.one_hot(y, 10).to(torch.bool)
    label_logit = outputs[label_mask]
    others = outputs[~label_mask].reshape(-1, 9)
    top_other_logit, _ = torch.max(others, dim=1)

    margin_w = (top_other_logit - label_logit)

    return margin_w

class ModelAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(ModelAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp_trades(self, inputs_adv, inputs_clean, targets, beta,mu,std):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        output = self.proxy(normalize(inputs_adv,mu,std))
        clean_output = self.proxy(normalize(inputs_clean,mu,std))
        loss_natural = F.cross_entropy(clean_output, targets,reduction='none')
        loss_robust = F.kl_div(F.log_softmax(output, dim=1),
                               F.softmax(clean_output, dim=1),
                               reduction='none').sum(1)
        loss = - 1.0 * (loss_robust *beta + loss_natural).mean()

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff
    
    def calc_awp_at(self, inputs_adv, inputs_clean, targets,mu,std):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        output = self.proxy(normalize(inputs_adv,mu,std))

        criterion = nn.CrossEntropyLoss(reduction='none')
        loss_ce = criterion(output,targets) 
        loss = - 1.0 * (loss_ce.mean())
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)











