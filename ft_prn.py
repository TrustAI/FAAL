import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import logging
import time
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from robustbench.utils import load_model


def cw(outputs, y):

    outputs = F.softmax(outputs, 1) 

    label_mask = torch.nn.functional.one_hot(y, 10).to(torch.bool)
    label_logit = outputs[label_mask]
    others = outputs[~label_mask].reshape(-1, 9)
    top_other_logit, _ = torch.max(others, dim=1)

    margin_w = (top_other_logit - label_logit)

    return margin_w





parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--data-dir', default='./cifar-data', type=str)
parser.add_argument('--distance', type=str)
parser.add_argument('--epochs', default=110, type=int)
parser.add_argument('--model', default='PRN', type=str, choices=['PRN'])
parser.add_argument('--lr-schedule', default='multistep', type=str, choices=['cyclic', 'flat', 'multistep'])
parser.add_argument('--lr-min', default=0.0, type=float)
parser.add_argument('--lr-max', default=0.1, type=float)
parser.add_argument('--scale', default=1, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--normalization', default='std', type=str, choices=['std', '01','+-1'])
parser.add_argument('--alpha', default=0.5, type=float, help='Step size')
parser.add_argument('--steps', default=2, type=int, help='number of pgd')
parser.add_argument('--fname', default='output', type=str)
parser.add_argument('--out-dir', default='faal_prn_out', type=str, help='Output directory')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--save-model', action='store_true')
parser.add_argument('--pre-trained', default='PGD', type=str, choices=['PGD','TRADES','None'])
parser.add_argument("--betas", type=tuple, default=(0.5,0.9999))
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--loss', type=str, choices=['MART', 'TRADES-AWP','TRADES','AT','AT-AWP'])



args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

from utils import *
from FAAL import * 
print(args)
print(torch.cuda.get_device_name(0))
if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
logfile = os.path.join(args.out_dir, args.fname+'.log')
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=logfile,
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO)

logger.info(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
epsilon = (args.epsilon / 255.)
alpha = (args.alpha / 255.)
if args.normalization == 'std':
    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()
elif args.normalization == '01':
    mu = torch.tensor((0.,0.,0.)).view(3,1,1).cuda()
    std = torch.tensor((1.,1.,1.)).view(3,1,1).cuda()
elif args.normalization == '+-1':
    mu = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()
    std = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()




# model = torch.nn.DataParallel(model).cuda()

if args.model == 'PRN':
    from preact_resnet import PreActResNet18
    model  = PreActResNet18().cuda()
else:
    assert 0 

if args.pre_trained == 'TRADES':
    model.load_state_dict(torch.load('PRN/models/cifar10/PRN_TRADES.pth'))
elif args.pre_trained == 'PGD':
    model.load_state_dict(torch.load('PRN/models/cifar10/PRN_PGD.pth'))
else:
    assert 0 


# for Evaluation
model_test = PreActResNet18().cuda()
model_test.eval()


model.train()

opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)


lr_steps = args.epochs * len(train_loader)
# lr_steps = 0
if args.lr_schedule == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
        # step_size_up=1, step_size_down=lr_steps)
        step_size_up=lr_steps/2, step_size_down=lr_steps/2)
elif args.lr_schedule == 'flat':
    lr_lamdbda = lambda t: 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lamdbda)
elif args.lr_schedule == 'multistep':
    if args.epochs==200:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.epochs / 2, args.epochs * 3 / 4], gamma=0.1)
    elif args.epochs==110:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.epochs * 100 / 110, args.epochs * 105 / 110], gamma=0.1)
    elif args.epochs==50:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.epochs * 25 / 50, args.epochs * 40 / 50], gamma=0.1)
    elif args.epochs==2 or args.epochs==4 or args.epochs==100:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.epochs * 1/2], gamma=0.1)
    elif args.epochs==10 or args.epochs==6 or args.epochs==20 :
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1], gamma=0.1)

    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.epochs * 1 / 3, args.epochs * 2 / 3], gamma=0.1)




DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Training
highest_acc = 0
highest_acc_worst = 0
highest_acc_both = 0
train_time = 0
logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t Val Acc \t PGD Acc \t CW Acc')
scale = args.scale

criterion = nn.CrossEntropyLoss(reduction='none')


lr = scheduler.get_last_lr()[0]


if args.loss=='AT-AWP' or args.loss=='TRADES-AWP':
    from utils_awp import ModelAWP
    proxy = copy.deepcopy(model) 
    proxy_optim = torch.optim.SGD(proxy.parameters(), lr=0.1)
    awp_adversary = ModelAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=0.005)


for epoch in range(args.epochs):
    start_epoch_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    
    lr = scheduler.get_last_lr()[0]
    for i, (X, y) in enumerate(train_loader):
        X, y = X.cuda(), y.cuda()

        model.eval()
        if args.loss=='AT' or args.loss=='AT-AWP':
            perturbation = pgd_attack(model, X, y, epsilon, alpha, args.steps, 1, mu, std, use_CWloss=False)
        elif args.loss=='TRADES' or args.loss=='TRADES-AWP' or args.loss=='MART':
            perturbation = attack_trade(model, X, epsilon, alpha, args.steps, mu, std)
        X_adv = (X+perturbation.detach()).clamp(0,1)
        model.train()
        if args.loss=='AT-AWP':
            awp = awp_adversary.calc_awp_at(inputs_adv=X_adv,
                                         inputs_clean=X,
                                         targets=y,
                                         mu=mu,
                                         std=std)
            awp_adversary.perturb(awp)
        if args.loss=='TRADES-AWP':
            awp = awp_adversary.calc_awp_trades(inputs_adv=X_adv,
                                         inputs_clean=X,
                                         targets=y,
                                         beta=6.,
                                         mu=mu,
                                         std=std)
            awp_adversary.perturb(awp)
        
        opt.zero_grad()
        output=model(normalize(X_adv,mu,std))
        output_clean=model(normalize( X,mu,std))
        loss_margin = cw(output,y)
        loss_ce =  criterion(output,y)
        
        class_margin = torch.zeros((10)).cuda()
        class_ce = torch.zeros((10)).cuda()
        class_n = torch.zeros((10)).cuda()

        for cc in range(10):
            class_margin[cc] += (loss_margin[y==cc]).mean() 
            class_ce[cc] += (loss_ce[y==cc]).mean() 
            class_n[cc] += 1/(y==cc).sum()
        class_margin = torch.where(torch.isnan(class_margin),-torch.ones_like(class_margin),class_margin)

        valid_num = len(class_margin)


   
        
        FAAL_DRO = DAW(train_batch_size =10,
                output_return = "weights",
                learning_approach = args.distance, 
                r_choice = scale)

        
        weights= FAAL_DRO.solve_weight(y,class_margin, device = 'cuda')

        if args.loss=='AT' or args.loss == 'AT-AWP':

            loss = (loss_ce*weights[y]).sum()*valid_num/len(y)

        elif args.loss=='TRADES' or args.loss=='TRADES-AWP':
            loss_kl = F.kl_div(F.log_softmax(output, dim=1),
                           F.softmax(output_clean, dim=1),
                           reduction='none').sum(1)*6 + criterion(output_clean,y)
            loss = (loss_kl*weights[y]).sum()*valid_num/len(y)

        elif args.loss=='MART':
            adv_probs = F.softmax(output, dim=1)

            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

            new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

            loss_adv = F.cross_entropy(output, y,reduction='none') + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y,reduction='none')

            nat_probs = F.softmax(output_clean, dim=1)

            true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
            kl = nn.KLDivLoss(reduction='none')
            loss_robust = torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs)

            loss_mart = loss_robust*6 +loss_adv

            loss = (loss_mart*weights[y]).sum()*valid_num/len(y)

            
        loss.backward()
        opt.step()
        if args.lr_schedule == 'cyclic':
            scheduler.step()

        if args.loss=='AT-AWP' or args.loss=='TRADES-AWP':
            awp_adversary.restore(awp)

        
        
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

    



    epoch_end_time = time.time()
    epoch_time = epoch_end_time - start_epoch_time
    train_time += epoch_time
    if args.lr_schedule != 'cyclic':
        scheduler.step()
    
    

    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    val_adv_loss, val_adv_acc,cw_results = evaluate_pgd(test_loader, model_test, mu, std, 10, 1, eps=args.epsilon,step = args.epsilon/4,val=20, use_CWloss=True)
    val_loss, val_acc = evaluate_standard(test_loader, model_test, mu, std, val=20)
    logger.info('%d \t %.1f \t \t %.7f \t %.4f \t %.4f \t %.4f\t %.4f \t %.4f',
        epoch, epoch_time, lr, train_loss/train_n, train_acc/train_n, val_acc, val_adv_acc,torch.min(cw_results))

    
    if val_adv_acc >= highest_acc and args.save_model:
        highest_acc = val_adv_acc
        highest_idx = epoch
        torch.save(model.state_dict(), os.path.join(args.out_dir, f'{args.model}_{args.pre_trained}_{args.lr_max}_{args.scale}_best.pth'))
    if torch.min(cw_results) >= highest_acc_worst and args.save_model:
        highest_acc_worst = torch.min(cw_results)
        highest_worst_idx = epoch
        torch.save(model.state_dict(), os.path.join(args.out_dir, f'{args.model}_{args.pre_trained}_{args.lr_max}_{args.scale}_worst_best.pth'))
    if (torch.min(cw_results)+val_adv_acc) >= highest_acc_both and args.save_model:
        highest_acc_both = torch.min(cw_results)+val_adv_acc
        highest_both_idx = epoch
        torch.save(model.state_dict(), os.path.join(args.out_dir, f'{args.model}_{args.pre_trained}_{args.lr_max}_{args.scale}_both_best.pth'))
    if epoch==(args.epochs-1):
        torch.save(model.state_dict(), os.path.join(args.out_dir, f'{args.model}_{args.pre_trained}_{args.lr_max}_{args.scale}_last.pth'))


    


logger.info('Total train time: %.4f minutes', (train_time)/60)
logger.info(f'Best avg checkpoint at {highest_idx}, {highest_acc}')
logger.info(f'Best worst checkpoint at {highest_worst_idx}, {highest_acc_worst}')
logger.info(f'Best both checkpoint at {highest_both_idx}, {highest_acc_both}')


# if __name__ == "__main__":
#     main()


