import os 

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import logging
from preact_resnet import PreActResNet18
# from pre_resnet import PreActResNet18

# from resnet import ResNet18
# from wide_resnet import Wide_ResNet




# from robustbench.model_zoo.architectures.wide_resnet import WideResNet


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=1000, type=int)
parser.add_argument('--normalization', default='std', type=str, choices=['std', '01','+-1'])
parser.add_argument('--data-dir', default='./cifar-data', type=str)
parser.add_argument('--out-dir', default='faaft_out', type=str, help='Output directory')
parser.add_argument('--model-name', default='best', type=str)
parser.add_argument('--fname', default='output', type=str)
parser.add_argument('--model', default='MART', type=str, choices=['WRN','PRN'])
parser.add_argument('--pre-trained', default='MART', type=str, choices=['MART', 'AWP','TRADES','PGD'])
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--gpuid', default=0, type=int)


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

from utils import *
from robustbench.utils import load_model


if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
logfile = os.path.join(args.out_dir, args.fname+'.log')
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=logfile,
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO)


train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)


if args.normalization == 'std':
    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()
elif args.normalization == '01':
    mu = torch.tensor((0.,0.,0.)).view(3,1,1).cuda()
    std = torch.tensor((1.,1.,1.)).view(3,1,1).cuda()
elif args.normalization == '+-1':
    mu = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()
    std = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()


if args.model == 'WRN':
    if args.pre_trained == 'MART':
        from wideresnet import WideResNet
        model_test = WideResNet().cuda()
        model_test = torch.nn.DataParallel(model_test).cuda()
    
    elif args.pre_trained == 'AWP':
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet
        model_test = WideResNet(depth=34, widen_factor=10).cuda()
        model_test = torch.nn.DataParallel(model_test).cuda()


    elif args.pre_trained == 'TRADES':
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet
        model_test = WideResNet(depth=34, widen_factor=10, sub_block1=True).cuda()

    elif args.pre_trained == 'PGD':
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet
        model_test = WideResNet(depth=34, widen_factor=10).cuda()
        model_test = torch.nn.DataParallel(model_test).cuda()
elif args.model == 'PRN':
    model_test  = PreActResNet18().cuda()
else:
    assert 0 


if args.model_name=='best':
    model_path = os.path.join(args.out_dir,args.model+'_best.pth')
elif args.model_name=='last':
    model_path = os.path.join(args.out_dir,args.model+'_last.pth')
elif args.model_name=='both':
    model_path = os.path.join(args.out_dir,args.model+'_both_best.pth')
elif args.model_name=='worst':
    model_path = os.path.join(args.out_dir,args.model+'_worst_best.pth')
else:
    model_path = os.path.join(args.out_dir,args.model_name)

checkpoint = torch.load(model_path)
model_test.load_state_dict(checkpoint)




logger.info(args)
model_test.float()
model_test.eval()
print(f'Evaluating {model_path}')
logger.info(f'Evaluating {model_path}')




correct = 0
correct_adv = 0

all_label = []
all_pred = []
all_pred_adv = []
device = 'cuda'
model = model_test

for batch_idx, (data, target) in enumerate(test_loader):

    data, target = data.to(device),target.to(device)
    all_label.append(target)

    ## clean test
    output = model(normalize(data, mu, std))
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    add = pred.eq(target.view_as(pred)).sum().item()
    correct += add
    model.zero_grad()
    all_pred.append(pred)

    ## adv test
    
    pgd_delta = attack_pgd(model, data, target, args.epsilon/ 255., args.epsilon/4/ 255., 20, 1, mu, std, use_CWloss=False)
    x_adv = pgd_delta + data
    output1 = model(normalize(x_adv, mu, std))
    pred1 = output1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    add1 = pred1.eq(target.view_as(pred1)).sum().item()
    correct_adv += add1
    all_pred_adv.append(pred1)

all_label = torch.cat(all_label).flatten()
all_pred = torch.cat(all_pred).flatten()
all_pred_adv = torch.cat(all_pred_adv).flatten()

acc = in_class(all_pred, all_label)
acc_adv = in_class(all_pred_adv, all_label)

total_clean_error = 1- correct / len(test_loader.dataset)
total_bndy_error = correct / len(test_loader.dataset) - correct_adv / len(test_loader.dataset)

class_clean_error = 1 - acc
class_bndy_error = acc - acc_adv

logger.info('Evaluating pgd boundary')
logger.info(np.array([total_clean_error,class_clean_error.max().item(),total_bndy_error,class_bndy_error.max().item(),(class_clean_error+class_bndy_error).mean().item(),(class_clean_error+class_bndy_error).max().item()]))

acc_adv_pgd = acc_adv
logger.info(acc_adv_pgd)





correct = 0
correct_adv = 0

all_label = []
all_pred = []
all_pred_adv = []
device = 'cuda'
model = model_test

for batch_idx, (data, target) in enumerate(test_loader):

    data, target = data.to(device),target.to(device)
    all_label.append(target)

    ## clean test
    output = model(normalize(data, mu, std))
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    add = pred.eq(target.view_as(pred)).sum().item()
    correct += add
    model.zero_grad()
    all_pred.append(pred)

    ## adv test
    
    pgd_delta = attack_pgd(model, data, target, args.epsilon/ 255., args.epsilon/4/ 255., 20, 1, mu, std, use_CWloss=True)
    x_adv = pgd_delta + data
    output1 = model(normalize(x_adv, mu, std))
    pred1 = output1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    add1 = pred1.eq(target.view_as(pred1)).sum().item()
    correct_adv += add1
    all_pred_adv.append(pred1)

all_label = torch.cat(all_label).flatten()
all_pred = torch.cat(all_pred).flatten()
all_pred_adv = torch.cat(all_pred_adv).flatten()

acc = in_class(all_pred, all_label)
acc_adv = in_class(all_pred_adv, all_label)

total_clean_error = 1- correct / len(test_loader.dataset)
total_bndy_error = correct / len(test_loader.dataset) - correct_adv / len(test_loader.dataset)

class_clean_error = 1 - acc
class_bndy_error = acc - acc_adv

logger.info('Evaluating cw boundary')
logger.info(np.array([total_clean_error,class_clean_error.max().item(),total_bndy_error,class_bndy_error.max().item(),(class_clean_error+class_bndy_error).mean().item(),(class_clean_error+class_bndy_error).max().item()]))
logger.info(acc_adv)
logger.info('clean class acc')
logger.info(acc)

logger.info('Clean Acc \t wosrt acc \t PGD20 Acc \t worst Acc \t CW Acc \t worst Acc')
logger.info('{:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(acc.mean().item(), acc.min().item(), acc_adv_pgd.mean().item(), acc_adv_pgd.min().item(),acc_adv.mean().item(), acc_adv.min().item()))
print()
logger.info([])
