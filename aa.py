import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


def get_test_loader(dir_, batch_size):
    
    num_workers = 2
    
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,)
    return test_loader


def evaluate_autoattack(test_loader, model, batch_size, eps=8, log=None):
    epsilon = (eps / 255.)
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, verbose=False,log_path=log, version='standard')
    model.eval()
    all_pred_adv = []
    all_label = []
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        x_adv,y_adv = adversary.run_standard_evaluation(X, y, bs=batch_size,return_labels=True)
        all_pred_adv.append(y_adv)
        all_label.append(y)
    all_pred_adv = torch.cat(all_pred_adv).flatten()
    all_label = torch.cat(all_label).flatten()

    acc_adv = in_class(all_pred_adv, all_label)
    return acc_adv

    


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=200, type=int)
parser.add_argument('--normalization', default='01', type=str, choices=['std', '01','+-1'])
parser.add_argument('--data-dir', default='./cifar-data', type=str)
parser.add_argument('--out-dir', default='mdeat_out', type=str)
parser.add_argument('--model-name', default='model_pre', type=str)
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--log-name', default='aa_score', type=str)
parser.add_argument('--model', default='PRN', type=str, choices=['WRN','PRN'])
parser.add_argument('--pre-trained', default='MART', type=str, choices=['MART', 'AWP','TRADES','PGD'])
parser.add_argument('--gpuid', default=0, type=int)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
from wideresnet import WideResNet
from preact_resnet import PreActResNet18
from autoattack import AutoAttack
# from resnet import ResNet18
from utils import in_class

if args.normalization == 'std':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
elif args.normalization == '01':
    mean = (0, 0, 0)
    std = (1, 1, 1)
elif args.normalization == '+-1':
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

log_path = os.path.join(args.out_dir,args.log_name+'.log')
test_loader = get_test_loader(args.data_dir, args.batch_size)
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

if args.model =='WRN': 
    if args.pre_trained == 'MART':
        from wideresnet import WideResNet
        net = WideResNet().cuda()
        net = torch.nn.DataParallel(net).cuda()

    elif args.pre_trained == 'AWP':
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet
        net = WideResNet(depth=34, widen_factor=10).cuda()
        net = torch.nn.DataParallel(net).cuda()

    elif args.pre_trained == 'TRADES':
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet
        net = WideResNet(depth=34, widen_factor=10, sub_block1=True).cuda()

    elif args.pre_trained == 'PGD':
        from robustbench.model_zoo.architectures.wide_resnet import WideResNet
        net = WideResNet(depth=34, widen_factor=10).cuda()
        net = torch.nn.DataParallel(net).cuda()
else:
    net = PreActResNet18().cuda()




# net = torch.nn.DataParallel(net).cuda()
net.load_state_dict(checkpoint)
model_test = nn.Sequential(Normalize(mean=mean, std=std), net)
model_test.float()
model_test.eval()
print(f'Evaluating {model_path}')
acc_adv = evaluate_autoattack(test_loader,model_test,args.batch_size,args.epsilon,log_path)
print(acc_adv)
print(acc_adv.mean())
print(acc_adv.min())


