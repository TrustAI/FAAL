# Towards Fairness-Aware Adversarial Learning (FAAL)

Code for CVPR 2024 paper "[Towards Fairness-Aware Adversarial Learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Towards_Fairness-Aware_Adversarial_Learning_CVPR_2024_paper.pdf)".


## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:
```
conda create -n faal python=3.8 -y
conda activate faal
pip install torch torchvision mosek gpustat matplotlib torchattacks cvxpy tensorboard pandas
pip install --upgrade git+https://github.com/fra31/auto-attack.git@6482e4d6fbeeb51ae9585c41b16d50d14576aadc#egg=autoattack
```

## Download pre-trained models for fine-tuning

1. Download cifar10_wide10_linf_eps8.pth and put it under WRN/models/cifar10/: 
```
wget -O WRN/models/cifar10/cifar10_wide10_linf_eps8.pth https://www.dropbox.com/scl/fi/8z2ltlcbdd0vuus2tnc6y/cifar10_wide10_linf_eps8.pth?rlkey=zmlrx4ffh3tkbgbqpt6t2jglu&dl=1
```
2. Download rob_cifar_mart.pt and put it under WRN/models/cifar10/:
```
wget -O WRN/models/cifar10/rob_cifar_mart.pt https://www.dropbox.com/scl/fi/kmjm9ycjpemvkugr6er0o/rob_cifar_mart.pt?rlkey=zgswxy7u3nim3llf8mghusodu&dl=1
```
3. Download PRN_PGD.pth and put it under PRN/models/cifar10/:
```
wget -O PRN/models/cifar10/PRN_PGD.pth https://www.dropbox.com/scl/fi/3eddk3em3ip7ll36xfn48/PRN_PGD.pth?rlkey=x8lzj5iyblrpiksmqfa9uv2qa&dl=1
```
4. Download PRN_TRADES.pth and put it under PRN/models/cifar10/:
```
wget -O PRN/models/cifar10/PRN_TRADES.pth https://www.dropbox.com/scl/fi/s38pdrqvhdcnqxf3n1362/PRN_TRADES.pth?rlkey=y05yznhok4dsi00iwll0qpab2&dl=1
```




## How to use it

For ${\rm FAAL}_{\rm AT}$ with a Preact-ResNet-18 on CIFAR-10 under L_inf threat model (8/255), run codes as follows, 
```
python ft_prn.py --loss AT --distance kl  --pre-trained PGD  --gpuid 0 --out-dir PRN/results_at/FT_PRN_PGD --lr-schedule multistep --save-model --model PRN --epsilon 8 --steps 10  --alpha 2 --lr-max 0.01 --lr-min 0.0  --normalization std --epochs 2 --scale 0.5
```

For evaluation:
```
python evaluate_classwise.py --gpuid 0 --model PRN --pre-trained PGD --model-name PRN_PGD_0.01_0.5_both_best.pth  --epsilon 8  --normalization std --out-dir PRN/results_at/FT_PRN_PGD --batch-size 250
```


For ${\rm FAAL}_{\rm AT-AWP}$ with a Wide-ResNet34-10 on CIFAR-10 under L_inf threat model (8/255), run codes as follows, 
```
python ft_wrn.py --loss AT-AWP --distance kl  --pre-trained PGD  --gpuid 0 --out-dir WRN/results_atawp/FT_WRN_PGD --lr-schedule multistep --save-model --model WRN --epsilon 8 --steps 10  --alpha 2 --lr-max 0.01 --lr-min 0.0  --normalization std --epochs 2 --scale 0.5
```

For evaluation:
```
python evaluate_classwise.py --pre-trained PGD --gpuid 0 --model WRN --model-name WRN_PGD_0.01_0.5_worst_best.pth  --epsilon 8  --normalization std --out-dir WRN/results_atawp/FT_WRN_PGD --batch-size 250
```


## Or simply reproduce results in the paper via the below commands:
For Preact-ResNet-18:
```
python run_prn.py
```

For Wide-ResNet34-10:
```
python run_wrn.py
```

## Model Checkpoints for CIFAR10 

[TRAINED_FAAL_AT_CIFAR10.pth](https://www.dropbox.com/scl/fi/smx0gvb1goe1upi28svl0/TRAINED_FAAL_AT_CIFAR10.pth?rlkey=151swazoypbrtviyatqdsv3ba&st=ryjfvlqk&dl=1)

[TRAINED_FAAL_TRADERS_CIFAR10.pth](https://www.dropbox.com/scl/fi/h6176jjna5pw0sqouxxm7/TRAINED_FAAL_TRADERS_CIFAR10.pth?rlkey=w681sahq36gmhv8udg52bi81o&st=6mxqr1g6&dl=1)

[TRAINED_FAAL_TRADERS_AWP_CIFAR10.pth](https://www.dropbox.com/scl/fi/9vnsdzsv6jw39z09vzx75/TRAINED_FAAL_TRADERS_AWP_CIFAR10.pth?rlkey=ox3fsugzv35k3i23xfmrrvlwg&st=vzknvo2p&dl=1)

## Model Checkpoints for CIFAR100 
[TRAINED_FAAL_AT_CIFAR100.pth](https://www.dropbox.com/scl/fi/jhl34p17h0p4h026m4m5h/TRAINED_FAAL_AT_CIFAR100.pth?rlkey=v86n50mvkm5423o9c1qal8r0v&st=tlf8f4q2&dl=1)

[TRAINED_FAAL_TRADERS_CIFAR100.pth](https://www.dropbox.com/scl/fi/ziftxk6ad4fy0fr059r25/TRAINED_FAAL_TRADERS_CIFAR100.pth?rlkey=0t4biut1gyzctvjatp4gxddek&st=01dk5cvh&dl=1)


