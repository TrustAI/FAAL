import os 

for model in ['PRN']:
    for base in ['PGD','TRADES']:
        for dis in ['kl']:
            for lr in [0.01]:    
                for scale in [0.5]: 
                    for norm in ['std']:   
                        os.system('python ft_prn.py --loss AT --distance '+dis+'  --pre-trained '+base+'  --gpuid 0 --out-dir PRN/results_at/FT_'+model+'_'+base+' --lr-schedule multistep --save-model --model '+model+' --epsilon 8 --steps 10  --alpha 2 --lr-max '+str(lr)+' --lr-min 0.0  --normalization '+norm+' --epochs 2 --scale '+str(scale))
                        os.system('python evaluate_classwise.py --gpuid 0 --model '+model+' --pre-trained '+base+' --model-name '+model+'_'+base+'_'+str(lr)+'_'+str(scale)+'_worst_best.pth  --epsilon 8  --normalization '+norm+' --out-dir PRN/results_at/FT_'+model+'_'+base+' --batch-size 250')



for model in ['PRN']:
    for base in ['PGD','TRADES']:
        for dis in ['kl']:
            for lr in [0.01]:    
                for scale in [0.5]: 
                    for norm in ['std']:   
                        os.system('python ft_prn.py --loss AT-AWP --distance '+dis+'  --pre-trained '+base+'  --gpuid 0 --out-dir PRN/results_atawp/FT_'+model+'_'+base+' --lr-schedule multistep --save-model --model '+model+' --epsilon 8 --steps 10  --alpha 2 --lr-max '+str(lr)+' --lr-min 0.0  --normalization '+norm+' --epochs 2 --scale '+str(scale))
                        os.system('python evaluate_classwise.py --gpuid 0 --model '+model+' --pre-trained '+base+' --model-name '+model+'_'+base+'_'+str(lr)+'_'+str(scale)+'_worst_best.pth  --epsilon 8  --normalization '+norm+' --out-dir PRN/results_atawp/FT_'+model+'_'+base+' --batch-size 250')
