import os 

for model in ['WRN']:
    for base in ['PGD']:
        for dis in ['kl']:
            for lr in [0.01]:    
                for scale in [0.5]: 
                    for norm in ['std']:   
                        os.system('python ft_wrn.py --loss AT --distance '+dis+'  --pre-trained '+base+'  --gpuid 0 --out-dir '+model+'/results_at/FT_'+model+'_'+base+' --lr-schedule multistep --save-model --model '+model+' --epsilon 8 --steps 10  --alpha 2 --lr-max '+str(lr)+' --lr-min 0.0  --normalization '+norm+' --epochs 2 --scale '+str(scale))
                        os.system('python evaluate_classwise.py --pre-trained '+base+' --gpuid 0 --model '+model+' --model-name '+model+'_'+base+'_'+str(lr)+'_'+str(scale)+'_worst_best.pth  --epsilon 8  --normalization '+norm+' --out-dir '+model+'/results_at/FT_'+model+'_'+base+' --batch-size 250')
                      
for model in ['WRN']:
    for base in ['TRADES','MART','AWP']:
        for dis in ['kl']:
            for lr in [0.01]:    
                for scale in [0.5]: 
                    for norm in ['01']:   
                        os.system('python ft_wrn.py --loss AT --distance '+dis+'  --pre-trained '+base+'  --gpuid 0 --out-dir '+model+'/results_at/FT_'+model+'_'+base+' --lr-schedule multistep --save-model --model '+model+' --epsilon 8 --steps 10  --alpha 2 --lr-max '+str(lr)+' --lr-min 0.0  --normalization '+norm+' --epochs 2 --scale '+str(scale))
                        os.system('python evaluate_classwise.py --pre-trained '+base+' --gpuid 0 --model '+model+' --model-name '+model+'_'+base+'_'+str(lr)+'_'+str(scale)+'_worst_best.pth  --epsilon 8  --normalization '+norm+' --out-dir '+model+'/results_at/FT_'+model+'_'+base+' --batch-size 250')
                       

for model in ['WRN']:
    for base in ['PGD']:
        for dis in ['kl']:
            for lr in [0.01]:    
                for scale in [0.5]: 
                    for norm in ['std']:   
                        os.system('python ft_wrn.py --loss AT-AWP --distance '+dis+'  --pre-trained '+base+'  --gpuid 0 --out-dir '+model+'/results_atawp/FT_'+model+'_'+base+' --lr-schedule multistep --save-model --model '+model+' --epsilon 8 --steps 10  --alpha 2 --lr-max '+str(lr)+' --lr-min 0.0  --normalization '+norm+' --epochs 2 --scale '+str(scale))
                        os.system('python evaluate_classwise.py --pre-trained '+base+' --gpuid 0 --model '+model+' --model-name '+model+'_'+base+'_'+str(lr)+'_'+str(scale)+'_worst_best.pth  --epsilon 8  --normalization '+norm+' --out-dir '+model+'/results_atawp/FT_'+model+'_'+base+' --batch-size 250')
                      
for model in ['WRN']:
    for base in ['TRADES','MART','AWP']:
        for dis in ['kl']:
            for lr in [0.01]:    
                for scale in [0.5]: 
                    for norm in ['01']:   
                        os.system('python ft_wrn.py --loss AT-AWP --distance '+dis+'  --pre-trained '+base+'  --gpuid 0 --out-dir '+model+'/results_atawp/FT_'+model+'_'+base+' --lr-schedule multistep --save-model --model '+model+' --epsilon 8 --steps 10  --alpha 2 --lr-max '+str(lr)+' --lr-min 0.0  --normalization '+norm+' --epochs 2 --scale '+str(scale))
                        os.system('python evaluate_classwise.py --pre-trained '+base+' --gpuid 0 --model '+model+' --model-name '+model+'_'+base+'_'+str(lr)+'_'+str(scale)+'_worst_best.pth  --epsilon 8  --normalization '+norm+' --out-dir '+model+'/results_atawp/FT_'+model+'_'+base+' --batch-size 250')
           