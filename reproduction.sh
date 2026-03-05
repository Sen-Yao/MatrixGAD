# MatrixGAD

## Amazon

# Cycle Density: 367.2513

CUDA_VISIBLE_DEVICES=1 python run.py --batch_size=1024 --dataset=Amazon --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

## Reddit

# Cycle Density: 0.0745

CUDA_VISIBLE_DEVICES=1 python run.py --batch_size=32768 --dataset=elliptic --end_lr=0.0003 --lambda_rec_emb=2 --num_epoch=150 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=6 --progregate_alpha=0.6 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=20 --seed=1 --train_rate=0.05 --warmup_updates=50

## photo

# Cycle Density: 14.8065

CUDA_VISIBLE_DEVICES=1 python run.py --batch_size=128 --dataset=photo --end_lr=1e-4 --lambda_rec_emb=0.1 --num_epoch=200 --peak_lr=5e-4 --pp_k=6 --progregate_alpha=0.2 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=2 --train_rate=0.05 --warmup_updates=50

## Elliptic

# AUC=0.7046, AP=0.1723
# https://wandb.ai/HCCS/MatrixGAD/runs/uzvt3v4n

CUDA_VISIBLE_DEVICES=2 python run.py --batch_size=32768 --dataset=elliptic --end_lr=0.0003 --lambda_rec_emb=2 --num_epoch=300 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.1 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=20 --seed=0 --train_rate=0.1 --warmup_updates=50

## T-Finance

# Cycle Density: 538.2318

CUDA_VISIBLE_DEVICES=1 python run.py --batch_size=8192 --dataset=t_finance --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=40 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.3 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.5 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

## Tolokers

# Cycle Density: 21.0702

python run.py --batch_size=1024 --dataset=tolokers --end_lr=0.0001 --lambda_rec_emb=0.5 --num_epoch=70 --outlier_beta=0.3 --peak_lr=0.0001 --pp_k=3 --progregate_alpha=0.3 --rec_loss_weight=0.1 --ring_R_max=0.5 --ring_R_min=0.5 --ring_loss_weight=20 --seed=0 --train_rate=0.05 --warmup_updates=50

CUDA_VISIBLE_DEVICES=1 python run.py --batch_size=32768 --dataset=elliptic --end_lr=0.0003 --lambda_rec_emb=2 --num_epoch=300 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.1 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=20 --seed=0 --train_rate=0.1 --warmup_updates=50 --GT_num_heads=3