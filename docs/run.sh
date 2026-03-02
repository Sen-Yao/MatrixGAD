# 以下为复现实验所使用的超参数配置：

######################################## GGAD ########################################

# Amazon
python run.py --embedding_dim=300 --model_type=GGAD --margin_loss_weight=1 --warmup_updates=0 --num_epoch=200 --peak_lr=1e-3 --end_lr=1e-3 --train_rate 0.05  --dataset=Amazon 

# reddit
python run.py --embedding_dim=300 --model_type=GGAD --margin_loss_weight=1 --warmup_updates=0 --num_epoch=50 --peak_lr=1e-3 --end_lr=1e-3 --train_rate 0.05  --dataset=reddit 

# photo
python run.py --embedding_dim=300 --model_type=GGAD --margin_loss_weight=1 --warmup_updates=0 --num_epoch=50 --peak_lr=1e-3 --end_lr=1e-3 --train_rate 0.05  --dataset=photo 

# elliptic
# sweep jbg0lp3m
python run.py --embedding_dim=300 --model_type=GGAD --margin_loss_weight=1 --warmup_updates=0 --num_epoch=70 --peak_lr=1e-3 --end_lr=1e-3 --train_rate 0.05  --dataset=elliptic 

# t_finance
python run.py --embedding_dim=300 --model_type=GGAD --margin_loss_weight=1 --warmup_updates=0 --num_epoch=70 --peak_lr=1e-3 --end_lr=1e-3 --train_rate 0.05 --confidence_margin=0.7 --dataset=t_finance --device -1

# tolokers
# sweep 956275lb
python run.py --embedding_dim=300 --model_type=GGAD --margin_loss_weight=1 --warmup_updates=0 --num_epoch=50 --peak_lr=1e-3 --end_lr=1e-3 --train_rate 0.05 --confidence_margin=0.7 --dataset=tolokers 

# questions
python run.py --embedding_dim=300 --model_type=GGAD --margin_loss_weight=1 --warmup_updates=0 --num_epoch=70 --peak_lr=1e-3 --end_lr=1e-3 --train_rate 0.05  --confidence_margin=0.7 --dataset=questions 

######################################## GGADFormer ########################################

# Amazon
# sweep zev7xcg9
python run.py --dataset=Amazon --num_epoch=50 --peak_lr=5e-4 --batch_size=1024 --pp_k=5 --progregate_alpha=0.3

# reddit
# sweep ecftoff3
python run.py --dataset=reddit --GT_ffn_dim=64 --GT_num_heads=4 --GT_num_layers=2 --embedding_dim=256 --peak_lr=1e-4  --end_lr=5e-5 --num_epoch=200 --warmup_updates=50 --pp_k=1 --progregate_alpha=0.2

# photo
# sweep 4i1chaya
python run.py --dataset=photo --num_epoch=200 --peak_lr=3e-4 --batch_size=128

# elliptic
# sweep 2e1yh14
python run.py --dataset=elliptic --GT_ffn_dim=256 --GT_num_layers=3 --embedding_dim=256 --peak_lr=5e-4 --end_lr=1e-4 --num_epoch=100 --warmup_updates=50 --pp_k=7 --progregate_alpha=0.5 --con_loss_weight=20 --confidence_margin=0.3 --batch_size=8192 --rec_loss_weight=1

# t_finance
# sweep sy6t99m6
python run.py --dataset=t_finance --GT_ffn_dim=256 --GT_num_layers=3 --embedding_dim=256   --peak_lr=5e-4 --end_lr=1e-4 --num_epoch=80 --warmup_updates=50 --pp_k=7 --progregate_alpha=0.5 --con_loss_weight=20 --confidence_margin=0.3 --batch_size=8192

# tolokers
# sweep sy6t99m6
python run.py --dataset=tolokers --GT_ffn_dim=256 --GT_num_layers=3 --embedding_dim=256   --peak_lr=1e-4 --end_lr=1e-4 --num_epoch=80 --warmup_updates=70 --pp_k=3 --progregate_alpha=0.3 --con_loss_weight=20 --confidence_margin=0.3 --batch_size=1024

# questions
# sweep 0b9r9tak
python run.py --dataset=questions --GT_ffn_dim=256 --GT_num_layers=3 --embedding_dim=256 --peak_lr=2e-4 --end_lr=1e-4 --num_epoch=100 --warmup_updates=70 --pp_k=3 --progregate_alpha=0.3 --con_loss_weight=20 --confidence_margin=0.3 --batch_size=2048 --rec_loss_weight=1 