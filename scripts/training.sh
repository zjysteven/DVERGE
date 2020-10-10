GPUID=7

cd ..

# DVERGE training
python train/train_dverge.py --gpu $GPUID --model-num 3 --distill-eps 0.07 --distill-alpha 0.007

# Baseline training
#python train/train_baseline.py --gpu $GPUID --model-num 3

# ADP training
#python train/train_adp.py --gpu $GPUID --model-num 3

# GAL training
#python train/train_gal.py --gpu $GPUID --model-num 3