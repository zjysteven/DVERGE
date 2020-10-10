GPUID=6

cd ..

# evaluation of black-box robustness
# remember to first download and put transfer_adv_examples/
# under ../data/
python eval/eval_bbox.py \
    --gpu $GPUID \
    --model-file checkpoints/dverge/seed_0/3_ResNet20_eps_0.07/epoch_200.pth \
    --folder transfer_adv_examples \
    --steps 100 \
    --save-to-csv

# evaluation of white-box robustness
#python eval/eval_wbox.py \
#    --gpu $GPUID \
#    --model-file checkpoints/dverge/seed_0/3_ResNet20_eps_0.07/epoch_200.pth \
#    --steps 50 \
#    --random-start 5 \
#    --save-to-csv

# evaluation of transferability
#python eval/eval_transferability.py \
#    --gpu $GPUID \
#    --model-file checkpoints/dverge/seed_0/3_ResNet20_eps_0.07/epoch_200.pth \
#    --steps 50 \
#    --random-start 5 \
#    --save-to-file

# evaluation of diversity
#python eval/eval_diversity.py \
#    --gpu $GPUID \
#    --model-file checkpoints/dverge/seed_0/3_ResNet20_eps_0.07/epoch_200.pth \
#    --save-to-file