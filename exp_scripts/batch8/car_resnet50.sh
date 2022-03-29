python tiny_train_repnet.py --transfer_learning_method tiny-reprogram+bias \
    --net resnet50 \
    --train_batch_size 8 --test_batch_size 100 \
    --n_epochs 50 --init_lr 8e-4 --init_lr_p 1e-3 --opt_type adam \
    --label_smoothing 0.7 --distort_color torch --frozen_param_bits 8 \
    --gpu 1 --dataset car --path .exp/batch8/car/repnet-R224    