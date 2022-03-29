python tiny_train_mask.py --transfer_learning_method tiny-reprogram \
    --train_batch_size 8 --test_batch_size 100 \
    --n_epochs 50 --init_lr 2e-4 --init_lr_p 1e-3 --opt_type adam \
    --label_smoothing 0.1 --distort_color None --frozen_param_bits 8 \
    --gpu 2 --dataset food101 --path .exp/batch8/food101/opt2-1e-3-6block-mask-thre0