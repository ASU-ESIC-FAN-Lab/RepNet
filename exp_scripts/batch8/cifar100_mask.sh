python tiny_train_mask.py --transfer_learning_method tiny-reprogram \
    --train_batch_size 8 --test_batch_size 100 \
    --n_epochs 50 --init_lr 4e-4 --init_lr_p 2e-3 --opt_type adam \
    --label_smoothing 0.1 --distort_color None --frozen_param_bits 8 \
    --gpu 0 --dataset cifar100 --path .exp/batch8/cifar100/opt2-2e-6blocks-mask-thre0