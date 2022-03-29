for lr in 3e-4 1e-3 2e-3
do
    python tiny_train_repnet.py --transfer_learning_method tiny-reprogram+bias \
        --train_batch_size 8 --test_batch_size 100 --image_size 224\
        --n_epochs 50 --init_lr 3e-4 --init_lr_p $lr --opt_type adam \
        --label_smoothing 0.1 --distort_color None --frozen_param_bits 8 \
        --gpu 1 --dataset cifar100 --path .exp/batch8/cifar100/opt2-R224-test-$lr
done