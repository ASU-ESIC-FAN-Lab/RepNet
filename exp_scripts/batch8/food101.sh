python tiny_train_test.py --transfer_learning_method tiny-reprogram+bias \
    --train_batch_size 8 --test_batch_size 100 --image_size 320\
    --n_epochs 50 --init_lr 2e-4 --init_lr_p 1e-3 --opt_type adam \
    --label_smoothing 0.1 --distort_color None --frozen_param_bits 8 \
    --gpu 0 --dataset food101 --path .exp/batch8/food101/opt2-R320-gn+bias-proxyless-6blocks-input