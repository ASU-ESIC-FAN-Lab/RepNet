python tiny_train_test.py --transfer_learning_method tiny-reprogram+bias \
    --net mobilenetv3 \
    --train_batch_size 8 --test_batch_size 100 --image_size 224\
    --n_epochs 50 --init_lr 1e-3 --init_lr_p 2e-3 --opt_type adam \
    --label_smoothing 0.7 --distort_color torch --frozen_param_bits 8 \
    --gpu 1 --dataset aircraft --path .exp/batch8/aircraft_mobilenetv3/opt2-R224-1e-3-2e-3+bias
