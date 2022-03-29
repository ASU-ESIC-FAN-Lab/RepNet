python tiny_train_test.py --transfer_learning_method tiny-reprogram\
    --net mobilenetv2 \
    --train_batch_size 8 --test_batch_size 100 --image_size 320\
    --n_epochs 50 --init_lr 16e-4 --init_lr_p 3e-3 --opt_type adam \
    --label_smoothing 0.7 --distort_color torch --frozen_param_bits 8 \
    --gpu 0 --dataset aircraft --path .exp/batch8/aircraft_mobilenetv2_repro/opt2-R224-Dual_connection-test
