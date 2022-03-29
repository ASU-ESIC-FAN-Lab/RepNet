python tiny_train_test.py --transfer_learning_method tiny-reprogram\
    --net mobilenetv3 \
    --train_batch_size 8 --test_batch_size 100 --image_size 224\
    --n_epochs 50 --init_lr 6e-4  --init_lr_p 3e-3 --opt_type adam \
    --label_smoothing 0.7 --distort_color None --frozen_param_bits 8 \
    --gpu 2 --dataset cub200 --path .exp/batch8/cub200_mobilenetv3_repro/opt2-R224