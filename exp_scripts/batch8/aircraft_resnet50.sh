python tiny_train_repnet.py --transfer_learning_method tiny-reprogram+bias \
    --net resnet50 \
    --train_batch_size 8 --test_batch_size 100 --image_size 224\
    --n_epochs 50 --init_lr 16e-4 --init_lr_p 3e-3 --opt_type adam \
    --label_smoothing 0.7 --distort_color torch --frozen_param_bits 8 \
    --gpu 2 --dataset aircraft --path .exp/batch8/aircraft_resnet50/repnet-R224
