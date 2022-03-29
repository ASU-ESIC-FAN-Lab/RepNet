python tiny_train_repnet.py --transfer_learning_method tiny-reprogram+bias\
    --train_batch_size 8 --test_batch_size 100 --image_size 224\
    --n_epochs 1 --init_lr 2e-4 --init_lr_p 1e-3 --opt_type adam \
    --label_smoothing 0.7 --distort_color None --frozen_param_bits 8 \
    --gpu 0 --dataset flowers102 --path .exp/batch8/flowers102_proxyless/opt2-R224-repnet-trainmemtest