python tiny_train_repnet.py --transfer_learning_method tiny-reprogram+bias \
    --train_batch_size 8 --test_batch_size 100 --image_size 224\
    --n_epochs 50 --init_lr 1e-4 --init_lr_p 1e-4 --opt_type adam \
    --label_smoothing 0.7 --distort_color None --frozen_param_bits 8 \
    --gpu 2 --dataset pets --path .exp/batch8/pets_proxyless/repnet-R224-all-1e-4-bn