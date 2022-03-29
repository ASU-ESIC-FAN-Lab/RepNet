python tiny_fine_tune.py --transfer_learning_method tiny-reprogram \
    --train_batch_size 256 --test_batch_size 500 \
    --n_epochs 20 --init_lr 0.05  --opt_type sgd --frozen_param_bits 8 \
    --gpu 0,1,2 --data imagenet --path .exp/batch8/fine_tune/proxyless_mobile