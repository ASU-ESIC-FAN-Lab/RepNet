python tiny_train_mask.py --transfer_learning_method tiny-reprogram \
    --train_batch_size 8 --test_batch_size 100 \
    --n_epochs 50 --init_lr 0.5e-4 --init_lr_p 2e-4 --init_lr_m 0.5e-4 --opt_type adam \
    --label_smoothing 0.7 --distort_color None --frozen_param_bits 8 \
    --gpu 0 --dataset pets --path .exp/batch8/pets/opt2-0.5e-4+mask_133_thres_0