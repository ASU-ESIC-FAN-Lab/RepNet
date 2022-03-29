import argparse
import os
import inspect
import sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
from torchvision import models

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'repnet'))

from ofa.utils.layers import LinearLayer
from ofa.model_zoo import proxylessnas_mobile
from ofa.imagenet_classification.run_manager import RunManager
from ofa.utils import init_models, download_url, list_mean
from ofa.utils import replace_conv2d_with_my_conv2d, replace_bn_with_gn
from repnet.data_providers import FGVCRunConfig
from repnet.utils import set_module_grad_status, enable_bn_update, enable_bias_update, enable_reprogram_update, weight_quantization
from repnet.utils import profile_memory_cost
from repnet.model import build_network_from_config,  \
                         ProgramNet, ProgramModule, ResProgramNet, MobiProgramNet
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--gpu', help='gpu available', default='0')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--manual_seed', default=0, type=int)

""" RunConfig: dataset related """
parser.add_argument('--dataset', type=str, default='flowers102', choices=[
    'aircraft', 'car', 'flowers102',
    'food101', 'cub200', 'pets',
    'cifar10', 'cifar100',
])
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--valid_size', type=float, default=None)

parser.add_argument('--n_worker', type=int, default=10)
parser.add_argument('--resize_scale', type=float, default=0.22)
parser.add_argument('--distort_color', type=str, default='tf', choices=['tf', 'torch', 'None'])
parser.add_argument('--image_size', type=int, default=224)

""" RunConfig: optimization related """
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--init_lr', type=float, default=0.05)
parser.add_argument('--init_lr_p', type=float, default=1e-3)

parser.add_argument('--lr_schedule_type', type=str, default='cosine')

parser.add_argument('--opt_type', type=str, default='adam', choices=['sgd', 'adam'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--no_decay_keys', type=str, default='bn#bias', choices=['None', 'bn', 'bn#bias', 'bias'])
parser.add_argument('--label_smoothing', type=float, default=0)

""" net config """
parser.add_argument('--net', type=str, default='proxyless_mobile', choices=['proxyless_mobile', 
        'specialized', 'resnet50','mobilenetv2','mobilenetv3'
])
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--ws_eps', type=float, default=1e-5)
parser.add_argument('--net_path', type=str, default=None)

""" transfer learning configs """
parser.add_argument('--transfer_learning_method', type=str, default='tiny-reprogram+bias', choices=[
	'full', 'bn+last', 'last',
	'tinytl-bias', 
    'tiny-reprogram', 'tiny-reprogram+bias',
])

""" weight quantization """
parser.add_argument('--frozen_param_bits', type=int, default=8)


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.path, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.path, 'args.txt'), 'w'), indent=4)
    writer = SummaryWriter(args.path)

    print(args)

    # setup transfer learning
    args.enable_feature_extractor_update = False
    args.enable_bn_update = False
    args.enable_bias_update = False
    args.enable_reprogram = False
    if args.transfer_learning_method == 'full':
        args.enable_feature_extractor_update = True
    elif args.transfer_learning_method == 'bn+last':
        args.enable_bn_update = True
    elif args.transfer_learning_method == 'last':
        pass
    elif args.transfer_learning_method == 'tinytl-bias':
        args.enable_bias_update = True
    elif args.transfer_learning_method == 'tiny-reprogram':
        args.enable_reprogram = True
    elif args.transfer_learning_method == 'tiny-reprogram+bias':
        args.enable_reprogram = True
        args.enable_bias_update = True  
    else:
        raise ValueError('Do not support %s' % args.transfer_learning_method)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.resume:
        args.manual_seed = int(time.time())  # set new manual seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # run config
    if isinstance(args.valid_size, float) and args.valid_size > 1:
        args.valid_size = int(args.valid_size)
    args.no_decay_keys = None if args.no_decay_keys == 'None' else args.no_decay_keys
    args.opt_param = {'momentum': args.momentum, 'nesterov': not args.no_nesterov}

    run_config = FGVCRunConfig(**args.__dict__)
    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    # network
    classification_head = []
    if args.net == 'proxyless_mobile':
        net = proxylessnas_mobile(pretrained=True)
        net.classifier = LinearLayer(
            net.classifier.in_features, run_config.data_provider.n_classes, dropout_rate=args.dropout)
        classification_head.append(net.classifier)
        init_models(classification_head)
           
        if args.enable_reprogram:
            net = ProgramNet(net, run_config.data_provider.active_img_size)
            # replace_bn_with_gn(net.sidemodules, gn_channel_per_group=8)
 
    elif args.net == 'resnet50':
        net = models.resnet50(pretrained=True)
        # net = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        #net = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        print(net)
        # for idx,(n,m) in enumerate(net.named_parameters()):
        #     print(idx,n,m.shape)
        # idx = 0
        # for n, m in net.named_children():
        #     print(n,m)
        net.fc = LinearLayer(
            net.fc.in_features, run_config.data_provider.n_classes, dropout_rate=0)

        classification_head.append(net.fc)
        init_models(classification_head)
        if args.enable_reprogram:
            net = ResProgramNet(net, run_config.data_provider.active_img_size)
            # replace_bn_with_gn(net.sidemodules, gn_channel_per_group=8)

    elif args.net == 'mobilenetv2':
        net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        for idx,(n,m) in enumerate(net.named_parameters()):
            print(idx,n,m.shape)
        idx = 0
        for n, m in net.named_children():
            print(n,m)
        net.classifier = LinearLayer(
            net.last_channel, run_config.data_provider.n_classes, dropout_rate=0.2)

        classification_head.append(net.classifier)
        init_models(classification_head)
        if args.enable_reprogram:
            net = MobiProgramNet(net, run_config.data_provider.active_img_size)
            # replace_bn_with_gn(net.sidemodules, gn_channel_per_group=8)

    elif args.net == 'mobilenetv3':
        net = models.mobilenet_v3_large(pretrained=True)
        for idx,(n,m) in enumerate(net.named_parameters()):
            print(idx,n,m.shape)
        idx = 0
        for n, m in net.named_children():
            print(n,m)
        net.classifier[3] = LinearLayer(
            net.classifier[3].in_features, run_config.data_provider.n_classes, dropout_rate=0.)

        classification_head.append(net.classifier)
        init_models(classification_head)
        if args.enable_reprogram:
            net = MobiProgramNet(net, run_config.data_provider.active_img_size)
            replace_bn_with_gn(net.sidemodules, gn_channel_per_group=8)

    # set transfer learning configs
    set_module_grad_status(net, args.enable_feature_extractor_update)
    set_module_grad_status(classification_head, True)
    if args.enable_bn_update:
        enable_bn_update(net)
    if args.enable_bias_update:
        enable_bias_update(net)
    if args.enable_reprogram:
        for m in net.modules():
            if isinstance(m, ProgramModule) or isinstance(m, nn.Linear):
                print('setting the grad to be True for {}'.format(m))
                set_module_grad_status(m, True)
   
    # weight quantization on frozen parameters
    if not args.resume and args.net == 'proxyless_mobile':
      weight_quantization(net, bits=args.frozen_param_bits, max_iter=20)

    # setup weight standardization
    if args.enable_reprogram:
        replace_conv2d_with_my_conv2d(net.sidemodules, args.ws_eps)

    # build run manager
    run_manager = RunManager(args.path, net, run_config, init=False)

    # profile memory cost
    require_backward = args.enable_feature_extractor_update or args.enable_bn_update or args.enable_bias_update \
                         or args.enable_reprogram 
    input_size = (1, 3, run_config.data_provider.active_img_size, run_config.data_provider.active_img_size)
    memory_cost, detailed_info = profile_memory_cost(
        net, input_size, require_backward, activation_bits=32, trainable_param_bits=32,
        frozen_param_bits=args.frozen_param_bits, batch_size=run_config.train_batch_size,
    )
    net_info = {
        'memory_cost': memory_cost / 1e6,
        'param_size': detailed_info['param_size'] / 1e6,
        'act_size': detailed_info['act_size'] / 1e6,
    }
    with open('%s/net_info.txt' % run_manager.path, 'a') as fout:
        fout.write(json.dumps(net_info, indent=4) + '\n')

    # information of parameters that will be updated via gradient
    run_manager.write_log('Updated params:', 'grad_params', False, 'w')
    for i, param_group in enumerate(run_manager.optimizer.param_groups):
        run_manager.write_log(
            'Group %d: %d params with wd %f' % (i + 1, len(param_group['params']), param_group['weight_decay']),
            'grad_params', True, 'a')
    for name, param in net.named_parameters():
        if param.requires_grad:
            run_manager.write_log('%s: %s' % (name, list(param.data.size())), 'grad_params', False, 'a')

    run_manager.save_config()
    if args.resume:
        run_manager.load_model()
    else:
        init_path = '%s/init' % args.path
        if os.path.isfile(init_path):
            checkpoint = torch.load(init_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            run_manager.network.load_state_dict(checkpoint)

    # train
    args.teacher_model = None
    run_manager.train(args, writer)
    # test
    # img_size, loss, acc1, acc5 = run_manager.validate_all_resolution(is_test=True)
    # log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f\t' % (list_mean(loss), list_mean(acc1), list_mean(acc5))
    # for i_s, v_a in zip(img_size, acc1):
    #     log += '(%d, %.3f), ' % (i_s, v_a)
    # run_manager.write_log(log, prefix='test')
    writer.close()