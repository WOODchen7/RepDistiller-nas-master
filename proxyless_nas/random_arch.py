from .nas_modules import *
import random
import json
import copy
from functools import partial
def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def proxyless_base(pretrained=True, net_config=None, net_weight=None):
    assert net_config is not None, "Please input a network config"
    net_config_path = download_url(net_config)
    print(net_config_path)
    net_config_json = json.load(open(net_config_path, 'r'))
    net_config_json_t = copy.deepcopy(net_config_json)
    if net_config_json['name'] == ProxylessNASNets.__name__:
        net = ProxylessNASNets.build_from_config(net_config_json)
    else:
        print("not ProxylessNASNets")
        exit(0)

    if 'bn' in net_config_json:
        net.set_bn_param(
            bn_momentum=net_config_json['bn']['momentum'],
            bn_eps=net_config_json['bn']['eps'])

    if pretrained:
        print(pretrained)
        assert net_weight is not None, "Please specify network weights"
        init_path = download_url(net_weight)
        init = torch.load(init_path, map_location='cpu')
        net.load_state_dict(init['state_dict'])

    return net,net_config_json_t

proxyless_gpu = partial(
        proxyless_base,
        net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_gpu.config",
        net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_gpu.pth")
proxyless_cpu = partial(
    proxyless_base,
    net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_cpu.config",
    net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_cpu.pth"
)
net, config = proxyless_gpu(pretrained = False)

print(config)
conv_candidates = [
            '3x3_MBConv3', '3x3_MBConv6',
            '5x5_MBConv3', '5x5_MBConv6',
            '7x7_MBConv3', '7x7_MBConv6',
        ]
width_stages='32,56,112,128,256,432'
n_cell_stages='4,4,4,4,4,1'
stride_stages='2,2,2,1,2,1'
width_stages = [int(val) for val in width_stages.split(',')]
n_cell_stages = [int(val) for val in n_cell_stages.split(',')]
stride_stages = [int(val) for val in stride_stages.split(',')]

print("#####################################")
print(0, config['blocks'][0])
k = 1
for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
    for i in range(n_cell):
        print(k, config['blocks'][k])
        k = k+1


print("######################3")
for step in range(10):
    nnum = 1
    input_channel = 24

    for i in range(len(width_stages)):
        width_stages[i] = make_divisible(width_stages[i] * 1, 8)
    print(width_stages)
    print(n_cell_stages)
    print(stride_stages)
    print(input_channel)
    print(0, config['blocks'][0])
    for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
        for i in range(n_cell):
            if i == 0:
                stride = s
            else:
                stride = 1
            # conv
            if stride == 1 and input_channel == width:
                modified_conv_candidates = conv_candidates + ['Zero']
            else:
                modified_conv_candidates = conv_candidates

            # shortcut
            if stride == 1 and input_channel == width:
                shortcut = IdentityLayer(input_channel, input_channel)
            else:
                shortcut = None

            conv_op = random.randint(0, len(modified_conv_candidates) - 1)

            if conv_op == 6:
                config['blocks'][nnum]['mobile_inverted_conv'] = {
                    "name": "ZeroLayer",
                    "stride": 1
                }
            else:
                config['blocks'][nnum]['mobile_inverted_conv'] = {
                    "name": "MBInvertedConvLayer",
                    'in_channels': input_channel,
                    'out_channels': width,
                    'kernel_size': int(modified_conv_candidates[conv_op][0]),
                    'stride': stride,
                    'expand_ratio': int(modified_conv_candidates[conv_op][-1])}
            print(nnum, config['blocks'][nnum])
            input_channel = width
            nnum = nnum + 1

    json.dump(config, open(os.path.join("./config", '/net'+str(step)+'.config'), 'w'), indent=4)


