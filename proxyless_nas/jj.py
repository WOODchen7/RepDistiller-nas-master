from functools import partial
import json

import torch
import os
import sys
from urllib.request import urlretrieve
from .nas_modules import ProxylessNASNets
from .cifar_modules import PyramidTreeNet

def download_url(url, model_dir="~/.torch/proxyless_nas", overwrite=False):
    model_dir = os.path.expanduser(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) or overwrite:
        os.makedirs(model_dir, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file

def proxyless_base(pretrained=True, net_config=None, net_weight=None):
    assert net_config is not None, "Please input a network config"
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))
    if net_config_json['name'] == ProxylessNASNets.__name__:
        net = ProxylessNASNets.build_from_config(net_config_json)
    else:
        net = PyramidTreeNet.build_from_config(net_config_json)

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

    return net

def proxyless_base_ex(pretrained=True, net_config_path=None, net_weight=None):
    assert net_config_path is not None, "Please input a network config"
    #net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))
    if net_config_json['name'] == ProxylessNASNets.__name__:
        net = ProxylessNASNets.build_from_config(net_config_json)
    else:
        net = PyramidTreeNet.build_from_config(net_config_json)

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

    return net

proxyless_cpu = partial(
    proxyless_base,
    net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_cpu.config",
    net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_cpu.pth"
)

proxyless_gpu = partial(
    proxyless_base,
    net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_gpu.config",
    net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_gpu.pth")

proxyless_mobile = partial(
    proxyless_base,
    net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_mobile.config",
    net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_mobile.pth")

proxyless_mobile_14 = partial(
    proxyless_base,
    net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_mobile_14.config",
    net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_mobile_14.pth")

proxyless_cifar = partial(
    proxyless_base,
    net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_cifar.config",
    net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_cifar.pth"
)

if __name__ == '__main__':

    print(ProxylessNASNets.__name__)
    proxyless_gpu = partial(
        proxyless_base,
        net_config="https://file.lzhu.me/projects/proxylessNAS/proxyless_gpu.config",
        net_weight="https://file.lzhu.me/projects/proxylessNAS/proxyless_gpu.pth")
    net = proxyless_gpu(pretrained = False)


    data = torch.randn(1, 3, 224, 224)
    feat_t= net(data)
    print("##############")
    for step in range(10):
        net2 = proxyless_base_ex(pretrained=False, net_config_path="./config/net"+str(step)+".config")
        feat_s= net2(data)
        print(feat_s)
def get_proxyless_model(net_config_path):
    return proxyless_base_ex(pretrained=False, net_config_path=net_config_path)


