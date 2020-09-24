a = "./proxyless_nas/config/net0.config"
print(a[-11:-7])
import os
model_path = './save/student_model'
model_name = 'S:b:{}_{}'.format(0, 1)
save_folder = os.path.join(model_path, model_name)

print(str(save_folder))
save_file = os.path.join(save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=1))
print(str(save_file))
import torch
from efficientnet_pytorch import EfficientNet
from proxyless_nas.jj import get_proxyless_model
model_s = get_proxyless_model(net_config_path="./config/net0.config")
model_t = EfficientNet.from_pretrained('efficientnet-b0',
                                       weights_path='./pretrain_efficientNet/pretrain_efficientNet.pth')
data = torch.randn(2, 3, 224, 224)
model_s.eval()
model_t.eval()
feat_s, ls = model_t(data, is_feat=True)
feat_t, l = model_t(data, is_feat=True)
print(l.shape)
print(ls.shape)

if os.path.exists('./save/log/'):
        pass
else:
    os.mkdir('./save/log/')