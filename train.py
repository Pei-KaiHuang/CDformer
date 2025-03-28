import torch
from models.ViT import *
from models.mobilevit import *
import torch.optim as optim
import torch.nn as nn
import itertools
import numpy as np
import os
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import logging
from pytz import timezone
from datetime import datetime
import timm
import sys
import torchvision.transforms as T
from thop import profile

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()

seed = 3407 #666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def imshow_np(img, filename):
    height, width, depth = img.shape
    if depth == 1:
        img = img[:, :, 0]
    plt.imshow(img)
    plt.axis("off")
    plt.savefig("./save_image_oulu/dynamic/" + filename + ".png",
                bbox_inches='tight', pad_inches=0)
    plt.close()


def get_data_loader(data_path="", data_path2="live", batch_size=5, shuffle=True, drop_last=True):
    # data path
    data = None
    live_spoof_label = None


    if data_path2 == "live":
        data = np.load(data_path)

        live_spoof_label = np.ones(len(data), dtype=np.int64)
    else:
        print_data = np.load(data_path)
        replay_data = np.load(data_path2)
        data = np.concatenate((print_data, replay_data), axis=0)

        print_lab = np.zeros(len(print_data), dtype=np.int64)
        replay_lab = np.ones(len(replay_data), dtype=np.int64) * 2

        live_spoof_label = np.zeros(len(data), dtype=np.int64)

    # dataset
    trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data, (0, 3, 1, 2))),
                                              torch.tensor(live_spoof_label))
    # free memory
    import gc
    del data

    gc.collect()
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    return data_loader


def get_inf_iterator(data_loader):
    while True:
        for images, live_spoof_labels in data_loader:
            yield (images, live_spoof_labels)


def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


device_id = 'cuda:0'
results_filename = 'O_CDformer'
results_path = "/shared/Jxchong/cdformer/" + results_filename
batch_size = 5

file_handler = logging.FileHandler(filename='/home/Jxchong/Transformer/logger/'+ results_filename +'_train.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

# replay casia Oulu MSU
dataset1 = "casia"
dataset2 = "replay"
dataset3 = "MSU"
logging.info(f"Train on {dataset1}, {dataset2}, {dataset3}")

# image shape: torch.Size([3, 256, 256])
live_path1 = '/shared/domain-generalization/' + dataset1 + '_images_live.npy'
live_path2 = '/shared/domain-generalization/' + dataset2 + '_images_live.npy'
live_path3 = '/shared/domain-generalization/' + dataset3 + '_images_live.npy'

print_path1 = '/shared/domain-generalization/' + dataset1 + '_print_images.npy'
print_path2 = '/shared/domain-generalization/' + dataset2 + '_print_images.npy'
print_path3 = '/shared/domain-generalization/' + dataset3 + '_print_images.npy'

replay_path1 = '/shared/domain-generalization/' + dataset1 + '_replay_images.npy'
replay_path2 = '/shared/domain-generalization/' + dataset2 + '_replay_images.npy'
replay_path3 = '/shared/domain-generalization/' + dataset3 + '_replay_images.npy'


Fas_Net = vit_base_patch16_224(pretrained=True, num_classes=2).to(device_id)
# Fas_Net = Mobilemain().to(device_id)

criterionCls = nn.CrossEntropyLoss().to(device_id)
criterionMSE = torch.nn.MSELoss().to(device_id)
optimizer = optim.RAdam(Fas_Net.parameters(), lr=1e-5) 
Fas_Net.train()

model_save_step = 20
model_save_epoch = 1

save_index = 0

data1_real = get_data_loader(data_path=live_path1, data_path2="live",
                             batch_size=batch_size, shuffle=True)
data2_real = get_data_loader(data_path=live_path2, data_path2="live",
                             batch_size=batch_size, shuffle=True)
data3_real = get_data_loader(data_path=live_path3, data_path2="live",
                             batch_size=batch_size, shuffle=True)
data1_fake = get_data_loader(data_path=print_path1, data_path2=replay_path1,
                             batch_size=batch_size, shuffle=True)
data2_fake = get_data_loader(data_path=print_path2, data_path2=replay_path2,
                             batch_size=batch_size, shuffle=True)
data3_fake = get_data_loader(data_path=print_path3, data_path2=replay_path3,
                             batch_size=batch_size, shuffle=True)

iternum = max(len(data1_real), len(data2_real),
              len(data3_real), len(data1_fake),
              len(data2_fake), len(data3_fake))
log_step = 20
logging.info(f"iternum={iternum}")
data1_real = get_inf_iterator(data1_real)
data2_real = get_inf_iterator(data2_real)
data3_real = get_inf_iterator(data3_real)
data1_fake = get_inf_iterator(data1_fake)
data2_fake = get_inf_iterator(data2_fake)
data3_fake = get_inf_iterator(data3_fake)
 
T_transform = torch.nn.Sequential(
        T.Pad(40, padding_mode="symmetric"),
        T.RandomRotation(30), 
        T.RandomHorizontalFlip(p=0.5),
        T.CenterCrop(224),
)


for epoch in range(200):

    for step in range(iternum):
        # ============ one batch extraction ============#
        img1_real, ls_lab1_real = next(data1_real)
        img1_fake, ls_lab1_fake = next(data1_fake)

        img2_real, ls_lab2_real = next(data2_real)
        img2_fake, ls_lab2_fake = next(data2_fake)

        img3_real, ls_lab3_real = next(data3_real)
        img3_fake, ls_lab3_fake = next(data3_fake)

        # ============ one batch collection ============#
        catimg = torch.cat([img1_real, img2_real, img3_real,
                            img1_fake, img2_fake, img3_fake], 0).to(device_id)
        ls_lab = torch.cat([ls_lab1_real, ls_lab2_real, ls_lab3_real,
                            ls_lab1_fake, ls_lab2_fake, ls_lab3_fake], 0).to(device_id)

        batchidx = list(range(len(catimg)))
        random.shuffle(batchidx)

        img_rand = catimg[batchidx, :]
        ls_lab_rand = ls_lab[batchidx]

        img_rand = T_transform(img_rand) 
        pred = Fas_Net(NormalizeData_torch(img_rand)).to(device_id)

        Loss_cls = criterionCls(pred.squeeze(), ls_lab_rand)

        optimizer.zero_grad()
        Loss_cls.backward()
        optimizer.step()


        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d]  Loss_cls %.8f'
                  % (epoch, step, Loss_cls.item()))

        if ((step + 1) % model_save_step == 0):
            mkdir(results_path)
            save_index += 1
            torch.save(Fas_Net.state_dict(), os.path.join(results_path,
                                                          "FASNet-{}.tar".format(save_index)))

    if ((epoch + 1) % model_save_epoch == 0):
        mkdir(results_path)
        save_index += 1
        torch.save(Fas_Net.state_dict(), os.path.join(results_path,
                                                      "FASNet-{}.tar".format(save_index)))
