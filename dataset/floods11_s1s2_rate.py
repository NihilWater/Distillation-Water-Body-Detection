import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image
import csv
import numpy as np
import rasterio
import os

means_s1 = [0.6851, 0.5235]
stds_s1  = [0.0820, 0.1102]

means_s2 = [1647.5275, 1409.1735, 1373.1986, 1220.8898, 1470.2974, 2395.1107, 2857.3290, 2633.3441, 3089.7533, 487.1713, 60.5535, 2033.6098, 1178.3920]
stds_s2  = [659.5315, 714.0080, 712.1827, 851.5684, 749.7281, 863.7205, 1017.6825, 968.1219, 1131.4698, 327.6270, 131.5372, 958.7381, 757.5991]
'''
以上的均值和方差是使用 preprocess/calc_mean_std_on_s2 获取的
'''

def getArrFlood(fname):
    return rasterio.open(fname).read()

def download_flood_water_data_from_list_with_rate(l):
    i = 0
    flood_data = []
    for (im_s1_fname, im_s2_fname, mask_fname) in l:

        if not os.path.exists(im_s1_fname):
            continue
        if not os.path.exists(im_s2_fname):
            continue

        arr_s1_x = np.nan_to_num(getArrFlood(im_s1_fname))
        arr_s2_x = np.nan_to_num(getArrFlood(im_s2_fname))

        arr_y = getArrFlood(mask_fname)
        arr_y[arr_y == -1] = 255

        arr_s1_x = np.concatenate((arr_s1_x, arr_s1_x[0:1, ...] / arr_s1_x[1:2, ...]), axis=0)
        arr_s1_x[2] = arr_s1_x[2].clip(-2, 2)
        arr_s1_x = np.nan_to_num(arr_s1_x)


        arr_s1_x = np.clip(arr_s1_x, -50, 1)
        arr_s1_x = (arr_s1_x + 50) / 51

        if i % 100 == 0:
            print(im_s1_fname, im_s2_fname, mask_fname)
        i += 1
        flood_data.append((arr_s1_x, arr_s2_x, arr_y))

    return flood_data

class FloodsS1S2WithRateDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)

def processAndAugmentS1S2_with_rate(data):
    (x1, x2, y) = data
    im1, im2, label = x1.copy(), x2.copy(), y.copy()

    # convert to PIL for easier transforms

    im_vv = Image.fromarray(im1[0])
    im_vh = Image.fromarray(im1[1])
    im_rate = Image.fromarray(im1[2])
    im_swir = Image.fromarray(im2[12])
    im_nir = Image.fromarray(im2[7])
    im_green =Image.fromarray(im2[3])
    im_blue  = Image.fromarray(im2[2])

    label = Image.fromarray(label.squeeze())

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(im_swir, (256, 256))
    im_vv = F.crop(im_vv, i, j, h, w)
    im_vh = F.crop(im_vh, i, j, h, w)
    im_rate = F.crop(im_rate, i, j, h, w)
    im_swir =  F.crop(im_swir, i, j, h, w)
    im_nir =  F.crop(im_nir, i, j, h, w)
    im_green =  F.crop(im_green, i, j, h, w)
    im_blue  = F.crop(im_blue, i, j, h, w)

    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im_vv = F.hflip(im_vv)
        im_vh = F.hflip(im_vh)
        im_rate = F.hflip(im_rate)
        im_swir = F.hflip(im_swir)
        im_nir = F.hflip(im_nir)
        im_green = F.hflip(im_green)
        im_blue = F.hflip(im_blue)
        label = F.hflip(label)
    if random.random() > 0.5:
        im_vv = F.vflip(im_vv)
        im_vh = F.vflip(im_vh)
        im_rate = F.vflip(im_rate)
        im_swir = F.vflip(im_swir)
        im_nir = F.vflip(im_nir)
        im_green = F.vflip(im_green)
        im_blue = F.vflip(im_blue)
        label = F.vflip(label)
    norm1 = transforms.Normalize([means_s1[0], means_s1[1]],[stds_s1[0], stds_s1[1]])
    norm2 = transforms.Normalize([means_s2[12], means_s2[7], means_s2[3], means_s2[2]], [stds_s2[12], stds_s2[7], stds_s2[3], stds_s2[2]])
    # im_n1 = torch.stack([transforms.ToTensor()(im_vv).squeeze(), transforms.ToTensor()(im_vh).squeeze()])

    im1a2 = torch.stack([transforms.ToTensor()(im_vv).squeeze(), transforms.ToTensor()(im_vh).squeeze()])
    im1a2 = norm1(im1a2)
    im_n1 = torch.concatenate([im1a2, transforms.ToTensor()(im_rate)], dim=0)
    im_n2 = torch.stack([
        transforms.ToTensor()(im_swir).squeeze()/1.0,
        transforms.ToTensor()(im_nir).squeeze()/1.0,
        transforms.ToTensor()(im_green).squeeze()/1.0,
        transforms.ToTensor()(im_blue).squeeze()/1.0,
    ])
    im_n2 = norm2(im_n2)
    im_n2 = torch.stack([
        im_n2[0],
        im_n2[1],
        (im_n2[1] + im_n2[2] + im_n2[3])/3,
    ])
    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(.003) * label.lt(.004)):
        label *= 255
    label = label.round()

    return im_n2, im_n1, label

def processTestImS1S2_with_rate(data):
    (x1, x2, y) = data
    im1, im2, label = x1.copy(), x2.copy(), y.copy()
    norm1 = transforms.Normalize([means_s1[0], means_s1[1]], [stds_s1[0], stds_s1[1]])
    norm2 = transforms.Normalize([means_s2[12], means_s2[7], means_s2[3], means_s2[2]], [stds_s2[12], stds_s2[7], stds_s2[3], stds_s2[2]])

    # convert to PIL for easier transforms
    im_vv = Image.fromarray(im1[0]).resize((512, 512))
    im_vh = Image.fromarray(im1[1]).resize((512, 512))
    im_rate =  Image.fromarray(im1[2]).resize((512, 512))
    im_swir = Image.fromarray(im2[12]).resize((512, 512))
    im_nir = Image.fromarray(im2[7]).resize((512, 512))
    im_green = Image.fromarray(im2[3]).resize((512, 512))
    im_blue = Image.fromarray(im2[2]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512, 512))


    im_vvs = [F.crop(im_vv, 0, 0, 256, 256), F.crop(im_vv, 0, 256, 256, 256),
              F.crop(im_vv, 256, 0, 256, 256), F.crop(im_vv, 256, 256, 256, 256)]
    im_vhs = [F.crop(im_vh, 0, 0, 256, 256), F.crop(im_vh, 0, 256, 256, 256),
              F.crop(im_vh, 256, 0, 256, 256), F.crop(im_vh, 256, 256, 256, 256)]
    im_rates = [F.crop(im_rate, 0, 0, 256, 256), F.crop(im_rate, 0, 256, 256, 256),
              F.crop(im_rate, 256, 0, 256, 256), F.crop(im_rate, 256, 256, 256, 256)]
    im_swirs = [F.crop(im_swir, 0, 0, 256, 256), F.crop(im_swir, 0, 256, 256, 256),
              F.crop(im_swir, 256, 0, 256, 256), F.crop(im_swir, 256, 256, 256, 256)]
    im_nirs = [F.crop(im_nir, 0, 0, 256, 256), F.crop(im_nir, 0, 256, 256, 256),
              F.crop(im_nir, 256, 0, 256, 256), F.crop(im_nir, 256, 256, 256, 256)]
    im_greens = [F.crop(im_green, 0, 0, 256, 256), F.crop(im_green, 0, 256, 256, 256),
              F.crop(im_green, 256, 0, 256, 256), F.crop(im_green, 256, 256, 256, 256)]
    im_blues = [F.crop(im_blue, 0, 0, 256, 256), F.crop(im_blue, 0, 256, 256, 256),
              F.crop(im_blue, 256, 0, 256, 256), F.crop(im_blue, 256, 256, 256, 256)]
    labels = [F.crop(label, 0, 0, 256, 256), F.crop(label, 0, 256, 256, 256),
              F.crop(label, 256, 0, 256, 256), F.crop(label, 256, 256, 256, 256)]

    ims_n1 = [torch.stack((transforms.ToTensor()(x).squeeze(),
                        transforms.ToTensor()(y).squeeze(),
                        transforms.ToTensor()(z).squeeze()))
           for (x, y, z) in zip(im_vvs, im_vhs, im_rates)]
    ims_n2 = [torch.stack((transforms.ToTensor()(w).squeeze()/1.0,
                        transforms.ToTensor()(x).squeeze()/1.0,
                        transforms.ToTensor()(y).squeeze()/1.0,
                        transforms.ToTensor()(z).squeeze()/1.0
                       ))
           for (w, x, y, z) in zip(im_swirs, im_nirs, im_greens, im_blues)]

    def _processIm(im):
        im = norm2(im)
        return torch.stack([
            im[0],
            im[1],
            (im[1] + im[2] + im[3]) / 3,
        ])

    ims_n1 = [torch.concatenate((norm1(im[0:2]), im[2:3]), dim=0) for im in ims_n1]
    ims_n1 = torch.stack(ims_n1)

    ims_n2 = [_processIm(im) for im in ims_n2]
    ims_n2 = torch.stack(ims_n2)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)

    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
    labels = labels.round()

    return ims_n2, ims_n1, labels

def load_flood11_s1s2_with_rate_data(dataset_root, data_type='train'):

    assert data_type in ['train', 'val', 'test']
    fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_train_data.csv"
    if data_type == 'val':
        fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_valid_data.csv"
    elif data_type == 'test':
        fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_test_data.csv"

    input1_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/S1Hand/"
    input2_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/S2Hand/"
    label_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/LabelHand/"

    files = []
    with open(fname) as f:
        for line in csv.reader(f):
            files.append(tuple(
                (input1_root + line[0], input2_root + line[0].replace("S1Hand", "S2Hand"), label_root + line[1])
            ))
    return download_flood_water_data_from_list_with_rate(files)


if __name__ == "__main__":
    train_data = load_flood11_s1s2_with_rate_data('/home/amax/SSD1/zjzRoot/project/data/sen1floods11', 'train')
    train_dataset = FloodsS1S2WithRateDataset(train_data, processAndAugmentS1S2_with_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, sampler=None,
                                               batch_sampler=None, num_workers=0, collate_fn=None,
                                               pin_memory=True, drop_last=False, timeout=0,
                                               worker_init_fn=None)
    train_iter = iter(train_loader)
    train_iter.__next__()

    valid_data = load_flood11_s1s2_with_rate_data('/home/amax/SSD1/zjzRoot/project/data/sen1floods11', 'val')
    valid_dataset = FloodsS1S2WithRateDataset(valid_data, processTestImS1S2_with_rate)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True, sampler=None,
                                               batch_sampler=None, num_workers=0, collate_fn=lambda x: (
        torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0), torch.cat([a[2] for a in x], 0)),
                                               pin_memory=True, drop_last=False, timeout=0,
                                               worker_init_fn=None)
    valid_iter = iter(valid_loader)
    valid_iter.__next__()
