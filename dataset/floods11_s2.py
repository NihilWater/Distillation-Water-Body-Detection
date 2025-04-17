import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image
import csv
import numpy as np
import rasterio
import os

means_s2 = [1647.5275, 1409.1735, 1373.1986, 1220.8898, 1470.2974, 2395.1107, 2857.3290, 2633.3441, 3089.7533, 487.1713, 60.5535, 2033.6098, 1178.3920]
stds_s2  = [659.5315, 714.0080, 712.1827, 851.5684, 749.7281, 863.7205, 1017.6825, 968.1219, 1131.4698, 327.6270, 131.5372, 958.7381, 757.5991]
'''
以上的均值和方差是使用 preprocess/calc_mean_std_on_s2 获取的
'''

def getArrFloodS2(fname):
    return rasterio.open(fname).read()

def download_flood_water_data_from_list_s2(l):
    i = 0
    tot_nan = 0
    tot_good = 0
    flood_data = []
    for (im_fname, mask_fname) in l:
        if not os.path.exists(im_fname):
            continue
        arr_x = np.nan_to_num(getArrFloodS2(im_fname))
        arr_y = getArrFloodS2(mask_fname)
        arr_y[arr_y == -1] = 255

        if i % 100 == 0:
            print(im_fname, mask_fname)
        i += 1
        flood_data.append((arr_x, arr_y))

    return flood_data

class FloodsS2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)

def processAndAugmentS2(data):
    (x, y) = data
    im, label = x.copy(), y.copy()

    # convert to PIL for easier transforms
    im_swir = Image.fromarray(im[12])
    im_nir = Image.fromarray(im[7])
    im_green =Image.fromarray(im[3])
    im_blue  = Image.fromarray(im[2])

    label = Image.fromarray(label.squeeze())

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(im_swir, (256, 256))

    im_swir =  F.crop(im_swir, i, j, h, w)
    im_nir =  F.crop(im_nir, i, j, h, w)
    im_green =  F.crop(im_green, i, j, h, w)
    im_blue  = F.crop(im_blue, i, j, h, w)

    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im_swir = F.hflip(im_swir)
        im_nir = F.hflip(im_nir)
        im_green = F.hflip(im_green)
        im_blue = F.hflip(im_blue)
        label = F.hflip(label)
    if random.random() > 0.5:
        im_swir = F.vflip(im_swir)
        im_nir = F.vflip(im_nir)
        im_green = F.vflip(im_green)
        im_blue = F.vflip(im_blue)
        label = F.vflip(label)

    norm = transforms.Normalize([means_s2[12], means_s2[7], means_s2[3], means_s2[2]], [stds_s2[12], stds_s2[7], stds_s2[3], stds_s2[2]])
    im = torch.stack([
        transforms.ToTensor()(im_swir).squeeze()/1.0,
        transforms.ToTensor()(im_nir).squeeze()/1.0,
        transforms.ToTensor()(im_green).squeeze()/1.0,
        transforms.ToTensor()(im_blue).squeeze()/1.0,
    ])
    im = norm(im)
    im = torch.stack([
        im[0],
        im[1],
        (im[1] + im[2] + im[3])/3,
    ])
    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(.003) * label.lt(.004)):
        label *= 255
    label = label.round()

    return im, label

def processTestImS2(data):
    (x, y) = data
    im, label = x.copy(), y.copy()
    norm = transforms.Normalize([means_s2[12], means_s2[7], means_s2[3], means_s2[2]], [stds_s2[12], stds_s2[7], stds_s2[3], stds_s2[2]])

    # convert to PIL for easier transforms
    im_swir = Image.fromarray(im[12]).resize((512, 512))
    im_nir = Image.fromarray(im[7]).resize((512, 512))
    im_green = Image.fromarray(im[3]).resize((512, 512))
    im_blue = Image.fromarray(im[2]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512, 512))


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

    ims = [torch.stack((transforms.ToTensor()(w).squeeze()/1.0,
                        transforms.ToTensor()(x).squeeze()/1.0,
                        transforms.ToTensor()(y).squeeze()/1.0,
                        transforms.ToTensor()(z).squeeze()/1.0
                       ))
           for (w, x, y, z) in zip(im_swirs, im_nirs, im_greens, im_blues)]

    def _processIm(im):
        im = norm(im)
        return torch.stack([
            im[0],
            im[1],
            (im[1] + im[2] + im[3]) / 3,
        ])
    ims = [_processIm(im) for im in ims]
    ims = torch.stack(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)

    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
    labels = labels.round()

    return ims, labels

def load_flood_train_data_s2(dataset_root):

  fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_train_data.csv"
  input_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/S2Hand/"
  label_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/LabelHand/"
  training_files = []
  with open(fname) as f:
    for line in csv.reader(f):
      training_files.append(tuple((input_root+line[0].replace("S1Hand", "S2Hand"),  label_root+line[1])))

  return download_flood_water_data_from_list_s2(training_files)

def load_flood_valid_data_s2(dataset_root):
    fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_valid_data.csv"
    input_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/S2Hand/"
    label_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/LabelHand/"
    validation_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            validation_files.append(tuple((input_root + line[0].replace("S1Hand", "S2Hand"), label_root + line[1])))

    return download_flood_water_data_from_list_s2(validation_files)

def load_flood_test_data_s2(dataset_root):
    fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_test_data.csv"
    input_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/S2Hand/"
    label_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/LabelHand/"
    testing_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            testing_files.append(tuple((input_root + line[0], label_root + line[1])))

    return download_flood_water_data_from_list_s2(testing_files)


if __name__ == "__main__":
    train_data = load_flood_train_data_s2('/home/amax/SSD1/zjzRoot/project/data/sen1floods11')
    train_dataset = FloodsS2Dataset(train_data, processAndAugmentS2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, sampler=None,
                                               batch_sampler=None, num_workers=0, collate_fn=None,
                                               pin_memory=True, drop_last=False, timeout=0,
                                               worker_init_fn=None)
    train_iter = iter(train_loader)
    train_iter.__next__()

    valid_data = load_flood_valid_data_s2('/home/amax/SSD1/zjzRoot/project/data/sen1floods11')
    valid_dataset = FloodsS2Dataset(valid_data, processTestImS2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True, sampler=None,
                                               batch_sampler=None, num_workers=0, collate_fn=lambda x: (
        torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                                               pin_memory=True, drop_last=False, timeout=0,
                                               worker_init_fn=None)
    valid_iter = iter(valid_loader)
    valid_iter.__next__()
