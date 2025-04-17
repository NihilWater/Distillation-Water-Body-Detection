import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image
import csv
import numpy as np
import rasterio
import os

def getArrFloodS1(fname):
    return rasterio.open(fname).read()


def download_flood_water_data_from_list_s1_with_rate(l):
    i = 0
    flood_data = []
    for (im_fname, mask_fname) in l:
        if not os.path.exists(im_fname):
            continue
        arr_x = np.nan_to_num(getArrFloodS1(im_fname))
        arr_y = getArrFloodS1(mask_fname)
        arr_y[arr_y == -1] = 255

        arr_x = np.concatenate((arr_x, arr_x[0:1, ...] / arr_x[1:2, ...]), axis=0)
        arr_x[2] = arr_x[2].clip(-2, 2)
        arr_x = np.nan_to_num(arr_x)

        arr_x[0:2] = np.clip(arr_x[0:2], -50, 1)
        arr_x[0:2] = (arr_x[0:2] + 50) / 51

        if i % 100 == 0:
            print(im_fname, mask_fname)
        i += 1
        flood_data.append((arr_x, arr_y))

    return flood_data

class FloodsS1WithRateDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)

def processAndAugmentS1_with_rate(data):
    (x, y) = data
    im, label = x.copy(), y.copy()

    # convert to PIL for easier transforms
    im1 = Image.fromarray(im[0])
    im2 = Image.fromarray(im[1])
    im3 = Image.fromarray(im[2])
    label = Image.fromarray(label.squeeze())

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(im1, (256, 256))

    im1 = F.crop(im1, i, j, h, w)
    im2 = F.crop(im2, i, j, h, w)
    im3 = F.crop(im3, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im1 = F.hflip(im1)
        im2 = F.hflip(im2)
        im3 = F.hflip(im3)
        label = F.hflip(label)
    if random.random() > 0.5:
        im1 = F.vflip(im1)
        im2 = F.vflip(im2)
        im3 = F.vflip(im3)
        label = F.vflip(label)

    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
    im1a2 = torch.stack([transforms.ToTensor()(im1).squeeze(), transforms.ToTensor()(im2).squeeze()])
    im1a2 = norm(im1a2)
    im = torch.concatenate([im1a2, transforms.ToTensor()(im3)], dim=0)
    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(.003) * label.lt(.004)):
        label *= 255
    label = label.round()

    return im, label

def processTestImS1_with_rate(data):
    (x, y) = data
    im, label = x.copy(), y.copy()
    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])

    # convert to PIL for easier transforms
    im_c1 = Image.fromarray(im[0]).resize((512, 512))
    im_c2 = Image.fromarray(im[1]).resize((512, 512))
    im_c3 = Image.fromarray(im[2]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512, 512))

    im_c1s = [F.crop(im_c1, 0, 0, 256, 256), F.crop(im_c1, 0, 256, 256, 256),
              F.crop(im_c1, 256, 0, 256, 256), F.crop(im_c1, 256, 256, 256, 256)]
    im_c2s = [F.crop(im_c2, 0, 0, 256, 256), F.crop(im_c2, 0, 256, 256, 256),
              F.crop(im_c2, 256, 0, 256, 256), F.crop(im_c2, 256, 256, 256, 256)]
    im_c3s = [F.crop(im_c3, 0, 0, 256, 256), F.crop(im_c3, 0, 256, 256, 256),
              F.crop(im_c3, 256, 0, 256, 256), F.crop(im_c3, 256, 256, 256, 256)]

    labels = [F.crop(label, 0, 0, 256, 256), F.crop(label, 0, 256, 256, 256),
              F.crop(label, 256, 0, 256, 256), F.crop(label, 256, 256, 256, 256)]

    ims = [torch.stack((transforms.ToTensor()(x).squeeze(),
                        transforms.ToTensor()(y).squeeze(),
                        transforms.ToTensor()(z).squeeze()))
           for (x, y, z) in zip(im_c1s, im_c2s, im_c3s)]

    ims = [torch.concatenate((norm(im[0:2]), im[2:3]), dim=0) for im in ims]

    ims = torch.stack(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)

    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
    labels = labels.round()

    return ims, labels

def load_flood_data_s1_with_rate(dataset_root, data_type='train'):
    assert data_type in ['train', 'val', 'test']
    fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_train_data.csv"
    if data_type == 'val':
        fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_valid_data.csv"
    elif data_type == 'test':
        fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_test_data.csv"

    input_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/S1Hand/"
    label_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/LabelHand/"
    files = []
    with open(fname) as f:
        for line in csv.reader(f):
            files.append(tuple((input_root+line[0],  label_root+line[1])))

    return download_flood_water_data_from_list_s1_with_rate(files)



if __name__ == "__main__":
    train_data = load_flood_data_s1_with_rate('/home/amax/SSD1/zjzRoot/project/data/sen1floods11', data_type='train')
    train_dataset = FloodsS1WithRateDataset(train_data, processAndAugmentS1_with_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, sampler=None,
                                               batch_sampler=None, num_workers=0, collate_fn=None,
                                               pin_memory=True, drop_last=False, timeout=0,
                                               worker_init_fn=None)
    train_iter = iter(train_loader)
    train_iter.__next__()

    valid_data = load_flood_data_s1_with_rate('/home/amax/SSD1/zjzRoot/project/data/sen1floods11', data_type='val')
    valid_dataset = FloodsS1WithRateDataset(valid_data, processTestImS1_with_rate)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True, sampler=None,
                                               batch_sampler=None, num_workers=0, collate_fn=lambda x: (
        torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                                               pin_memory=True, drop_last=False, timeout=0,
                                               worker_init_fn=None)
    valid_iter = iter(valid_loader)
    valid_iter.__next__()