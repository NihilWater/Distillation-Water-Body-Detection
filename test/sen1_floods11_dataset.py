import torch
import os
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image
import csv
import numpy as np
import rasterio

import torchvision.models as models
import torch.nn as nn

def getArrFlood(fname):
    return rasterio.open(fname).read()


def download_flood_water_data_from_list(l):
    i = 0
    tot_nan = 0
    tot_good = 0
    flood_data = []
    for (im_fname, mask_fname) in l:
        if not os.path.exists(im_fname):
            continue
        arr_x = np.nan_to_num(getArrFlood(im_fname))
        arr_y = getArrFlood(mask_fname)
        arr_y[arr_y == -1] = 255

        arr_x = np.clip(arr_x, -50, 1)
        arr_x = (arr_x + 50) / 51

        if i % 100 == 0:
            print(im_fname, mask_fname)
        i += 1
        flood_data.append((arr_x, arr_y))

    return flood_data


class InMemoryDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)


def processAndAugment(data):
    (x, y) = data
    im, label = x.copy(), y.copy()

    # convert to PIL for easier transforms
    im1 = Image.fromarray(im[0])
    im2 = Image.fromarray(im[1])
    label = Image.fromarray(label.squeeze())

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(im1, (256, 256))

    im1 = F.crop(im1, i, j, h, w)
    im2 = F.crop(im2, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im1 = F.hflip(im1)
        im2 = F.hflip(im2)
        label = F.hflip(label)
    if random.random() > 0.5:
        im1 = F.vflip(im1)
        im2 = F.vflip(im2)
        label = F.vflip(label)

    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
    im = torch.stack([transforms.ToTensor()(im1).squeeze(), transforms.ToTensor()(im2).squeeze()])
    im = norm(im)
    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(.003) * label.lt(.004)):
        label *= 255
    label = label.round()

    return im, label


def processTestIm(data):
    (x, y) = data
    im, label = x.copy(), y.copy()
    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])

    # convert to PIL for easier transforms
    im_c1 = Image.fromarray(im[0]).resize((512, 512))
    im_c2 = Image.fromarray(im[1]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512, 512))

    im_c1s = [F.crop(im_c1, 0, 0, 256, 256), F.crop(im_c1, 0, 256, 256, 256),
              F.crop(im_c1, 256, 0, 256, 256), F.crop(im_c1, 256, 256, 256, 256)]
    im_c2s = [F.crop(im_c2, 0, 0, 256, 256), F.crop(im_c2, 0, 256, 256, 256),
              F.crop(im_c2, 256, 0, 256, 256), F.crop(im_c2, 256, 256, 256, 256)]
    labels = [F.crop(label, 0, 0, 256, 256), F.crop(label, 0, 256, 256, 256),
              F.crop(label, 256, 0, 256, 256), F.crop(label, 256, 256, 256, 256)]

    ims = [torch.stack((transforms.ToTensor()(x).squeeze(),
                        transforms.ToTensor()(y).squeeze()))
           for (x, y) in zip(im_c1s, im_c2s)]

    ims = [norm(im) for im in ims]
    ims = torch.stack(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)

    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
    labels = labels.round()

    return ims, labels


def load_flood_train_data(dataset_root):

  fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_train_data.csv"
  input_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/S1Hand/"
  label_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/LabelHand/"
  training_files = []
  with open(fname) as f:
    for line in csv.reader(f):
      training_files.append(tuple((input_root+line[0], label_root+line[1])))

  return download_flood_water_data_from_list(training_files)

def load_flood_valid_data(dataset_root):
    fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_valid_data.csv"
    input_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/S1Hand/"
    label_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/LabelHand/"
    validation_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            validation_files.append(tuple((input_root + line[0], label_root + line[1])))

    return download_flood_water_data_from_list(validation_files)


def load_flood_test_data(dataset_root):
    fname = dataset_root + "/v1.1/splits/flood_handlabeled/flood_test_data.csv"
    input_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/S1Hand/"
    label_root = dataset_root + "/v1.1/data/flood_events/HandLabeled/LabelHand/"
    testing_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            testing_files.append(tuple((input_root + line[0], label_root + line[1])))

    return download_flood_water_data_from_list(testing_files)


if __name__ == '__main__':
    # train_data = load_flood_train_data('S1/', 'Labels/')
    train_data = load_flood_train_data('/home/amax/SSD1/zjzRoot/project/data/sen1floods11')
    train_dataset = InMemoryDataset(train_data, processAndAugment)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, sampler=None,
                                               batch_sampler=None, num_workers=0, collate_fn=None,
                                               pin_memory=True, drop_last=False, timeout=0,
                                               worker_init_fn=None)
    train_iter = iter(train_loader)

    valid_data = load_flood_valid_data('/home/amax/SSD1/zjzRoot/project/data/sen1floods11')
    valid_dataset = InMemoryDataset(valid_data, processTestIm)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True, sampler=None,
                                               batch_sampler=None, num_workers=0, collate_fn=lambda x: (
        torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                                               pin_memory=True, drop_last=False, timeout=0,
                                               worker_init_fn=None)
    valid_iter = iter(valid_loader)


    # --------------------------- 网络定义 -------------------------------------------------
    LR = 5e-4
    net = models.segmentation.fcn_resnet50(pretrained=False, num_classes=2, pretrained_backbone=False)
    net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 8]).float().cuda(), ignore_index=255)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader) * 10, T_mult=2,
                                                                     eta_min=0, last_epoch=-1)

    def convertBNtoGN(module, num_groups=16):
        if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            return nn.GroupNorm(num_groups, module.num_features,
                                eps=module.eps, affine=module.affine)
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

        for name, child in module.named_children():
            module.add_module(name, convertBNtoGN(child, num_groups=num_groups))

        return module

    net = convertBNtoGN(net)

    # -------------------------------------------------------------------------------------------------
    def computeIOU(output, target):
        output = torch.argmax(output, dim=1).flatten()
        target = target.flatten()

        no_ignore = target.ne(255).cuda()
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        intersection = torch.sum(output * target)
        union = torch.sum(target) + torch.sum(output) - intersection
        iou = (intersection + .0000001) / (union + .0000001)

        if iou != iou:
            print("failed, replacing with 0")
            iou = torch.tensor(0).float()

        return iou


    def computeAccuracy(output, target):
        output = torch.argmax(output, dim=1).flatten()
        target = target.flatten()

        no_ignore = target.ne(255).cuda()
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        correct = torch.sum(output.eq(target))

        return correct.float() / len(target)


    def truePositives(output, target):
        output = torch.argmax(output, dim=1).flatten()
        target = target.flatten()
        no_ignore = target.ne(255).cuda()
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        correct = torch.sum(output * target)

        return correct


    def trueNegatives(output, target):
        output = torch.argmax(output, dim=1).flatten()
        target = target.flatten()
        no_ignore = target.ne(255).cuda()
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        output = (output == 0)
        target = (target == 0)
        correct = torch.sum(output * target)

        return correct


    def falsePositives(output, target):
        output = torch.argmax(output, dim=1).flatten()
        target = target.flatten()
        no_ignore = target.ne(255).cuda()
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        output = (output == 1)
        target = (target == 0)
        correct = torch.sum(output * target)

        return correct


    def falseNegatives(output, target):
        output = torch.argmax(output, dim=1).flatten()
        target = target.flatten()
        no_ignore = target.ne(255).cuda()
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        output = (output == 0)
        target = (target == 1)
        correct = torch.sum(output * target)

        return correct

    # -----------------------------Define training loop------------------------------------------------------------
    training_losses = []
    training_accuracies = []
    training_ious = []


    def train_loop(inputs, labels, net, optimizer, scheduler):
        global running_loss
        global running_iou
        global running_count
        global running_accuracy

        # zero the parameter gradients
        optimizer.zero_grad()
        net = net.cuda()

        # forward + backward + optimize
        outputs = net(inputs.cuda())
        loss = criterion(outputs["out"], labels.long().cuda())
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss
        running_iou += computeIOU(outputs["out"], labels.cuda())
        running_accuracy += computeAccuracy(outputs["out"], labels.cuda())
        running_count += 1

    # -----------------------------  Define validation loop------------------------------------------------------------
    valid_losses = []
    valid_accuracies = []
    valid_ious = []

    def validation_loop(validation_data_loader, net, i):
        global running_loss
        global running_iou
        global running_count
        global running_accuracy
        global max_valid_iou

        global training_losses
        global training_accuracies
        global training_ious
        global valid_losses
        global valid_accuracies
        global valid_ious

        net = net.eval()
        net = net.cuda()
        count = 0
        iou = 0
        loss = 0
        accuracy = 0
        with torch.no_grad():
            for (images, labels) in validation_data_loader:
                net = net.cuda()
                outputs = net(images.cuda())
                valid_loss = criterion(outputs["out"], labels.long().cuda())
                valid_iou = computeIOU(outputs["out"], labels.cuda())
                valid_accuracy = computeAccuracy(outputs["out"], labels.cuda())
                iou += valid_iou
                loss += valid_loss
                accuracy += valid_accuracy
                count += 1

        iou = iou / count
        accuracy = accuracy / count

        if iou > max_valid_iou:
            max_valid_iou = iou
            save_path = os.path.join("{}_{}_{}.cp".format("Sen1Floods11", i, iou.item()))
            torch.save(net.state_dict(), save_path)
            print("model saved at", save_path)

        loss = loss / count
        print("Training Loss:", running_loss / running_count)
        print("Training IOU:", running_iou / running_count)
        print("Training Accuracy:", running_accuracy / running_count)
        print("Validation Loss:", loss)
        print("Validation IOU:", iou)
        print("Validation Accuracy:", accuracy)

        training_losses.append(running_loss / running_count)
        training_accuracies.append(running_accuracy / running_count)
        training_ious.append(running_iou / running_count)
        valid_losses.append(loss)
        valid_accuracies.append(accuracy)
        valid_ious.append(iou)

    # ---------------------------  Define testing loop   ------------------------------------------------------
    def test_loop(test_data_loader, net):
        net = net.eval()
        net = net.cuda()
        count = 0
        iou = 0
        loss = 0
        accuracy = 0
        with torch.no_grad():
            for (images, labels) in tqdm(test_data_loader):
                net = net.cuda()
                outputs = net(images.cuda())
                valid_loss = criterion(outputs["out"], labels.long().cuda())
                valid_iou = computeIOU(outputs["out"], labels.cuda())
                iou += valid_iou
                accuracy += computeAccuracy(outputs["out"], labels.cuda())
                count += 1

        iou = iou / count
        print("Test IOU:", iou)
        print("Test Accuracy:", accuracy / count)


    # --------------------------------   Train model and assess metrics over epochs  ---------------------------------

    from tqdm.notebook import tqdm
    from IPython.display import clear_output

    running_loss = 0
    running_iou = 0
    running_count = 0
    running_accuracy = 0


    def train_epoch(net, optimizer, scheduler, train_iter):
        for (inputs, labels) in tqdm(train_iter):
            train_loop(inputs.cuda(), labels.cuda(), net.cuda(), optimizer, scheduler)


    def train_validation_loop(net, optimizer, scheduler, train_loader,
                              valid_loader, num_epochs, cur_epoch):
        global running_lossa
        global running_iou
        global running_count
        global running_accuracy
        net = net.train()
        running_loss = 0
        running_iou = 0
        running_count = 0
        running_accuracy = 0

        for i in tqdm(range(num_epochs)):
            train_iter = iter(train_loader)
            train_epoch(net, optimizer, scheduler, train_iter)
        clear_output()

        print("Current Epoch:", cur_epoch)
        validation_loop(iter(valid_loader), net, i)


    # ------------------------- Train model and assess metrics over epochs ------------------------------------------
    import os
    import matplotlib.pyplot as plt

    max_valid_iou = 0
    start = 0

    epochs = []

    for i in range(start, 1000):
        train_validation_loop(net, optimizer, scheduler, train_loader, valid_loader, 10, i)
        epochs.append(i)
        x = epochs
        print([loss.cpu().detach().numpy() for loss in training_losses])
        print([acc.cpu().detach().numpy() for acc in training_accuracies])
        print([iou.cpu().detach().numpy() for iou in training_ious])
        print([loss.cpu().detach().numpy() for loss in valid_losses])
        print([acc.cpu().detach().numpy() for acc in valid_accuracies])
        print([iou.cpu().detach().numpy() for iou in valid_ious])
        # plt.legend(loc="upper left")
        #
        # filename = os.path.join("pic", f"epoch_{i}.png")
        # plt.savefig(filename)
        # plt.close()  # 关闭当前图像，否则后续的图像会在同一张图上绘制

        print("max valid iou:", max_valid_iou)