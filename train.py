from torch import nn

from models import get_net
from dataset import get_data_load

from utils import *
from tqdm import tqdm

# from configs.unet_s1_with_rate_config import *
# from configs.s2_unet_config import *
# from configs.swin_unet_s1_with_rate import *
# from configs.swin_unet_s2 import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

running_loss = 0
running_iou = 0
runing_background_iou = 0
running_count = 0
running_accuracy = 0

training_losses = []
training_accuracies = []
training_ious = []
training_bg_ious = []
valid_losses = []
valid_accuracies = []
valid_ious = []
valid_bg_ious = []

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 8]).float().cuda(), ignore_index=255)

def train_loop(inputs1,  inputs2, labels, net, optimizer, scheduler):
    global running_loss
    global running_iou
    global runing_background_iou
    global running_count
    global running_accuracy

    # zero the parameter gradients
    net.train()
    optimizer.zero_grad()


    # forward + backward + optimize
    if DATASET == "floods_s1" or DATASET == "floods_s2" or DATASET == "floods_s1_with_rate" or DATASET == "test":
        outputs = net(inputs1)
    else:
        outputs = net(inputs1, inputs2)

    loss = criterion(outputs, labels.long())
    loss.backward()
    optimizer.step()

    if not scheduler == None:
        scheduler.step()

    running_loss += loss.cpu().detach().numpy()
    running_iou += computeIOU(outputs, labels).cpu().detach().numpy()
    runing_background_iou += compute_background_IOU(outputs, labels).cpu().detach().numpy()
    running_accuracy += computeAccuracy(outputs, labels).cpu().detach().numpy()
    running_count += 1


def validation_loop(validation_data_loader, net, i):
    global running_loss
    global running_iou
    global runing_background_iou
    global running_count
    global running_accuracy
    global max_valid_iou

    global training_losses
    global training_accuracies
    global training_ious
    global training_bg_ious
    global valid_losses
    global valid_accuracies
    global valid_ious
    global valid_bg_ious

    net.eval()
    count = 0
    iou = 0
    bg_iou = 0
    loss = 0
    accuracy = 0
    with torch.no_grad():
        if DATASET == "floods_s1" or DATASET == "floods_s2" or DATASET == "floods_s1_with_rate"  or DATASET == "test":
            for (images, labels) in validation_data_loader:
                outputs = net(images.cuda())
                valid_loss = criterion(outputs, labels.long().cuda())
                valid_iou = computeIOU(outputs, labels.cuda())
                valid_bg_iou = compute_background_IOU(outputs, labels.cuda())
                valid_accuracy = computeAccuracy(outputs, labels.cuda())
                iou += valid_iou
                bg_iou += valid_bg_iou
                loss += valid_loss
                accuracy += valid_accuracy
                count += 1
        else:
            for (images1, images2, labels) in validation_data_loader:
                outputs = net(images1.cuda(), images2.cuda())
                valid_loss = criterion(outputs, labels.long().cuda())
                valid_iou = computeIOU(outputs, labels.cuda())
                valid_bg_iou = compute_background_IOU(outputs, labels.cuda())
                valid_accuracy = computeAccuracy(outputs, labels.cuda())
                iou += valid_iou
                bg_iou += valid_bg_iou
                loss += valid_loss
                accuracy += valid_accuracy
                count += 1

    iou = iou / count
    bg_iou = bg_iou / count
    accuracy = accuracy / count

    if iou > max_valid_iou:
        max_valid_iou = iou
        save_path = os.path.join("{}_{}_{}.cp".format(NET_NAME, i, iou.item()))
        torch.save(net.state_dict(), save_path)
        print("model saved at", save_path)

    loss = loss / count
    print("Training Loss:", running_loss / running_count)
    print("Training IOU:", running_iou / running_count)
    print("Training bg IoU:", runing_background_iou / running_count)
    print("Training Accuracy:", running_accuracy / running_count)
    print("Validation Loss:", loss)
    print("Validation IOU:", iou)
    print("Validation bg IoU:", bg_iou)
    print("Validation Accuracy:", accuracy)

    training_losses.append(running_loss / running_count)
    training_accuracies.append(running_accuracy / running_count)
    training_ious.append(running_iou / running_count)
    training_bg_ious.append(runing_background_iou / running_count)
    valid_losses.append(loss.cpu().detach().numpy())
    valid_accuracies.append(accuracy.cpu().detach().numpy())
    valid_ious.append(iou.cpu().detach().numpy())
    valid_bg_ious.append(bg_iou.cpu().detach().numpy())


def train_epoch(net, optimizer, scheduler, train_iter):
    if DATASET == "floods_s1" or DATASET == "floods_s2" or DATASET == "floods_s1_with_rate" or "test":
        for (inputs, labels) in tqdm(train_iter):
            train_loop(inputs.cuda(), None, labels.cuda(), net, optimizer, scheduler)
    else:
        for (inputs1, inputs2, labels) in tqdm(train_iter):
            train_loop(inputs1.cuda(), inputs2.cuda(), labels.cuda(), net, optimizer, scheduler)

def train_validation_loop(net, optimizer, scheduler, train_loader,
                          valid_loader, num_epochs, cur_epoch):
    global running_loss
    global running_iou
    global runing_background_iou
    global running_count
    global running_accuracy
    running_loss = 0
    running_iou = 0
    runing_background_iou = 0
    running_count = 0
    running_accuracy = 0

    for i in tqdm(range(num_epochs)):
        train_iter = iter(train_loader)
        train_epoch(net, optimizer, scheduler, train_iter)

    print("Current Epoch:", cur_epoch)
    validation_loop(iter(valid_loader), net, cur_epoch)

if __name__ == '__main__':

    max_valid_iou = 0
    start = 0

    train_loader, valid_loader = get_data_load(DATASET)
    net = get_net(NET_NAME)
    net = net.cuda()

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 8]).float().cuda(), ignore_index=255)
    # optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.0, nesterov=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader) * 10, T_mult=2, eta_min=0, last_epoch=-1)
    # CosineAnnealingWarmRestarts( optimizer, T_0, T_mult=2, eta_min=0, last_epoch=-1 )
    # T_0：初始的热重启周期，即第一次完整的余弦退火周期的步数（这里是 len(train_loader) * 10）。
    # T_mult：周期的倍增系数。每次重启后，周期会按照 T_mult 倍数递增。例如，如果 T_0=10 且 T_mult=2，则周期变化为：10 → 20 → 40 → 80 …
    # eta_min：学习率的最小值，余弦退火曲线会逐渐减小到这个最小值。
    # last_epoch：初始epoch，默认为 -1，表示从头开始训练。
    # scheduler = None

    for i in range(start, MAX_EPOCHS):
        train_validation_loop(net, optimizer, scheduler, train_loader, valid_loader, 10, i)
        # 创建一个字符串变量来存储所有输出
        output = ""
        output += "training_losses: " + str([float(los) for los in training_losses]) + "\n"
        output += "training_accuracies: " + str([float(acc) for acc in training_accuracies]) + "\n"
        output += "training_ious: " + str([float(iou) for iou in training_ious]) + "\n"
        output += "training_bg_ious: " + str([float(iou) for iou in training_bg_ious]) + "\n"
        output += "valid_losses: " + str([float(los) for los in valid_losses]) + "\n"
        output += "valid_accuracies: " + str([float(acc) for acc in valid_accuracies]) + "\n"
        output += "valid_ious: " + str([float(iou) for iou in valid_ious]) + "\n"
        output += "valid_bg_ious: " + str([float(iou) for iou in valid_bg_ious]) + "\n"
        output += "max valid iou: " + str(max_valid_iou) + "\n"
        print("max valid iou: " + str(max_valid_iou) + "\n")
        # 打开一个文件以写入模式
        with open(NET_NAME + '_trlogaining.log', 'w') as f:
            f.write(output)