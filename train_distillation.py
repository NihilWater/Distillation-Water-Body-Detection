from torch import nn

from models import get_net
from dataset import get_data_load

from utils import *
from tqdm import tqdm

import os
# from configs.trans_unet_st_config import *
# from configs.deeplabv3_st_config import *
# from configs.unet_st_config import *
# from configs.psp_st_config import *
# from configs.swin_unet_st_config import *
from configs.attentive_unet_st_config import *


os.environ["CUDA_VISIBLE_DEVICES"] = "7"

LR = 0.01
MAX_EPOCHS = 200
DATASET = "floods_s1s2_with_rate"

# 蒸馏超参数
alpha = 0.5  # 蒸馏损失和监督损失的权重
temperature = 3.0  # 温度值


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

def train_loop(inputs1, inputs2, labels, net, teacher_net, optimizer, scheduler):
    global running_loss
    global running_iou
    global runing_background_iou
    global running_count
    global running_accuracy

    # zero the parameter gradients
    net.train()
    optimizer.zero_grad()
    net = net.cuda()
    teacher_net = teacher_net.cuda()

    # 使用教师模型生成软标签
    with torch.no_grad():
        teacher_outputs = teacher_net(inputs1.cuda())

    # forward + backward + optimize
    outputs = net(inputs2)

    # 蒸馏损失（基于温度调整的KL散度）
    teacher_probs = nn.functional.softmax(teacher_outputs / temperature, dim=1)
    student_probs = nn.functional.log_softmax(outputs / temperature, dim=1)
    # 像素级 KL 散度计算
    pixelwise_kl = nn.functional.kl_div(student_probs, teacher_probs, reduction='none')  # 不直接归约
    distillation_loss = pixelwise_kl.mean(dim=(1, 2, 3)).mean() * (temperature ** 2)  # 像素、通道归约后再平均

    # supervision_loss = criterion(outputs, labels.long().cuda())
    #
    # loss = alpha * distillation_loss + (1 - alpha) * supervision_loss
    loss = distillation_loss
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
        for (_, images2, labels) in validation_data_loader:
            net = net.cuda()
            outputs = net(images2.cuda())
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
        save_path = os.path.join("{}_{}_{}.cp".format("TS_"+Student_NET_NAME, i, iou.item()))
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


def train_epoch(net, teacher_net, optimizer, scheduler, train_iter):
    for (inputs1, inputs2, labels) in tqdm(train_iter):
        train_loop(inputs1.cuda(), inputs2.cuda(), labels.cuda(), net, teacher_net, optimizer, scheduler)

def train_validation_loop(net, teacher_net, optimizer, scheduler, train_loader,
                          valid_loader, num_epochs, cur_epoch):
    global running_loss
    global running_iou
    global runing_background_iou
    global running_count
    global running_accuracy
    net = net.train()
    running_loss = 0
    running_iou = 0
    runing_background_iou = 0
    running_count = 0
    running_accuracy = 0

    for i in tqdm(range(num_epochs)):
        train_iter = iter(train_loader)
        train_epoch(net, teacher_net, optimizer, scheduler, train_iter)

    print("Current Epoch:", cur_epoch)
    validation_loop(iter(valid_loader), net, cur_epoch)

if __name__ == '__main__':

    max_valid_iou = 0
    start = 0

    train_loader, valid_loader = get_data_load(DATASET)
    teacher_net = get_net(Teacher_NET_NAME)
    teacher_net.load_state_dict(torch.load(Teacher_PRETRAIN_MODEL))
    net = get_net(Student_NET_NAME)
    teacher_net.cuda()
    net.cuda()

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 8]).float().cuda(), ignore_index=255)
    # optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.0, nesterov=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader) * 10, T_mult=2, eta_min=0, last_epoch=-1)
    # 定义自定义 Lambda 函数
    def lr_lambda(epoch):
        if epoch in [30, 60, 100, 150]:
            return 0.5  # 学习率减半
        return 1.0  # 保持原学习率
    # 使用 LambdaLR
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # CosineAnnealingWarmRestarts( optimizer, T_0, T_mult=2, eta_min=0, last_epoch=-1 )
    # T_0：初始的热重启周期，即第一次完整的余弦退火周期的步数（这里是 len(train_loader) * 10）。
    # T_mult：周期的倍增系数。每次重启后，周期会按照 T_mult 倍数递增。例如，如果 T_0=10 且 T_mult=2，则周期变化为：10 → 20 → 40 → 80 …
    # eta_min：学习率的最小值，余弦退火曲线会逐渐减小到这个最小值。
    # last_epoch：初始epoch，默认为 -1，表示从头开始训练。
    # scheduler = None

    for i in range(start, MAX_EPOCHS):
        train_validation_loop(net, teacher_net, optimizer, scheduler, train_loader, valid_loader, 10, i)
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
        with open("ST_" + Student_NET_NAME + '_training.log', 'w') as f:
            f.write(output)