import torch

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


def compute_background_IOU(output, target):
    # 将 output 处理为类别标签
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()

    # 忽略标签为 255 的无效区域
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)

    # 提取背景类别的预测和目标
    background_output = (output == 0).float()
    background_target = (target == 0).float()

    # 计算交集和并集
    intersection = torch.sum(background_output * background_target)
    union = torch.sum(background_target) + torch.sum(background_output) - intersection

    # 计算 IOU
    iou = (intersection + 1e-7) / (union + 1e-7)

    if iou != iou:  # 检查 NaN
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