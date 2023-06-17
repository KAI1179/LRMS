'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
from fine_grained_datasets import load_dataset
import os
import argparse
import numpy as np
from utils.AverageMeter import AverageMeter
from cs_kd_models import resnet18
from utils.metric import metric_ece_aurc_eaurc
from utils.etc import progress_bar, is_main_process, save_on_master, paser_config_save, set_logging_defaults

def seed_it(seed):
    random.seed(seed) 
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.enabled = True  
    torch.manual_seed(seed)
# seed_it(114514)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument('--alpha', default
=0.9, type=float, help='KD loss alpha')
parser.add_argument('--temperature_good', default=8, type=int, help='KD loss temperature')
parser.add_argument('--temperature_other', default=1, type=int, help='KD loss temperature')
parser.add_argument('--warmup', default=20, type=int, help='warm up epoch')
# parser.add_argument('--model_name', action='_5_model',
#                     help='model name for save path')
parser.add_argument("--root", type=str, default="./data")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--classes_num", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--cls", type=bool, default=True)
parser.add_argument('--dataset', default='CUB200', type=str, help='the name for dataset cifar100 | tinyimagenet | CUB200 | STANFORD120 | MIT67')
parser.add_argument('--dataroot', default='./dataset/CUB_200_2011/', type=str, help='data directory') # '.../CUB_200_2011/' | '.../StandFordDogs' | './dataset/'(MIT67)

parser.add_argument("--aug_nums", type=int, default=2)  #

args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

save_path = './checkpoint/LRMS_R18_CUB'
save_path_pth = os.path.join(save_path, 'ckpt.pth')


if not args.cls:
    trainloader, valloader = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size)  ## CUB / MIT / StanFord
else:
    trainloader, valloader = load_dataset(args.dataset, args.dataroot, 'pair', batch_size=args.batch_size)  ## CUB / MIT / StanFord



num_class = trainloader.dataset.num_classes



# Model
print('==> Building model..')

# net = small_network(class_num=num_class)
net = resnet18(num_classes=num_class)
# net_exp = small_network()
# net = VGG('VGG16')
# net = torchvision.models.vgg16_bn(num_classes=num_class)
# net = torchvision.models.resnet18(num_classes=num_class)
net = net.cuda()
# net_teacher = net_teacher.cuda()

# checkpoint = torch.load('./checkpoint/resnet50/ckpt.pth')
# checkpoint = torch.load('./checkpoint/_956_1_1_model/ckpt.pth')
# net_teacher.load_state_dict(checkpoint['net'])
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/ckpt.pth')
    checkpoint = torch.load(save_path_pth)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
bce_WL = nn.BCEWithLogitsLoss()
ls_loss = LabelSmoothing(smoothing=0.1)
L2Loss = nn.MSELoss()



def loss_fn_kd_crd(outputs, teacher_outputs, temperature):
    p_s = F.log_softmax(outputs/temperature, dim=1)
    p_t = F.softmax(teacher_outputs/temperature, dim=1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (temperature**2) / outputs.shape[0]

    return loss



def gram_matrix(f1, f2):

    # f1 = torch.unsqueeze(f1, 1)
    f1 = f1.view(f1.size(0), 1, -1)
    # f2 = torch.unsqueeze(f2, 1)
    f2 = f2.view(f2.size(0), 1, -1)
    tmp = []
    tmp.append(f1)
    tmp.append(f2)
    tmp = torch.cat(tmp, dim=1)
    gram = torch.bmm(tmp, tmp.permute(0, 2, 1))
    return gram

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
# torch.set_printoptions(profile="full")

def select_as_teacher(out_all, target):

    out_all_softmax = F.softmax(out_all, dim=1)

    good_each_value = torch.zeros((num_class, 1), device='cuda')

    good_for_each_category = torch.zeros((num_class, out_all.size(1)), device='cuda')
    good = torch.zeros((len(target), out_all.size(1)), device='cuda')

    # for i in range(len(target)):
    #     if out_all[i][target[i]] > good_for_each_category[target[i]][target[i]]:
    #         good_for_each_category[target[i]] = out_all[i]

    for i in range(len(target)):
        if out_all_softmax[i][target[i]] > good_each_value[target[i]]:
            good_each_value[target[i]] = out_all_softmax[i][target[i]]
            good_for_each_category[target[i]] = out_all[i]

    for i in range(len(target)):
        good[i] = good_for_each_category[target[i]]

    return good



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)



# Training
save_change = False


def train(epoch, pre_data):
    global save_change
    # save_change = False
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss_all = 0
    train_loss_sub = 0
    train_loss_intra = 0

    correct_sub = 0
    correct_all = 0
    correct_intra = 0


    total = 0
    # for batch_idx, (inputs, targets) in enumerate(trainloader): ## 
    for batch_idx, data in enumerate(trainloader): ## 

        inputs, targets = data
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        if pre_data != None:
            pre_inputs, pre_targets = pre_data
            if torch.cuda.is_available():
                pre_inputs = pre_inputs.cuda()
                pre_targets = pre_targets.cuda()

            inputs = torch.cat([inputs[:, 0, ...], pre_inputs[:, 1, ...]])
            targets = torch.cat([targets, pre_targets])
        else:
            inputs = inputs[:, 0, ...]
            targets = targets


        pre_data = data  



        out_all, exp_fea = net(inputs)


        good_students = select_as_teacher(out_all, targets)



        loss_all = criterion(out_all, targets)
        loss_kl = loss_fn_kd_our(out_all, good_students.detach(), temperature_good=args.temperature_good,
                                 temperature_other=args.temperature_other)


        loss = loss_all + loss_kl  ## 
        loss = min((epoch+1) / args.warmup, 1.0) * loss

        loss.backward()
        optimizer.step()

        train_loss_all += loss_all.item()
        train_loss_sub += loss_kl.item()

        _, predicted_all = out_all.max(1)
        # _, predicted_sub = out_sub.max(1)
        # _, predicted_intra = out_intra.max(1)

        total += targets.size(0)

        correct_all += predicted_all.eq(targets).sum().item()

    epoch_loss_all = train_loss_all / (batch_idx + 1)
    epoch_loss_sub = train_loss_sub / (batch_idx + 1)
    
    epoch_acc_all = correct_all / total

    print('Train Loss_ce: {:.4f} Acc: {:.4f}'.format(epoch_loss_all, epoch_acc_all))
    print('Train epoch_loss_sub: {:.4f}'.format(epoch_loss_sub))

    print('-' * 20)

    return pre_data


def test(epoch):
    global best_acc
    global save_change
    net.eval()

    test_loss_all = 0

    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_losses = AverageMeter()

    targets_list = []
    confidences = []

    correct_all = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # for ECE, AURC, EAURC
            targets_numpy = targets.cpu().numpy()
            targets_list.extend(targets_numpy.tolist())

            # out_all= net(inputs)
            out_all, _ = net(inputs)

            # for ECE, AURC, EAURC
            softmax_predictions = F.softmax(out_all, dim=1)
            softmax_predictions = softmax_predictions.cpu().numpy()
            for values_ in softmax_predictions:
                confidences.append(values_.tolist())


            _, predicted_all = out_all.max(1)
            total += targets.size(0)
            correct_all += predicted_all.eq(targets).sum().item()

            loss_all = criterion(out_all, targets)
            val_losses.update(loss_all.item(), inputs.size(0))
            # test_loss_all += loss_all.item()

            # Top1, Top5 Err
            err1, err5 = accuracy(out_all.data, targets, topk=(1, 5))
            val_top1.update(err1.item(), inputs.size(0))
            val_top5.update(err5.item(), inputs.size(0))

        if is_main_process():
            ece, aurc, eaurc = metric_ece_aurc_eaurc(confidences,
                                                     targets_list,
                                                     bin_size=0.1)

        print('[Epoch {}] [val_loss {:.3f}] [val_top1_acc {:.3f}] [val_top5_acc {:.3f}] [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}] [correct/total {}/{}]'.format(
                epoch,
                val_losses.avg,
                val_top1.avg,
                val_top5.avg,
                ece,
                aurc,
                eaurc,
                correct_all,
                total))

    # Save checkpoint.
    acc = val_top1.avg
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path_pth)
        best_acc = acc
        save_change = True
        print('save change!')


if __name__ == '__main__':
    pre_data = None
    for epoch in range(start_epoch, start_epoch+240):
        pre_data = train(epoch, pre_data)
        test(epoch)
        scheduler.step()
