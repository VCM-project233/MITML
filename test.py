from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from eval_metrics import eval_sysu, eval_regdb, evaluate
from model_main import embed_net, modal_Classifier
from utils import *
from loss import OriTripletLoss
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import math

from data_manager import VCM
from data_loader import VideoDataset_train, VideoDataset_test, VideoDataset_train_evaluation
import transforms as T

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='VCM', help='dataset name: VCM')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--resume', '-r', default='CUHK60_drop_0.2_2_8_lr_0.1_seed_0_a_1t2v_map_best.t', type=str,help='resume from checkpoint')

parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='./save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log_vcm/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--part', default=3, type=int,
                    metavar='tb', help=' part number')
parser.add_argument('--method', default='id+tri', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.2, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=2, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda0', default=1.0, type=float,
                    metavar='lambda0', help='graph attention weights')
parser.add_argument('--graph', action='store_true', help='either add graph attention or not')
parser.add_argument('--wpa', action='store_true', help='either add weighted part attention')

args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#set_seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dataset = args.dataset

np.random.seed(1)

#添加
seq_lenth = 6
test_batch = 32
data_set = VCM()
log_path = args.log_path + 'VCM_log/'
test_mode = [1, 2]
height = args.img_h
width = args.img_w

 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
# n_class = data_set.num_train_pids
n_class = 500
nquery = data_set.num_query_tracklets

ngall = data_set.num_gallery_tracklets
nquery_1 = data_set.num_query_tracklets_1
ngall_1 = data_set.num_gallery_tracklets_1

print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop=args.drop, part=args.part, arch=args.arch, wpa=args.wpa)
net.to(device)    
cudnn.benchmark = True

print('==> Resuming from checkpoint..')
checkpoint_path = args.model_path
if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    # model_path = checkpoint_path + 'test_best.t'
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        # pdb.set_trace()
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}!!!!!!!!!!'.format(args.resume))


# if args.method =='id':
criterion = nn.CrossEntropyLoss()
criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.Pad(10),

    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset == 'VCM':
    rgb_pos, ir_pos = GenIdx(data_set.rgb_label, data_set.ir_label)
queryloader = DataLoader(
    VideoDataset_test(data_set.query, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

galleryloader = DataLoader(
    VideoDataset_test(data_set.gallery, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

queryloader_1 = DataLoader(
    VideoDataset_test(data_set.query_1, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

galleryloader_1 = DataLoader(
    VideoDataset_test(data_set.gallery_1, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

print('Data Loading Time:\t {:.3f}'.format(time.time()-end))

feature_dim = 2048
if args.arch =='resnet50':
    pool_dim = 2048
elif args.arch =='resnet18':
    pool_dim = 512

def test1():
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery(rgb) Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            input = imgs
            input = Variable(input.cuda())
            label = pids
            batch_num = input.size(0)
            feat = net(input, input, 0, test_mode[0], seq_len=seq_lenth)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
            #
            g_pids.extend(pids)
            g_camids.extend(camids)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query(ir) Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            input = imgs
            label = pids

            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input, input, 0, test_mode[1], seq_len=seq_lenth)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    return cmc, mAP


def test2():
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery(ir) Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall_1, 2048))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader_1):
            input = imgs
            input = Variable(input.cuda())
            label = pids
            batch_num = input.size(0)
            feat= net(input, input, 0, test_mode[1], seq_len=seq_lenth)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
            #
            g_pids.extend(pids)
            g_camids.extend(camids)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query(rgb) Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery_1, 2048))
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader_1):
            input = imgs
            label = pids

            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat= net(input, input, 0, test_mode[0], seq_len=seq_lenth)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)


    return cmc, mAP

# testing
cmc, mAP = test1()
#---------visible to infrared---------
cmc_1, mAP_1 = test2()
# log output
print('ir2rgb:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
    cmc[0], cmc[4], cmc[9], cmc[19], mAP))

print('rgb2ir:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
    cmc_1[0], cmc_1[4], cmc_1[9], cmc_1[19], mAP_1))