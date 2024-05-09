
import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
import torch.optim.lr_scheduler

LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def poly_lr_scheduler(args, optimizer, epoch, power=2):
    lr = round(args.lr * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr



def train(args, train_loader, model, criterion, optimizer, epoch):
    model.train()

    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
    pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in pbar:
        if args.onGPU == True:
            input = input.cuda().float() / 255.0        
        output = model(input)
        
        # target=target.cuda()
        optimizer.zero_grad()

        focal_loss,tversky_loss,loss = criterion(output,target)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                                     (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, loss.item()))
        

def train16fp(args, train_loader, model, criterion, optimizer, epoch,scaler):
    model.train()
    print("16fp-------------------")
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
    pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in pbar:
        optimizer.zero_grad()
        if args.onGPU == True:
            input = input.cuda().float() / 255.0        
        output = model(input)
        with torch.cuda.amp.autocast():
            focal_loss,tversky_loss,loss = criterion(output,target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                                     (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, loss.item()))


@torch.no_grad()
def val(val_loader, model):

    model.eval()


    DA=SegmentationMetric(2)
    LL=SegmentationMetric(2)

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    total_batches = len(val_loader)
    
    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.cuda().float() / 255.0
            # target = target.cuda()

        input_var = input
        target_var = target

        # run the mdoel
        with torch.no_grad():
            output = model(input_var)

        out_da,out_ll=output
        target_da,target_ll=target

        _,da_predict=torch.max(out_da, 1)
        _,da_gt=torch.max(target_da, 1)

        _,ll_predict=torch.max(out_ll, 1)
        _,ll_gt=torch.max(target_ll, 1)
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())


        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc,input.size(0))
        da_IoU_seg.update(da_IoU,input.size(0))
        da_mIoU_seg.update(da_mIoU,input.size(0))


        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())


        ll_acc = LL.pixelAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()

        ll_acc_seg.update(ll_acc,input.size(0))
        ll_IoU_seg.update(ll_IoU,input.size(0))
        ll_mIoU_seg.update(ll_mIoU,input.size(0))

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)
    return da_segment_result,ll_segment_result





def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

def validation(model,valLoader):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''

    # load the model
    # model = net.TwinLiteNet()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        # model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
        

    # valLoader = torch.utils.data.DataLoader(
    #     myDataLoader.colabIADDataset(valid=True),
    #     batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
    
    # model.load_state_dict(torch.load(args.weight))
    model.eval()
    # example = torch.rand(1, 3, 360, 640).cuda()
    # model = torch.jit.trace(model, example)
    da_segment_results,ll_segment_results = val(valLoader, model)

    msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
                          da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
    print(msg)

