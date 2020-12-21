import argparse
import logging
import os
import random
import shutil
import time
from pathlib import Path
from warnings import warn

import math
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
    compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
    check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution, set_logging, init_seeds)
from utils.google_utils import attempt_download
from utils.torch_utils import ModelEMA, select_device, intersect_dicts

logger = logging.getLogger(__name__)

#
# @param hyp 超参hyperparameters
# @param opt 输入参数
# @param device 训练的设备，cpu或者cuda
# @param tb_writer tensorboard的输出通道
#
def train(hyp, opt, device, tb_writer=None):
    
    #确定训练结果的输出位置和文件
    logger.info(f'Hyperparameters {hyp}')
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory 默认是runs/exp0
    #权重的输出位置
    wdir = log_dir / 'weights'  # weights directory
    os.makedirs(wdir, exist_ok=True)
    #权重文件
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = str(log_dir / 'results.txt')
	
    # 获得迭代次数、batch_size、权重等参数
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Save run settings 保存运行时的超参和参数在hyp.yaml和opt.yaml文件中
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'
    # 设置随机数种子，与并行训练相关
    init_seeds(2 + rank)
    # 加载训练数据描述文件，例如coco.yaml
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    # 检查训练数据是否存在，不存在则下载
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    # 获取训练和测试数据的文件清单
    train_path = data_dict['train']
    test_path = data_dict['val']
    # 获取训练数据的类别数量及名称
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    # 获得预训练权重
    pretrained = weights.endswith('.pt')
    if pretrained:
        # 如果权重文件不存在则下载
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        # 加载权重文件
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        # 加载模型文件，例如yolov5s.yaml，如果没有指定模型文件则根据权重文件确定模型
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
        # 预训练权重里面保存了默认coco数据集对应的anchor，加载预训练权重就会覆盖掉用户自定义了anchor
		# 如果opt.cfg存在(表示采用预训练权重进行训练)就设置去除anchor，使用用户自定义的anchor
		# 如果是resume，就不去除权重中的anchor，接着训练
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        # 显示加载预训练权重的键值对和创建模型的键值对
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        # 没有权重文件，直接根据模型文件创建
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create

    # Freeze
    # 冻结模型的部分层，设置需要冻结的层名即可
    # 具体可以查看https://github.com/ultralytics/yolov5/issues/679
    freeze = ['', ]  # parameter names to freeze (full or partial)
    if any(freeze):
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    # 设置权重衰减系数
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    # 将模型分成三组分别优化
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    # 选用优化器，设置pg0组的优化方式
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # 设置pg1组的优化方式
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # 设置pg2组的优化方式
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # 打印优化信息
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # 设置学习率衰减，这里使用余弦退火方式
    # 就是根据以下公式lf,epoch和超参数hyp['lrf']进行衰减	
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    # 初始化开始训练的epoch和最好的结果
    # best_fitness是以[0.0, 0.0, 0.1, 0.9]为系数并乘以[精确度, 召回率, mAP@0.5, mAP@0.5:0.95]再求和所得
    # 根据best_fitness来保存best.pt
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        # 加载优化器与best_fitness
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        # 创建训练结果results.txt文件
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
		# 加载已训练的迭代次数
        start_epoch = ckpt['epoch'] + 1
		
        # 如果resume，则备份权重
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
            shutil.copytree(wdir, wdir.parent / f'weights_backup_epoch{start_epoch - 1}')  # save previous weights
        
        #如果新设置的epochs小于已训练的，则视新设置的epochs为需要再训练的迭代次数，而不是总的训练次数
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    # 获取模型总步长
    gs = int(max(model.stride))  # grid size (max stride)
    # 检查输入图片尺寸，确保尺寸是步长的整倍数
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    # 并行模式，针对单机多卡，参照:https://github.com/ultralytics/yolov5/issues/475
    # rank设置-1，且GPU数量大于1，则启动并行模式
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    # 使用跨卡同步BN，仅在DDP模式下有效
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Exponential moving average
    # 为模型创建EMA指数滑动平均,如果GPU进程数大于1,则不创建
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    # 如果rank不等于-1,则使用DistributedDataParallel模式
    # local_rank为gpu编号,rank为进程,例如rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU。
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Trainloader
	# 创建训练数据集
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                            rank=rank, world_size=opt.world_size, workers=opt.workers)
    # 获取标签的最大类别值，并与类别总数比较
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        # 更新ema模型的updates参数,保持ema的平滑性
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        # 创建测试集dataloader
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,
                                       hyp=hyp, augment=False, cache=opt.cache_images and not opt.notest, rect=True,
                                       rank=-1, world_size=opt.world_size, workers=opt.workers)[0]  # testloader

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            plot_labels(labels, save_dir=log_dir)
            if tb_writer:
                # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
                tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Model parameters
    # 根据自己数据集的类别数设置分类损失的系数
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    # 设置类别数，超参数
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    """
    设置giou的值在objectness loss中做标签的系数, 使用代码如下
    tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)
    这里model.gr=1，也就是说完全使用标签框与预测框的giou值来作为该预测框的objectness标签
    """
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # 根据各类别的数量初始化类别权重，数量越多，权重越低
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    # 获得类别名称
    model.names = names

    # Start training
    # 开始训练
    t0 = time.time()
    # 获取热身训练的迭代次数
    nw = max(round(hyp['warmup_epochs'] * nb), 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
	# 初始化mAP和results
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    """
    设置学习率衰减所进行到的轮次，
    目的是打断训练后，--resume接着训练也能正常的衔接之前的训练进行学习率衰减
    """
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 混合精度训练中的梯度尺度变换，可以显著缩小显存占用并提升运算速度
    scaler = amp.GradScaler(enabled=cuda)
    """
    打印训练和测试输入图片分辨率
    加载图片时调用的cpu进程数
    从哪个epoch开始训练
    """
    logger.info('Image sizes %g train, %g test\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (imgsz, imgsz_test, dataloader.num_workers, log_dir, epochs))
				
    # 开始训练
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            """
           如果设置进行图片采样策略，
           则根据前面初始化的图片采样权重model.class_weights以及maps配合每张图片包含的类别数
           通过random.choices生成图片索引indices从而进行采样
           """
            if rank in [-1, 0]:
                # 获得类别权重
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                # 获得图像权重
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                # 生成随机的权重索引
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            # 如果是DDP模式则广播采样策略到其他的节点
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            # DDP模式下打乱数据, ddp.sampler的随机采样数据是基于epoch+seed作为随机种子，
            # 每次epoch不同，随机种子就不同
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        if rank in [-1, 0]:
            # 创建进度条，方便训练时信息的展示
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # 计算迭代的次数
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 将图像的8位映射为float32
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            """
            热身训练(前nw次迭代)
            在前nw次迭代中，根据以下方式选取accumulate和学习率
            """
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    """
                    bias的学习率从0.1下降到基准学习率lr*lf(epoch)，
                    其他的参数学习率从0增加到lr*lf(epoch).
                    lf为上面设置的余弦退火的衰减函数
                    """
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    # 动量momentum也从0.9慢慢变到hyp['momentum'](default=0.937)
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            # 设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            # 前向传播，混合精度
            with amp.autocast(enabled=cuda):
                # 前向传播
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size
                if rank != -1:
                    # 平均不同gpu之间的梯度
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode

            # Backward
            # 反向传播
            scaler.scale(loss).backward()

            # Optimize
            # 模型反向传播accumulate次之后再根据累积的梯度更新一次参数
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                # 进度条显示以上信息
                pbar.set_description(s)

                # Plot
                # 将前三次迭代batch的标签框在图片上画出来并保存
                if ni < 3:
                    f = str(log_dir / f'train_batch{ni}.jpg')  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    # if tb_writer and result is not None:
                    # tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        # 进行学习率衰减
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema:
                # 更新EMA的属性
                # 添加include的属性
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            # 判断该epoch是否为最后一轮
            final_epoch = epoch + 1 == epochs
            # 对测试集进行测试，计算mAP等指标
            # 测试时使用的是EMA模型
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps, times = test.test(opt.data,
                                                 batch_size=total_batch_size,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=log_dir,
                                                 plots=epoch == 0 or final_epoch)  # plot first and last

            # Write
            # 将指标写入result.txt
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            # 如果设置opt.bucket, 上传results.txt到谷歌云盘
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Tensorboard
            # 添加指标，损失等信息到tensorboard显示
            if tb_writer:
                tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                        'x/lr0', 'x/lr1', 'x/lr2']  # params
                for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            # 更新best_fitness
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            """
            保存模型，还保存了epoch，results，optimizer等信息，
            optimizer将不会在最后一轮完成后保存
            model保存的是EMA的模型
            """
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        """
        模型训练完后，strip_optimizer函数将optimizer从ckpt中去除；
        并且对模型进行model.half(), 将Float32的模型->Float16，
        可以减少模型大小，提高inference速度
        """
        n = opt.name if opt.name.isnumeric() else ''
        fresults, flast, fbest = log_dir / f'results{n}.txt', wdir / f'last{n}.pt', wdir / f'best{n}.pt'
        for f1, f2 in zip([wdir / 'last.pt', wdir / 'best.pt', results_file], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                if str(f2).endswith('.pt'):  # is *.pt
                    strip_optimizer(f2)  # strip optimizer
                    # 上传结果到谷歌云盘
                    os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket else None  # upload
        # Finish
        # 可视化results.txt文件
        if not opt.evolve:
            plot_results(save_dir=log_dir)  # save as results.png
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    # 释放显存
    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # weights 初始权重文件
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    # cfg 模型配置文件，网络结构
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # data 数据集配置文件
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    # hyp 超参数文件
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    # epochs 训练的总轮次
    parser.add_argument('--epochs', type=int, default=300)
    # batch-size 批次大小
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    # img-size 输入图片分辨率大小
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    # rect 是否采用矩形训练
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # resume 继续最近的训练
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # nosave 只保存最终的checkpoint，默认FALSE
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # notest 只测试最后一个轮次，默认FALSE
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    # noautoanchor 不自动调整anchor，默认FALSE
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    # evolve 超参数进化，默认FALSE
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    # bucket 谷歌云盘bucket
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # cache-images 是否提前缓存图片到内存，以加快训练，默认FALSE
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    # image-weights 使用带权重的图像选择进行训练
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # name 如果指定将数据文件夹exp{N} 改成 exp{N}_{name}的形式
    parser.add_argument('--name', default='', help='renames experiment folder exp{N} to exp{N}_{name} if supplied')
    # device 指定训练的设备，cuda或CPU
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # multi-scale 进行多尺度训练
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # single-cls 训练单类别数据集
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    # adam 使用adam优化器
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    # sync-bn 使用SyncBatchNorm，仅在DDP模式下可用
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # local_rank DDP参数，不要修改
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    # logdir 存放运行结果的目录，默认是runs
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    # workers 数据加载器dataloader的最大worker数量
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    opt = parser.parse_args() # 解析参数，把所有的参数变成opt的成员变量

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        log_dir = Path(ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(log_dir / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)

    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/exp1 创建默认的输出文件夹，每次文件夹的后缀自动加1

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.logdir}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0

        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.logdir) / 'evolve' / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
