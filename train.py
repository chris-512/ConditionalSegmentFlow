import sys
import os
import torch
import cv2
import torch.distributed as dist
import warnings
import torch.distributed
import numpy as np
import random
import faulthandler
import torch.multiprocessing as mp
import time
from models.networks_regression import HyperRegression
from torch import optim
from args import get_args
from torch.backends import cudnn
from utils import AverageValueMeter, set_random_seed, resume, save
from dataset_coco import SamplePointData

from utils import draw_hyps

faulthandler.enable()


def main_worker(gpu, save_dir, ngpus_per_node, args):
    # basic setup
    cudnn.benchmark = True
    normalize = False
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = HyperRegression(args, input_width=256, input_height=256)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    start_epoch = 0
    optimizer = model.make_optimizer(args)
    if args.resume_checkpoint is None and os.path.exists(os.path.join(save_dir, 'checkpoint-latest.pt')):
        args.resume_checkpoint = os.path.join(
            save_dir, 'checkpoint-latest.pt')  # use the latest checkpoint
    if args.resume_checkpoint is not None:
        if args.resume_optimizer:
            model, optimizer, start_epoch = resume(
                args.resume_checkpoint, model, optimizer, strict=(not args.resume_non_strict))
        else:
            model, _, start_epoch = resume(
                args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))
        print('Resumed from: ' + args.resume_checkpoint)

    # initialize the learning rate scheduler
    if args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / \
                float(0.5 * args.epochs)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule)
    else:
        assert 0, "args.schedulers should be either 'exponential' or 'linear'"

    # main training loop
    start_time = time.time()
    entropy_avg_meter = AverageValueMeter()
    latent_nats_avg_meter = AverageValueMeter()
    point_nats_avg_meter = AverageValueMeter()
    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    # initialize datasets and loaders

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    train_set = SamplePointData(
        split='train2017', root=args.data_dir, width=256, height=256)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)
    test_set = SamplePointData(
        split='val2017', root=args.data_dir, width=256, height=256)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    # train iteration
    for epoch in range(start_epoch, args.epochs):
        # adjust the learning rate
        if (epoch + 1) % args.exp_decay_freq == 0:
            scheduler.step(epoch=epoch)

        # train for one epoch
        print("Epoch starts:")
        for bidx, data in enumerate(train_loader):
            # if bidx < 2:
            x, y = data
            # x : [args.batch_size, 5, W, H]
            # y : [args.batch_size, 30, 2]
            x = x.float().to(args.gpu)
            y = y.float().to(args.gpu)

            step = bidx + len(train_loader) * epoch
            model.train()
            recon_nats = model(x, y, optimizer, step, None)

            pos = y[0].cpu().detach().numpy().squeeze()
            pos = (
                (x[0].shape[1], x[0].shape[2]) * pos).astype(int)
            img = x[0].cpu().detach().numpy().squeeze().copy()
            img = np.transpose(img[:3], (1, 2, 0))
            img = ((img + 1) * 255/2.).astype(np.uint8)
            img = cv2.UMat(img)

            for i, (posy, posx) in enumerate(pos):
                if posy < 0 or posy >= x[0].shape[1]:
                    continue
                if posx < 0 or posx >= x[0].shape[2]:
                    continue
                if i < 30:
                    cv2.circle(img, (posx, posy), 2, (255, 0, 0), -1)
                else:
                    cv2.circle(img, (posx, posy), 2, (0, 0, 255), -1)

            cv2.imshow('test', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            point_nats_avg_meter.update(recon_nats.item())
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print("[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] PointNats %2.5f"
                      % (args.rank, epoch, bidx, len(train_loader), duration, point_nats_avg_meter.avg))
                # print("Memory")
                # print(process.memory_info().rss / (1024.0 ** 3))
        # save visualizations
        if (epoch + 1) % args.viz_freq == 0:
            # reconstructions
            model.eval()
            for bidx, data in enumerate(test_loader):
                x, _ = data
                x = x.float().to(args.gpu)
                if args.timeit:
                    t1 = time.time()
                _, y_pred = model.decode(x, 250)
                if args.timeit:
                    t2 = time.time()
                    print('inference speed (1/s): ', 1.0/(t2-t1))
                y_pred = y_pred.cpu().detach().numpy().squeeze()
                y_pred_scaled = (
                    (x[0].shape[1], x[0].shape[2]) * y_pred).astype(int)
                img = x.cpu().detach().numpy().squeeze().copy()
                img = np.transpose(img[:3], (1, 2, 0))
                img = ((img + 1) * 255/2.).astype(np.uint8)
                img = cv2.UMat(img)

                pos_list = []
                for (posy, posx) in y_pred_scaled:
                    if posy < 0 or posy >= x[0].shape[1]:
                        continue
                    if posx < 0 or posx >= x[0].shape[2]:
                        continue
                    # import pdb
                    # pdb.set_trace()
                    pos_list.append((posy, posx))
                    cv2.circle(img, (posx, posy), 2, (0, 255, 0), -1)

                pos_array = np.array(pos_list)
                mean = np.mean(pos_array, axis=0).astype(np.int32)
                cv2.circle(img, (mean[1], mean[0]), 4, (0, 0, 255), -1)

                epoch_save_dir = os.path.join(
                    save_dir, 'images', 'epoch-' + str(epoch))
                if not os.path.exists(epoch_save_dir):
                    os.makedirs(epoch_save_dir)

                cv2.imwrite(os.path.join(save_dir, 'images',
                                         'epoch-' + str(epoch), str(bidx) + '.jpg'), img)

        if (epoch + 1) % args.save_freq == 0:
            save(model, optimizer, epoch + 1,
                 os.path.join(save_dir, 'checkpoint-%d.pt' % epoch))
            save(model, optimizer, epoch + 1,
                 os.path.join(save_dir, 'checkpoint-latest.pt'))


def main():
    # command line args
    args = get_args()
    if args.root_dir is None:
        save_dir = os.path.join("checkpoints", args.log_name)
    else:
        save_dir = os.path.join(args.root_dir, "checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'images'))

    with open(os.path.join(save_dir, 'command.sh'), 'w') as f:
        f.write('python -X faulthandler ' + ' '.join(sys.argv))
        f.write('\n')

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.sync_bn:
        assert args.distributed

    print("Arguments:")
    print(args)

    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(save_dir, ngpus_per_node, args))
    else:
        main_worker(args.gpu, save_dir, ngpus_per_node, args)


if __name__ == '__main__':
    main()
