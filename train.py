import sys
import os
import random
import warnings
import faulthandler
import time

import cv2
import numpy as np

import torch
from torch.backends import cudnn
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import draw_hyps, draw_heatmap
# from models.networks_regression import HyperRegression
from models.cond_inn import CondINNWrapper
from models import feature_net

from args import get_args
from utils import AverageValueMeter, set_random_seed, resume, save
from dataset_coco import SamplePointData

import mmfp_utils
from utils import draw_hyps

faulthandler.enable()


def get_grid_logprob(args,
                     height, width, x, model, class_cond
                     ):
    x_sp = np.linspace(0, width - 1, width // 4)
    y = np.linspace(0, height - 1, height // 4)
    X, Y = np.meshgrid(x_sp, y)
    XY = np.array([X.ravel(), Y.ravel()]).T
    _, _, (log_py_grid, log_px_grid) = model.get_logprob(
        x,
        torch.tensor(XY, dtype=torch.float32).unsqueeze(
            0).to(args.gpu), class_cond
    )
    return (X, Y), (log_px_grid.detach().cpu().numpy(), log_py_grid.detach().cpu().numpy())


def parse_first_example_to_npy(input_tensor, gt_mask_tensor, pred_mask_tensor, class_label):

    rgb_im = input_tensor[0].cpu(
    ).detach().numpy().squeeze().copy()
    rgb_im = np.transpose(rgb_im[:3], (1, 2, 0))
    rgb_im = ((rgb_im + 1) * 255/2.).astype(np.uint8)

    seg_im = gt_mask_tensor[0].cpu().detach().numpy().squeeze()
    seg_im *= 255
    seg_im = np.tile(np.expand_dims(cv2.resize(seg_im, (256, 256)), axis=2), (1, 1, 3))
    seg_im = seg_im.astype(np.uint8)

    pred_seg_im = pred_mask_tensor[0]
    pred_seg_im = pred_seg_im[0].unsqueeze(0).repeat(3, 1, 1)
    pred_seg_im = pred_seg_im.cpu().detach().numpy().squeeze()
    pred_seg_im *= 255
    pred_seg_im = np.transpose(pred_seg_im, (1, 2, 0))
    pred_seg_im = cv2.resize(pred_seg_im, (256, 256))
    pred_seg_im = pred_seg_im.astype(np.uint8)

    return rgb_im, seg_im, pred_seg_im, class_label[0]

def main_worker(gpu, save_dir, ngpus_per_node, args):
    # basic setup
    cudnn.benchmark = True
    normalize = False
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    img_dims = (256, 256)
    model = CondINNWrapper(args, img_dims=img_dims).cuda(args.gpu)

    torch.cuda.set_device(args.gpu)

    start_epoch = 0
    """
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
    """

    # main training loop
    start_time = time.time()
    loss_avg_meter = AverageValueMeter()
    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    # initialize datasets and loaders

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    train_set = SamplePointData(args,
                                split='train2017', root=args.data_dir, width=256, height=256)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)
    test_set = SamplePointData(args,
                               split='val2017', root=args.data_dir, width=256, height=256)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    # train iteration
    for epoch in range(start_epoch, args.epochs):
        # adjust the learning rate
        #if (epoch + 1) % args.exp_decay_freq == 0:
        #    scheduler.step(epoch=epoch)

        # train for one epoch
        print("Epoch starts:")
        for bidx, (input_tensor, gt_mask_tensor, class_condition, class_label) in enumerate(train_loader):
            # x : [args.batch_size, 5, W, H]
            # y : [args.batch_size, 30, 2]s
            input_tensor = input_tensor.float().to(args.gpu)
            gt_mask_tensor = gt_mask_tensor.float().to(args.gpu)
            # gt_mask_tensor += torch.empty(gt_mask_tensor.shape).normal_(mean=0, std=0.005).cuda()
            step = bidx + len(train_loader) * epoch
            model.train()

            # backpropagate
            reverse_sample, loss = model(input_tensor, gt_mask_tensor,
                  class_condition)

            # torch.tensor -> np.array
            rgb_im, gt_seg_im, pred_seg_im, label_str = parse_first_example_to_npy(input_tensor, gt_mask_tensor, reverse_sample, class_label)

            print('current label class: ', label_str)

            cv2.imshow('test', np.hstack([rgb_im, gt_seg_im, pred_seg_im]))
            key = cv2.waitKey(1)

            loss_avg_meter.update(loss.item())
            # multi_ce_loss_avg_meter.update(ce_loss)
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print("[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loss %2.5f"
                      % (args.rank, epoch, bidx, len(train_loader), duration, loss_avg_meter.avg))

            continue

            _, height, width = input_tensor[0].shape

            (X, Y), (log_px_grid, log_py_grid) = get_grid_logprob(args,
                                                                  height, width, input_tensor[0].unsqueeze(0), model, class_condition[0].unsqueeze(0))

            epoch_save_dir = os.path.join(
                save_dir, 'images', 'epoch-' + str(epoch))

            if not os.path.exists(epoch_save_dir):
                os.makedirs(epoch_save_dir)

            save_path = os.path.join(save_dir, 'images',
                                     'epoch-' + str(epoch))

            draw_heatmap(rgb_image.get(),
                         log_px_pred=log_px_grid,
                         X=X, Y=Y,
                         save_path=os.path.join(
                             save_path, f"{bidx}-train-heatmap.png")
                         )

            img = cv2.imread(os.path.join(
                save_path, f"{bidx}-train-heatmap.png"))
            img = cv2.resize(img, (256, 256))

            cv2.imshow('heatmap', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break


                # print("Memory")
                # print(process.memory_info().rss / (1024.0 ** 3))
        # save visualizations
        if (epoch + 1) % args.viz_freq == 0:
            # reconstructions
            print('test start')
            model.eval()
            for bidx, (input_tensor, gt_mask_tensor, class_condition, class_label) in enumerate(test_loader):
                input_tensor = input_tensor.float().to(args.gpu)
                gt_mask_tensor = gt_mask_tensor.float().to(args.gpu)
                if args.timeit:
                    t1 = time.time()
                pred_seg_mask = model.decode(
                    input_tensor, class_condition)
                if args.timeit:
                    t2 = time.time()
                    print('inference speed (1/s): ', 1.0/(t2-t1))

                #pred_seg_mask = pred_seg_mask.mean(dim=0)
                #import pdb; pdb.set_trace()

                rgb_im, gt_seg_im, pred_seg_im, label_str = parse_first_example_to_npy(input_tensor, gt_mask_tensor, pred_seg_mask, class_label)

                save_img = np.hstack([rgb_im, gt_seg_im, pred_seg_im])
                cv2.putText(save_img, label_str, (save_img.shape[1]//2, save_img.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                epoch_save_dir = os.path.join(
                save_dir, 'images', 'epoch-' + str(epoch))

                if not os.path.exists(epoch_save_dir):
                    os.makedirs(epoch_save_dir)

                save_path = os.path.join(save_dir, 'images',
                                         'epoch-' + str(epoch))
                cv2.imwrite(os.path.join(
                    save_path, str(bidx) + '.jpg'), save_img)

                # calculate log probability
                """
                log_py, log_px, (log_py_grid, log_px_grid) = model.get_logprob(
                    input_tensor, points_label, class_condition)

                log_py = log_py.cpu().detach().numpy().squeeze()
                log_px = log_px.cpu().detach().numpy().squeeze()

                hyps_name = f"{bidx}-hyps.jpg"
                print(hyps_name)
                print("nll_x", str(-1.0 * log_px))
                print("nll_y", str(-1.0 * log_py))
                print("nll_(x+y)", str(-1.0 * (log_px + log_py)))

                # nll_px_sum = nll_px_sum + -1.0 * log_px
                # nll_py_sum = nll_py_sum + -1.0 * log_py
                # counter = counter + 1.0

                # multimod_emd = mmfp_utils.wemd_from_pred_samples(
                #    estimated_points)
                # multimod_emd_sum += multimod_emd
                # print("multimod_emd", multimod_emd)
                """

                """
                _, _, height, width = input_tensor.shape

                (X, Y), (log_px_grid, log_py_grid) = get_grid_logprob(args,
                                                                      height, width, input_tensor, model, class_condition)

                print('Save heatmap: %s' % os.path.join(
                    save_path, f"{bidx}-heatmap.png"))
                print('- max prob: ', np.exp(log_px_grid.max()))
                print('- min prob: ', np.exp(log_px_grid.min()))

                draw_heatmap(rgb_heatmap_img,
                             log_px_pred=log_px_grid,
                             X=X, Y=Y,
                             save_path=os.path.join(
                                 save_path, f"{bidx}-heatmap.png")
                             )
                """

        if (epoch + 1) % args.save_freq == 0:
            save(model, model.get_optimizer(), epoch + 1,
                 os.path.join(save_dir, 'checkpoint-%d.pt' % epoch))
            save(model, model.get_optimizer(), epoch + 1,
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
