import sys, os, argparse, time
import mxnet
from mxnet import autograd, nd

import numpy as np
import cv2
import matplotlib.pyplot as plt

import datasets_112, hopenet_112

Parent_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=30, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.00001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default=os.path.join(Parent_DIR, 'mxnet_hopnet'), type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default=os.path.join(Parent_DIR, 'mxnet_hopnet.txt'), type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=1, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    alpha = args.alpha
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = 1
    ctx_init = mxnet.gpu(gpu)

    snapshot_dir = os.path.join(Parent_DIR, "output/snapshots")
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    # ResNet50 structure
    model = hopenet_112.Hopenet(hopenet_112.Bottleneck, [3, 4, 6, 3], 66)
    model.initialize(ctx=ctx_init)
    #model.initialize()

    print ('Loading data.')

    # 数据加载迭代器
    train_loader = datasets_112.Custom_iter(args.data_dir, args.filename_list, batch_size = batch_size)

    # 两种损失函数
    ce_error= mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
    l2_error = mxnet.gluon.loss.L2Loss()

    idx_tensor = [idx for idx in xrange(66)]
    idx_tensor = mxnet.nd.array(idx_tensor, dtype=np.float32, ctx = ctx_init)

    trainer = mxnet.gluon.Trainer(model.collect_params(), 'Adam', {'learning_rate': lr})

    print ('Ready to train network.')
    for epoch in range(num_epochs):

        for i, data_batch in enumerate(train_loader):
            images = data_batch.batchdata_
            labels = data_batch.batchlabel_
            cont_labels = data_batch.batchcontlabel_

            # 两种label
            # Binned labels
            label_yaw = nd.array(labels[:,0])
            label_pitch = nd.array(labels[:,1])
            label_roll = nd.array(labels[:,2])

            # Continuous labels
            label_yaw_cont = nd.array(cont_labels[:,0])
            label_pitch_cont = nd.array(cont_labels[:,1])
            label_roll_cont = nd.array(cont_labels[:,2])

            images = nd.array(images)

            with autograd.record():
                # Forward pass
                # 网络返回三种角度
                yaw, pitch, roll = model(images.as_in_context(ctx_init))

                ce_yaw_loss = ce_error(yaw, label_yaw.as_in_context(ctx_init))
                ce_pitch_loss = ce_error(pitch, label_pitch.as_in_context(ctx_init))
                ce_roll_loss = ce_error(roll, label_roll.as_in_context(ctx_init))
                yaw_predicted = mxnet.nd.softmax(yaw)
                pitch_predicted = mxnet.nd.softmax(pitch)
                roll_predicted = mxnet.nd.softmax(roll)

                yaw_predicted = mxnet.nd.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
                pitch_predicted = mxnet.nd.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
                roll_predicted = mxnet.nd.sum(roll_predicted * idx_tensor, 1) * 3 - 99

                l2_yaw_loss = l2_error(yaw_predicted, label_yaw_cont.as_in_context(ctx_init))
                l2_pitch_loss = l2_error(pitch_predicted, label_pitch_cont.as_in_context(ctx_init))
                l2_rool_loss = l2_error(roll_predicted, label_roll_cont.as_in_context(ctx_init))

                # Total loss
                ce_yaw_loss = ce_yaw_loss + alpha * l2_yaw_loss
                ce_pitch_loss = ce_pitch_loss + alpha * l2_pitch_loss
                ce_roll_loss = ce_roll_loss + alpha * l2_rool_loss
                # 三种损失函数放在一起
                loss_seq = [ce_yaw_loss, ce_pitch_loss, ce_roll_loss]

            mxnet.autograd.backward(loss_seq)

            trainer.step(batch_size)

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                       %(epoch+1, num_epochs, i+1, len(train_loader)//batch_size,
                         ce_yaw_loss.mean().asscalar(), ce_pitch_loss.mean().asscalar(), ce_roll_loss.mean().asscalar()))

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print ('Taking snapshot...')
            filename = os.path.join(snapshot_dir, 'snapshot_' + str(epoch) + '.params')
            #model.save_params(filename)
            model.collect_params().save(filename)








