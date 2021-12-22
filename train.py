import os
import numpy as np
from tqdm import tqdm
import scipy.stats
import pandas as pd
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import utils
import torch
from torch import nn
from model import CNN_BLSTM
from torch.utils.data import DataLoader
import json
from getdata import getdataset


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, epoch))
    return model, optimizer, epoch


def save_checkpoint(model, optimizer, learning_rate, epoch, filepath):
    print("Saving model and optimizer state at epoch {} to {}".format(
        epoch, filepath))
    model_for_saving = CNN_BLSTM().cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate
                }, filepath)


def train(rank, output_directory, epochs, learning_rate,
          batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard, earlystopping):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # compute loss
    criterion = None

    # build model
    model = CNN_BLSTM().cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # apex
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    epoch_offset = 0
    if checkpoint_path != "":
        model, optimizer, epoch_offset = load_checkpoint(checkpoint_path, model,
                                                         optimizer)
        epoch_offset += 1  # next iteration is iteration + 1

    # loaddata
    trainset = getdataset(loaddata_config, train_config["seed"], "train")
    validset = getdataset(loaddata_config, train_config["seed"], "valid")

    train_loader = DataLoader(trainset, num_workers=0,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)
    valid_loader = DataLoader(validset, num_workers=0,
                              batch_size=batch_size,
                              pin_memory=False,
                              )

    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, 'logs'))

    model.train()

    # TRAINING LOOP
    stop_step = 0
    min_loss = float("inf")
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(tqdm(train_loader)):
            model.train()
            model.zero_grad()
            model_input, [mos_y, frame_mos_y] = batch
            model_input = torch.autograd.Variable(model_input.cuda())
            mos_y = mos_y.cuda()
            frame_mos_y = frame_mos_y.cuda()

            avg_score, frame_score = model(model_input)
            fn_mse1 = nn.MSELoss()
            fn_mse2 = nn.MSELoss()
            loss = fn_mse1(batch[1][0].cuda(), avg_score) + fn_mse2(batch[1][1].cuda(), frame_score)
            reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            print("epoch:{},loss:\t{:.9f}".format(epoch, reduced_loss))
            if with_tensorboard and rank == 0:
                logger.add_scalar('training_loss_batch', reduced_loss, i + len(train_loader) * epoch)

        # validate
        if rank == 0:
            checkpoint_path = "{}/mosnet_{}".format(
                output_directory, epoch)
            save_checkpoint(model, optimizer, learning_rate, epoch,
                            checkpoint_path)
            if with_tensorboard:
                logger.add_scalar('training_loss_epoch', reduced_loss, epoch)

        # earlystopping
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                model_input, [mos_y, frame_mos_y] = batch
                model_input = torch.autograd.Variable(model_input.cuda())
                mos_y = mos_y.cuda()
                frame_mos_y = frame_mos_y.cuda()

                avg_score, frame_score = model(model_input)
                fn_mse1 = nn.MSELoss()
                fn_mse2 = nn.MSELoss()
                loss = fn_mse1(batch[1][0].cuda(), avg_score) + fn_mse2(batch[1][1].cuda(), frame_score)
                reduced_loss = loss.item()

            print("validloss:\t{:.9f}".format(reduced_loss))
            print("minloss:\t{:.9f}".format(reduced_loss))
            if with_tensorboard and rank == 0:
                logger.add_scalar('valid_loss_epoch', reduced_loss, epoch)

            if min_loss > reduced_loss:
                min_loss = reduced_loss
                min_epoch = epoch
            if (min_loss - reduced_loss) > -0.01:
                stop_step = 0

            else:
                stop_step += 1
                print("minloss:\t{:.9f},min_epoch:{}".format(min_loss, min_epoch))
                if stop_step > earlystopping:
                    print("earlystopping!")
                    return min_epoch
    return min_epoch


def test(train_config, loaddata_config, min_epoch, is_fp16):
    checkpoint_path = "{}/mosnet_{}".format(
        train_config["output_directory"], min_epoch)
    model = CNN_BLSTM().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model, optimizer, epoch_offset = load_checkpoint(checkpoint_path, model,
                                                     optimizer)
    if is_fp16:
        from apex import amp
        model, _ = amp.initialize(model, [], opt_level="O3")

    print('testing...')
    model.eval()
    testset = getdataset(loaddata_config, train_config["seed"], "test")
    test_loader = DataLoader(testset, num_workers=0,
                             batch_size=1,
                             pin_memory=False,
                             )
    MOS_Predict = np.zeros([len(testset), ])
    MOS_true = np.zeros([len(testset), ])
    df = pd.DataFrame(columns=['audio', 'true_mos', 'predict_mos'])

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            model_input, [mos_y, frame_mos_y] = batch
            model_input = torch.autograd.Variable(model_input.cuda())

            avg_score, frame_score = model(model_input)

            MOS_Predict[i] = avg_score.item()
            MOS_true[i] = mos_y.item()
            df = df.append({'true_mos': MOS_true[i],
                            'predict_mos': MOS_Predict[i]},
                           ignore_index=True)

    plt.style.use('seaborn-deep')
    x = df['true_mos']
    y = df['predict_mos']
    bins = np.linspace(1, 5, 40)
    plt.figure(2)
    plt.hist([x, y], bins, label=['true_mos', 'predict_mos'])
    plt.legend(loc='upper right')
    plt.xlabel('MOS')
    plt.ylabel('number')
    plt.show()
    plt.savefig('./output/MOSNet_distribution.png', dpi=150)

    MSE = np.mean((MOS_true - MOS_Predict) ** 2)
    print('[UTTERANCE] Test error= %f' % MSE)
    LCC = np.corrcoef(MOS_true, MOS_Predict)
    print('[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1])
    SRCC = scipy.stats.spearmanr(MOS_true.T, MOS_Predict.T)
    print('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0])

    # Plotting scatter plot
    M = np.max([np.max(MOS_Predict), 5])
    plt.figure(3)
    plt.scatter(MOS_true, MOS_Predict, s=15, color='b', marker='o', edgecolors='b', alpha=.20)
    plt.xlim([0.5, M])
    plt.ylim([0.5, M])
    plt.xlabel('True MOS')
    plt.ylabel('Predicted MOS')
    plt.title('LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}'.format(LCC[0][1], SRCC[0], MSE))
    plt.show()
    plt.savefig('./output/MOSNet_scatter_plot.png', dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')

    args = parser.parse_args()
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    global train_config
    train_config = config["train_config"]
    global loaddata_config
    loaddata_config = config["loaddata_config"]

    num_gpus = torch.cuda.device_count()
    assert num_gpus < 2
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    min_epoch = train(args.rank, **train_config)
    # testing
    test(train_config, loaddata_config, min_epoch, train_config["fp16_run"])
