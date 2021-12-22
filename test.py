from train import test
import torch
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('--epoch', type=int,
                        help='the model you want to test')
    parser.add_argument("--is_fp16", action="store_true")

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
    # testing
    min_epoch = args.epoch
    test(train_config, loaddata_config, min_epoch, args.is_fp16)