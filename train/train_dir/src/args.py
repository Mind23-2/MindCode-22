"""global args for CoAtNet"""
import argparse
import ast
import os
import sys
import yaml
from .configs import parser as _parser


def parse_arguments():
    """parse_arguments"""
    parser = argparse.ArgumentParser(description="MindSpore CoAtNet Training")

    parser.add_argument("-a", "--arch", metavar="ARCH", default="msnet",
                        help="model architecture")
    parser.add_argument("--amp_level", default="O2", choices=["O0", "O1", "O2", "O3"], help="AMP Level")
    parser.add_argument("--batch_size", default=128, type=int, metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--beta", default=[0.9, 0.999], type=lambda x: [float(a) for a in x.split(",")],
                        help="beta for optimizer")
    parser.add_argument("--clip_global_norm_value", default=5., type=float, help="Clip grad value")
    parser.add_argument("--crop", default=True, type=ast.literal_eval, help="Crop when testing")

    parser.add_argument('--pretrain_path', type=str, default='', help='the pretrain model')
    parser.add_argument('--train_data_path', type=str, default='', help='the training data')
    parser.add_argument('--model_path', type=str, default='', help='the model data')
    parser.add_argument('--test_data_path', type=str, default='', help='the testing data')
    parser.add_argument('--output_path', default='', type=str, help='the path model saved')

    parser.add_argument("--device_id", default=0, type=int, help="Device Id")
    parser.add_argument("--device_num", default=1, type=int, help="device num")
    parser.add_argument("--device_target", default="Ascend", choices=["GPU", "Ascend", "CPU"], type=str)
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--eps", default=1e-5, type=float)
    parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR"], default="MINDIR", help="file format")
    parser.add_argument("--in_channel", default=3, type=int)
    parser.add_argument("--is_dynamic_loss_scale", default=1, type=int, help="is_dynamic_loss_scale ")
    parser.add_argument("--keep_checkpoint_max", default=2, type=int, help="keep checkpoint max num")
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument("--graph_mode", default=0, type=int, help="graph mode with 0, python with 1")
    parser.add_argument("--mix_up", default=0., type=float, help="mix up")
    parser.add_argument("--mlp_ratio", help="mlp ", default=4., type=float)
    parser.add_argument("-j", "--num_parallel_workers", default=20, type=int, metavar="N",
                        help="number of data loading workers (default: 20)")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--warmup_length", default=0, type=int, help="Number of warmup iterations")
    parser.add_argument("--warmup_lr", default=5e-7, type=float, help="warm up learning rate")
    parser.add_argument("--wd", "--weight_decay", default=0.05, type=float, metavar="W",
                        help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--loss_scale", default=1024, type=int, help="loss_scale")
    parser.add_argument("--lr", "--learning_rate", default=5e-4, type=float, help="initial lr", dest="lr")
    parser.add_argument("--lr_scheduler", default="cosine_lr", help="Schedule for the learning rate.")
    parser.add_argument("--lr_adjust", default=30, type=float, help="Interval to drop lr")
    parser.add_argument("--lr_gamma", default=0.95, type=int, help="Multistep multiplier")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--num_classes", default=256, type=int)
    parser.add_argument("--drop_rate", help="dropout", default=0., type=float)
    parser.add_argument("--swin_config", help="Config file to use (see configs dir)",
                        default='/home/ma-user/modelarts/user-job-dir/train_dir/src/configs/msnet.yaml')
    parser.add_argument("--seed", default=0, type=int, help="seed for initializing training. ")
    parser.add_argument("--save_every", default=1, type=int, help="Save every ___ epochs(default:2)")
    parser.add_argument("--label_smoothing", type=float, help="Label smoothing to use, default 0.0", default=0.1)
    parser.add_argument("--image_size", default=224, help="Image Size.", type=int)
    parameter = parser.parse_args()

    # Allow for use from notebook without config file
    parameter = get_config(parameter)
    return parameter


def get_config(paras):
    """get_config"""
    override_args = _parser.argv_to_vars(sys.argv)
    # load yaml file
    yaml_txt = open(paras.swin_config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    print(f"=> Reading YAML config from {paras.swin_config}")
    for v in override_args:
        loaded_yaml[v] = getattr(paras, v)

    paras.__dict__.update(loaded_yaml)
    print(paras)

    if "DEVICE_NUM" not in os.environ.keys():
        os.environ["DEVICE_NUM"] = str(paras.device_num)
        os.environ["RANK_SIZE"] = str(paras.device_num)
    return paras


args = parse_arguments()
