import os
from mindspore import nn
from mindspore import Model
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.callback import LossMonitor, TimeMonitor

from src.args import args
from src.tools.cell import cast_amp
from src.tools.callback import EvaluateCallBack
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, get_train_one_step
from src.tools.optimizer import get_optimizer


def main():
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)
    set_seed(args.seed + rank)

    # get model and cast amp_level
    net = get_model(args)
    cast_amp(net)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)

    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters
    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    eval_network = nn.WithEvalCell(net, criterion, args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={"acc", "loss"},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())
    args.output_path = os.path.join(args.output_path, f'ckpt_{rank}')
    loss_cb = LossMonitor()

    eval_cb = EvaluateCallBack(model, eval_dataset=data, args=args)

    print("begin train")
    model.train(int(args.epochs - args.start_epoch), data.train_dataset, callbacks=[time_cb, loss_cb, eval_cb],
                dataset_sink_mode=True)
    print("train success")


if __name__ == '__main__':
    main()
