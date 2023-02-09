import os
import numpy as np
from mindspore import Model
from mindspore import context
from mindspore.common import set_seed

from src.args import args
from src.tools.cell import cast_amp
from src.tools.get_misc import get_dataset, set_device, get_model, pretrained

set_seed(args.seed)


def main():
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    set_device(args)

    # get model
    net = get_model(args)
    cast_amp(net)

    if args.model_path:
        pretrained(args, net)

    data = get_dataset(args, training=False)

    model = Model(net)
    print("begin predict")
    ckpt_save_dir = os.path.join(args.output_path, f'result.txt')
    result = ''
    for x in data.test_dataset.create_dict_iterator():
        outcome = model.predict(x["image"]).asnumpy()
        result += ''.join([str(i+1)+'\n' for i in np.argmax(outcome, axis=1)])
    with open(ckpt_save_dir, 'w') as f:
        f.write(result)
    print("File saved")


if __name__ == '__main__':
    main()
