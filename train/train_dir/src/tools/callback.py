"""callback function"""

import os
import numpy as np
import mindspore as ms
from mindspore.train.callback import Callback
from mindspore.common.tensor import Tensor


class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, args):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.args = args
        self.src_url = args.output_path
        self.best_acc = 0.
        self.best_epoch = 0
        self.best_loss = 10
        if not os.path.exists(os.path.realpath(self.src_url)):
            os.makedirs(os.path.realpath(self.src_url))

    def epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        if self.args.val_split:
            result = self.model.eval(self.eval_dataset.val_dataset)
            if result["acc"] > self.best_acc:
                self.best_acc = result["acc"]
                self.best_epoch = cur_epoch_num
                ms.save_checkpoint(cb_params.train_network, f"{self.src_url}/best_model.ckpt",
                                   append_dict={"acc": self.best_acc})
                print("Best model saved")
            print("epoch: %s acc: %s, best epoch and acc is %s: %s" %
                  (cur_epoch_num, result["acc"], self.best_epoch, self.best_acc), flush=True)
        else:
            loss = cb_params.net_outputs
            if isinstance(loss, (tuple, list)):
                if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                    loss = loss[0]
            if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
                loss = float(np.mean(loss.asnumpy()))
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_epoch = cur_epoch_num
                ms.save_checkpoint(cb_params.train_network, f"{self.src_url}/best_model.ckpt",
                                   append_dict={"loss": self.best_loss})
                print("Best model saved")
            print("epoch: %s loss: %s, best epoch and loss is %s: %s" %
                  (cur_epoch_num, loss, self.best_epoch, self.best_loss), flush=True)
