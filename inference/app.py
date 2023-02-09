import gradio as gr
import numpy as np
import json
import argparse
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.dataset.vision.utils import Inter
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.nn as nn
from mindspore import dtype as mstype
from models.Nets import MSNet
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def cast_amp(net):
    """cast network amp_level"""
    net.to_float(mstype.float16)
    cell_types = (nn.LayerNorm, nn.Softmax, nn.BatchNorm2d, nn.GELU, nn.SyncBatchNorm, nn.GroupNorm)
    do_keep_fp32(net, cell_types)


def do_keep_fp32(network, cell_types):
    """Cast cell to fp32 if cell in cell_types"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, cell_types):
            cell.to_float(mstype.float32)


def predict_image(img):
    img = np.array(img).astype(np.float32)
    img = c_vision.Resize((256, 256), interpolation=Inter.BICUBIC)(img)
    img = c_vision.CenterCrop(224)(img)
    mean = np.array([0.5 * 255, 0.5 * 255, 0.5 * 255])
    std = np.array([0.5 * 255, 0.5 * 255, 0.5 * 255])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 224, 224)

    predict_score = model.predict(Tensor(img)).reshape(-1)
    predict_score = nn.Softmax()(predict_score).asnumpy()

    return {class_names[str(i+1)]: float(predict_score[i]) for i in range(args.num_classes)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_classes = 256
    args.drop_rate = 0.0
    args.drop_path_rate = 0.5

    # class_names = [str(i) for i in range(1, args.num_classes + 1)]
    with open('./label_dic_modify.json', 'r') as f:
        class_names = json.load(f)

    param_dict = load_checkpoint('./best_model.ckpt')
    network = MSNet(args)
    cast_amp(network)
    load_param_into_net(network, param_dict)

    model = Model(network)
    image = gr.inputs.Image()
    label = gr.outputs.Label(num_top_classes=args.num_classes, label="预测类别")

    gr.Interface(css=".footer {display:none !important}",
                 fn=predict_image,
                 inputs=image,
                 live=False,
                 description="Please upload a image in JPG, JPEG or PNG.",
                 title='Image Classification by RepLKNet31XL',
                 outputs=label,
                 examples=['example_img/airplane.jpg', 'example_img/glass.jpg', 'example_img/ostrich.jpg',
                           'example_img/piano.jpg', 'example_img/sheep.jpg']
                 ).launch()
