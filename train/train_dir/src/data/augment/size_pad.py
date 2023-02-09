import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.vision import Border


class SizePad:
    def __init__(self):
        pass

    def _img_pad(self, img):
        h, w, c = img.shape
        if h > w:
            pads = (h - w) // 2
            img = c_vision.Pad([pads, 0, pads, 0], padding_mode=Border.EDGE)(img)
        if w > h:
            pads = (w - h) // 2
            img = c_vision.Pad([0, pads, 0, pads], padding_mode=Border.EDGE)(img)
        return img

    def __call__(self, x):
        output = self._img_pad(x)
        return output
