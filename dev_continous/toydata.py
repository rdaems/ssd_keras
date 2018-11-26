import numpy as np
import cv2

from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Reshape, Activation, Concatenate
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import Callback

from ssd_encoder_decoder.ssd_input_encoder_continuous import SSDInputEncoderContinuous
from ssd_encoder_decoder.ssd_output_decoder_continuous import decode_detections
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_loss_function.keras_ssd_loss_continuous import SSDLossContinuous


def visualize(x):
    img = x + 0.5
    img = np.clip(img, 0, 1)
    cv2.imshow('visualize', img)
    cv2.waitKey(0)


class ToyDataGenerator:
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def random_data(self, seed=None):
        npr = np.random.RandomState(seed)

        x = np.zeros((self.img_height, self.img_width, 3))
        y = []
        for _ in range(npr.randint(1, 4)):
            cx = npr.uniform(0.10, 0.90)
            cy = npr.uniform(0.10, 0.90)
            w = npr.uniform(0.10, 0.20)
            h = npr.uniform(0.10, 0.20)
            r = npr.uniform(- 0.5, 0.5)
            g = npr.uniform(- 0.5, 0.5)
            b = npr.uniform(- 0.5, 0.5)
            xmin = int((cx - w / 2) * self.img_width)
            ymin = int((cy - h / 2) * self.img_height)
            xmax = int((cx + w / 2) * self.img_width)
            ymax = int((cy + h / 2) * self.img_height)
            xmin, ymin, xmax, ymax = self.make_valid_box(xmin, ymin, xmax, ymax)
            x[ymin:ymax, xmin:xmax, :] = (r, g, b)
            y.append(np.array([1, r, g, b, xmin, ymin, xmax, ymax]))
        return x, np.array(y)

    def make_valid_box(self, xmin, ymin, xmax, ymax):
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > self.img_width - 1:
            xmax = self.img_width - 1
        if ymax > self.img_height - 1:
            ymax = self.img_height - 1
        return xmin, ymin, xmax, ymax

    def generate(self, batch_size=32, label_encoder=None):
        while True:
            batch_X, batch_y = [], []
            for _ in range(batch_size):
                x, y = self.random_data()
                batch_X.append(x)
                batch_y.append(y)
            batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
            yield np.array(batch_X), batch_y_encoded


def show(x, y, show=False, to_file=None):
    out_scale = 32
    img = (x + 0.5) * 255
    # img *= 0.8 # dim
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=out_scale, fy=out_scale, interpolation=cv2.INTER_NEAREST)
    for _, _, r, g, b, xmin, ymin, xmax, ymax in y:
        r = np.clip((r + 0.5) * 255, 0, 255)
        g = np.clip((g + 0.5) * 255, 0, 255)
        b = np.clip((b + 0.5) * 255, 0, 255)
        xmin = int(xmin * out_scale)
        ymin = int(ymin * out_scale)
        xmax = int(xmax * out_scale)
        ymax = int(ymax * out_scale)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (r, g, b), 2)

    if to_file:
        cv2.imwrite(to_file, img)
    if show:
        cv2.imshow('show', img)
        cv2.waitKey(0)
    return img


def build_model(image_size, scales, aspect_ratios, two_boxes_for_ar1, n_classes, n_clabels, return_predictor_sizes=False):
    n_classes += 1 # background class
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    n_boxes = len(aspect_ratios)

    x = Input(shape=(img_height, img_width, img_channels))

    conv1 = Conv2D(32, 3, padding='same', activation='elu')(x)
    pool1 = MaxPool2D()(conv1)
    norm1 = BatchNormalization()(pool1)

    conv2 = Conv2D(32, 3, padding='same', activation='elu')(norm1)
    pool2 = MaxPool2D()(conv2)
    norm2 = BatchNormalization()(pool2)

    classes = Conv2D(n_boxes * n_classes, 3, padding='same', activation='elu', name='classes')(norm2)

    clabels = Conv2D(n_boxes * n_clabels, 3, padding='same', activation='linear', name='clabels')(norm2)

    boxes = Conv2D(n_boxes * 4, 3, padding='same', activation='linear', name='boxes')(norm2)

    anchors = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios,
                          two_boxes_for_ar1=two_boxes_for_ar1, name='anchors')(boxes)

    classes_reshaped = Reshape((-1, n_classes), name='classes_reshape')(classes)

    clabels_reshaped = Reshape((-1, n_clabels), name='clabels_reshape')(clabels)

    boxes_reshaped = Reshape((-1, 4), name='boxes_reshape')(boxes)

    anchors_reshaped = Reshape((-1, 8), name='anchors_reshape')(anchors)

    classes_softmax = Activation('softmax', name='classes_softmax')(classes_reshaped)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + n_clabels + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, clabels_reshaped, boxes_reshaped, anchors_reshaped])

    model = Model(inputs=x, outputs=predictions)

    if return_predictor_sizes:
        # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
        predictor_sizes = np.array([classes._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model


class ValCallback(Callback):
    def __init__(self, data_generator):
        self.data_generator = data_generator
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.data_generator.random_data(seed=None)
        y_pred = self.model.predict(x[np.newaxis, :])

        # for a in y_pred[0, :, 10]:
        #     print(a)

        # y_pred[:, :, 5:9] = 0
        # y_pred[:, :3, [0, 1]] = [0, 1]
        # y_pred[:, 3:, [0, 1]] = [1, 0]

        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.5,
                                           iou_threshold=0.45,
                                           top_k=10,
                                           img_height=self.data_generator.img_height,
                                           img_width=self.data_generator.img_width)
        show(x, y_pred_decoded[0], show=False, to_file='val_output/{:02d}.bmp'.format(epoch))


if __name__ == '__main__':
    img_width = 16
    img_height = 16
    dg = ToyDataGenerator(img_width, img_height)

    # x, y = dg.random_data()
    # print(y)
    # visualize(x)

    batch_size = 256
    n_steps = 20
    n_epochs = 1000

    scales = [0.10, 0.10] # second value is not used
    aspect_ratios = [0.5, 1, 2]
    two_boxes_for_ar1 = False

    n_classes = 1
    n_clabels = 3

    model, predictor_sizes = build_model((img_height, img_width, 3), scales, aspect_ratios,
                                         two_boxes_for_ar1, n_classes, n_clabels, return_predictor_sizes=True)
    # model.summary()

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    ssd_loss = SSDLossContinuous(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    ssd_input_encoder = SSDInputEncoderContinuous(img_height, img_width, n_classes, n_clabels, predictor_sizes,
                                                  scales=scales, aspect_ratios_global=aspect_ratios,
                                                  two_boxes_for_ar1=two_boxes_for_ar1, pos_iou_threshold=0.5, neg_iou_limit=0.3)

    # x, y = next(dg.generate(batch_size, ssd_input_encoder))
    # print(x.shape)
    # print(y.shape)

    val_callback = ValCallback(dg)

    model.fit_generator(dg.generate(batch_size, ssd_input_encoder), steps_per_epoch=n_steps, epochs=n_epochs, callbacks=[val_callback])
