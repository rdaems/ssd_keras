import numpy as np
import cv2

from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Reshape, Activation, Concatenate
from keras import Model
from keras.optimizers import Adam

from ssd_encoder_decoder.ssd_input_encoder_continuous import SSDInputEncoderContinuous
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

    def random_data(self):
        x = np.zeros((self.img_height, self.img_width, 3))
        y = []
        for _ in range(np.random.randint(1, 4)):
            cx = np.random.rand()
            cy = np.random.rand()
            w = np.random.uniform(0.05, 0.15)
            h = np.random.uniform(0.05, 0.15)
            r = np.random.uniform(- 0.5, 0.5)
            g = np.random.uniform(- 0.5, 0.5)
            b = np.random.uniform(- 0.5, 0.5)
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

    def generate(self,
                 batch_size=32,
                 label_encoder=None):

        while True:
            batch_X, batch_y = [], []

            for _ in range(batch_size):
                x, y = self.random_data()
                batch_X.append(x)
                batch_y.append(y)

            batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)

            yield batch_X, batch_y_encoded


def build_model(image_size, mode, scales, aspect_ratios, two_boxes_for_ar1, n_classes, n_labels, return_predictor_sizes=False):
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    n_boxes = len(aspect_ratios)

    coords = 'centroids'
    confidence_thresh = 0.01
    iou_threshold = 0.45
    top_k = 50
    nms_max_output_size = 100

    x = Input(shape=(img_height, img_width, img_channels))

    conv1 = Conv2D(16, 3, padding='same', activation='elu')(x)
    pool1 = MaxPool2D()(conv1)
    norm1 = BatchNormalization()(pool1)

    conv2 = Conv2D(16, 3, padding='same', activation='elu')(norm1)
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

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
        predictor_sizes = np.array([classes._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model


if __name__ == '__main__':
    img_width = 64
    img_height = 64
    dg = ToyDataGenerator(img_width, img_height)

    # x, y = dg.random_data()
    # print(y)
    # visualize(x)

    scales = [0.10, 0.10] # second value is not used
    aspect_ratios = [0.5, 1, 2]
    two_boxes_for_ar1 = False

    n_classes = 2
    n_clabels = 3

    model, predictor_sizes = build_model((img_height, img_width, 3), 'training', scales, aspect_ratios,
                                         two_boxes_for_ar1, n_classes, n_clabels, return_predictor_sizes=True)
    # model.summary()

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLossContinuous(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    ssd_input_encoder = SSDInputEncoderContinuous(img_height, img_width, n_classes, n_clabels, predictor_sizes,
                                                  scales=scales, aspect_ratios_global=aspect_ratios,
                                                  pos_iou_threshold=0.5, neg_iou_limit=0.3)
