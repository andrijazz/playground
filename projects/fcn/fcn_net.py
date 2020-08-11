import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Conv2DTranspose, Cropping2D, Add


class FCNNet(Model):
    def __init__(self, config):
        super(FCNNet, self).__init__()
        self.config = config
        self.num_classes = self.config.DATASET_NUM_CLASSES
        # pad image 100
        # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/1305c7378a9f0ab44b2c936f4d60e4687e3d8743/voc-fcn32s/net.py#L28
        # vgg encoder
        self.conv1_1 = Conv2D(64, 3, activation='relu', padding='same')
        self.conv1_2 = Conv2D(64, 3, activation='relu', padding='same')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv2_1 = Conv2D(128, 3, activation='relu', padding='same')
        self.conv2_2 = Conv2D(128, 3, activation='relu', padding='same')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv3_1 = Conv2D(256, 3, activation='relu', padding='same')
        self.conv3_2 = Conv2D(256, 3, activation='relu', padding='same')
        self.conv3_3 = Conv2D(256, 3, activation='relu', padding='same')
        self.pool3 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv4_1 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv4_2 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv4_3 = Conv2D(512, 3, activation='relu', padding='same')
        self.pool4 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv5_1 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv5_2 = Conv2D(512, 3, activation='relu', padding='same')
        self.conv5_3 = Conv2D(512, 3, activation='relu', padding='same')
        self.pool5 = MaxPool2D(pool_size=(2, 2), strides=2)
        # decoder
        self.fc6 = Conv2D(4096, 7, activation='relu', padding='same')
        self.drop6 = Dropout(0.5)
        self.fc7 = Conv2D(4096, 1, activation='relu', padding='same')
        self.drop7 = Dropout(0.5)
        self.score_fr = Conv2D(self.num_classes, 1)
        self.upscore2 = Conv2DTranspose(self.num_classes, kernel_size=4, strides=2, padding='same')
        self.score_pool4 = Conv2D(self.num_classes, 1)
        self.fuse_pool4 = Add()
        self.upscore_pool4 = Conv2DTranspose(self.num_classes, kernel_size=4, strides=2, padding='same')
        self.score_pool3 = Conv2D(self.num_classes, 1)
        self.fuse_pool3 = Add()
        self.upscore8 = Conv2DTranspose(self.num_classes, kernel_size=16, strides=8, padding='same')

    def call(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        pool3 = x
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        pool4 = x
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        x = self.fc6(x)
        x = self.drop6(x)
        x = self.fc7(x)
        x = self.drop7(x)
        x = self.score_fr(x)
        x = self.upscore2(x)
        score_pool4 = self.score_pool4(pool4)
        # crop score_pool4 to match upscore2 result
        score_pool4c = self.crop_to_match(score_pool4, x)
        x = self.fuse_pool4([x, score_pool4c])
        x = self.upscore_pool4(x)
        score_pool3 = self.score_pool3(pool3)
        # crop score_pool3 to match upscore_pool4 result
        score_pool3c = self.crop_to_match(score_pool3, x)
        x = self.fuse_pool3([x, score_pool3c])
        x = self.upscore8(x)
        return x

    def crop_to_match(self, x, y):
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D
        crop_h = (x.shape[1] - y.shape[1]) // 2
        crop_w = (x.shape[2] - y.shape[2]) // 2
        if crop_h == 0 and crop_w == 0:
            return x
        cropped_x = Cropping2D(((crop_h, crop_h), (crop_w, crop_w)))(x)
        return cropped_x

    def get_prediction(self, x):
        output = self.call(x)
        idx_output = tf.argmax(output, axis=-1)
