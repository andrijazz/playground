import argparse
from train import train
from test import test


def main():
    run_str = train(args)
    test(run_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fully Convolutional Networks TensorFlow implementation [Training]')

    parser.add_argument('-m', '--model_name', type=str, help='model name. fcn32 or fcn16 or fcn8', default='fcn32')
    parser.add_argument('-d', '--dataset', type=str, help='dataset to train on. kitti or cityscapes', default='kitti')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=2)
    parser.add_argument('-n', '--num_epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-3)
    parser.add_argument('-g', '--gpu', type=int, help='GPU to use for training', default=1)
    parser.add_argument('-k', '--keep_prob', type=float, help='dropout keep prob. default is 0.5.', default=float(0.5))
    parser.add_argument('-i', '--init_weights', type=bool, help='use pretrained vgg-16 weights', default=True)
    args = parser.parse_args()

    main()
