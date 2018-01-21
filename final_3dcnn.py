import argparse
import os

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
#from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import EarlyStopping 
#from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import videoto3d
from tqdm import tqdm


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

def plot_confusion_matrix(cm):
    plt.matshow(cm, cmap=plt.get_cmap('GnBu'))
    plt.title('Confusion Matrix')
    plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    #plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join('finalmy3dcnnresult/', 'confusion_matrix.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def loadTraindata(video_dir, vid3d, nclass, result_dir, color=False, skip=True):
    #files = os.listdir(video_dir)

    ###########################
    train_file = '3dCNN/ucfTrainTestlist/trainlist01.txt'
    with open(train_file) as f:
        train_list = f.readlines()
    train_list = [(x.strip()).split('/')[1].split(' ')[0] for x in train_list]
    ###########################

    X_train = []
    train_labels = []
    train_labellist = []

    pbar = tqdm(total=len(train_list))

    for filename in train_list:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(video_dir, filename)
        label = vid3d.get_UCF_classname(filename)
        if label not in train_labellist:
            if len(train_labellist) >= nclass:
                continue
            train_labellist.append(label)
        train_labels.append(label)
        X_train.append(vid3d.video3d(name, color=color, skip=skip))

    pbar.close()
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(train_labellist)):
            fp.write('{}\n'.format(train_labellist[i]))

    for num, label in enumerate(train_labellist):
        for i in range(len(train_labels)):
            if label == train_labels[i]:
                train_labels[i] = num
    if color:
        return np.array(X_train).transpose((0, 2, 3, 4, 1)), train_labels
    else:
        return np.array(X_train).transpose((0, 2, 3, 1)), train_labels


def loadTestdata(video_dir, vid3d, nclass, result_dir, color=False, skip=True):
    ###########################
    test_file = '3dCNN/ucfTrainTestlist/testlist01.txt'
    with open(test_file) as f:
        test_list = f.readlines()
    test_list = [(x.strip()).split('/')[1].split(' ')[0] for x in test_list]
    ###########################
    X_test = []
    test_labels = []
    test_labellist = []

    pbar = tqdm(total=len(test_list))

    for filename in test_list:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(video_dir, filename)
        label = vid3d.get_UCF_classname(filename)
        if label not in test_labellist:
            if len(test_labellist) >= nclass:
                continue
            test_labellist.append(label)
        test_labels.append(label)
        X_test.append(vid3d.video3d(name, color=color, skip=skip))

    pbar.close()

    for num, label in enumerate(test_labellist):
        for i in range(len(test_labels)):
            if label == test_labels[i]:
                test_labels[i] = num
    if color:
        return np.array(X_test).transpose((0, 2, 3, 4, 1)), test_labels
    else:
        return np.array(X_test).transpose((0, 2, 3, 1)), test_labels


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--videos', type=str, default='3dCNN/UCF101/',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=101)
    #parser.add_argument('--nsplit', type=int, default=1)    # Add
    parser.add_argument('--output', type=str, default='finalmy3dcnnresult/')
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--skip', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=15)
    args = parser.parse_args()

    #img_rows, img_cols, frames = 32, 32, args.depth - Defaultvalue
    # The origin size of UCF101 is 320x240
    img_rows, img_cols, frames = 24, 32, args.depth
    channel = 3 if args.color else 1


    # Train
    fname_train_npz = 'train_dataset_{}_{}_{}.npz'.format(
        args.nclass, args.depth, args.skip)

    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
    nb_classes = args.nclass
    if os.path.exists(fname_train_npz):
        train_loadeddata = np.load(fname_train_npz)
        trainX, trainY = train_loadeddata["X"], train_loadeddata["Y"]
    else:
        x, y = loadTraindata(args.videos, vid3d, args.nclass,
                        args.output, args.color, args.skip)    # Add args.nsplit as argument
        trainX = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        trainY = np_utils.to_categorical(y, nb_classes)

        trainX = trainX.astype('float32')
        np.savez(fname_train_npz, X=trainX, Y=trainY)
        print('Saved train dataset to train_dataset.npz.')
    print('trainX_shape:{}\ntrainY_shape:{}'.format(trainX.shape, trainY.shape))


    # Test
    fname_test_npz = 'test_dataset_{}_{}_{}.npz'.format(
        args.nclass, args.depth, args.skip)

    #vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
    #nb_classes = args.nclass
    if os.path.exists(fname_test_npz):
        test_loadeddata = np.load(fname_test_npz)
        testX, testY = test_loadeddata["X"], test_loadeddata["Y"]
    else:
        x, y = loadTestdata(args.videos, vid3d, args.nclass,
                        args.output, args.color, args.skip)    # Add args.nsplit as argument
        testX = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        testY = np_utils.to_categorical(y, nb_classes)

        testX = testX.astype('float32')
        np.savez(fname_test_npz, X=testX, Y=testY)
        print('Saved test dataset to test_dataset.npz.')
    print('testX_shape:{}\ntestY_shape:{}'.format(testX.shape, testY.shape))

    # Define model
    model = Sequential()
    # Default initialization in Keras is glorot_uniform (same as Xavier)
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
        trainX.shape[1:]), border_mode='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
    #model.add(Activation('softmax'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    #model.add(Activation('softmax'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5,
                          verbose=1, mode='auto')
    callbacks_list = [earlystop]

    #plot_model(model, show_shapes=True,
    #           to_file=os.path.join(args.output, 'model.png'))

    #Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(trainX, trainY, test_size=0.1, random_state=43)

    # Just for testing
    #Xtrain, Xvalid, Ytrain, Yvalid = Xtrain[:2560], Xvalid[:256], Ytrain[:2560], Yvalid[:256]

    #history = model.fit(Xtrain, Ytrain, validation_data=(Xvalid, Yvalid), batch_size=args.batch,
    #                    epochs=args.epoch, callbacks=callbacks_list, verbose=1, shuffle=True)
    history = model.fit(trainX, trainY, batch_size=args.batch, epochs=args.epoch, 
        callbacks=callbacks_list, verbose=1, shuffle=True)
    #model.evaluate(testX, testY, verbose=0)
    model_json = model.to_json()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'ucf101_3dcnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output, 'ucf101_3dcnnmodel.hd5'))

    loss, acc = model.evaluate(testX, testY, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    plot_history(history, args.output)
    save_history(history, args.output)

    predY = model.predict(testX)
    # To get the confusion matrix
    testYle = np.argmax(testY, axis=1)
    predYle = np.argmax(predY, axis=1)
    print 'testY: {0}'.format(np.argmax(testY, axis=1))
    print 'predY: {0}'.format(np.argmax(predY, axis=1))

    cm = confusion_matrix(testYle, predYle)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm=cm)


if __name__ == '__main__':
    main()
