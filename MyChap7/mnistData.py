############################ mnistData.py ##########################
from keras.datasets import mnist

def getData():
    (images, labels), (test_images, test_labels) = mnist.load_data()
    images = images.reshape((60000, 28 * 28)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
    train_images, val_images = images[10000:], images[:10000]
    train_labels, val_labels = labels[10000:], labels[:10000]

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


