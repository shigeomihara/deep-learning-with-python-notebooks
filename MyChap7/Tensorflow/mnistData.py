############################ mnistData.py ##########################
from keras.datasets import mnist
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getData():
    (images, labels), (test_images, test_labels) = mnist.load_data()
    print("images.shape=", images.shape)
    print("test_images.shape=", test_images.shape)
    # # print("images[0]=", images[0])
    # for i in range(10):
    #     print(f"labels[{i}]= {labels[i]}")
    #     pil_img = Image.fromarray(images[i])
    #     pil_img.save("output.png")

    #     img = mpimg.imread("output.png") # png形式の画像データをnumpyのndarray形式の配列に変換
    #     plt.imshow(img) # imshowは配列を引数に取ることができる。
    #     plt.show() # 読み込んだ配列形式のデータを画像形式として表示
    # exit()
    
    images = images.reshape((60000, 28 * 28)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
    train_images, val_images = images[10000:], images[:10000]
    train_labels, val_labels = labels[10000:], labels[:10000]

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


