import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)


    # Create loader instance
loader = MnistDataloader(
    "D:\\data\\mnist\\train-images.idx3-ubyte",
    "D:\\data\\mnist\\train-labels.idx1-ubyte", 
    "D:\\data\\mnist\\t10k-images.idx3-ubyte",
    "D:\\data\\mnist\\t10k-labels.idx1-ubyte"
)

# Load data
(x_train, y_train), (x_test, y_test) = loader.load_data()

# x_train: list of 60,000 numpy arrays (28×28 images)
# y_train: list of 60,000 labels (0-9)
     
batch_size = 1000
data = np.zeros((batch_size , 28*28) )
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
data = x_train[:batch_size].reshape(batch_size, 28*28)  # Flatten images
labels = y_train[:batch_size]


test_n = np.random.randint(1,1000)
test=x_test[ test_n , : ].reshape(28*28)/255

print(test)
print(y_test[test_n])

def visualize_1d_pixels(pixel_array, image_shape=(28, 28)):
    """
    Visualize a 1D array of pixel values
    
    Args:
        pixel_array: 1D array of pixel values (0 to 1)
        image_shape: expected shape of the image (height, width)
    """
    # Reshape to 2D
    if len(pixel_array) == image_shape[0] * image_shape[1]:
        image_2d = pixel_array.reshape(image_shape)
    else:
        # If not perfect square, find closest square
        side_length = int(np.sqrt(len(pixel_array)))
        image_2d = pixel_array[:side_length*side_length].reshape(side_length, side_length)
    
    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image_2d, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label='Pixel intensity')
    plt.title('Pixel Visualization')
    plt.axis('off')
    plt.show()
visualize_1d_pixels(test)
print("done")