import numpy as np
from PIL import Image
from pathlib import Path
from array import array
import matplotlib.pyplot as plt
from os.path  import join
import struct



i = 0
exit = "c"
clac_percent = 0
savemaybe = "b"
batch_size = 60000
training_batch = 100
learning_handler = 0

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
    plt.imshow(image_2d, cmap='gray', vmin=-1, vmax=1)
    plt.colorbar(label='Pixel intensity')
    plt.title('Pixel Visualization')
    plt.axis('off')
    plt.show()





class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        label = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            label = array("B", file.read())        
        
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
        
        return images, label
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

loader = MnistDataloader(
    "D:\\data\\mnist\\train-images.idx3-ubyte",
    "D:\\data\\mnist\\train-labels.idx1-ubyte", 
    "D:\\data\\mnist\\t10k-images.idx3-ubyte",
    "D:\\data\\mnist\\t10k-labels.idx1-ubyte"
)







def save_model():
    name = "latest_3.2_turbo"

    script_path = Path(__file__).parent 
    new_folder = script_path / "models" / name
    new_folder.mkdir(exist_ok=True) 

    np.save(new_folder / "weights1.npy" , weights1 )
    np.save(new_folder / "weights2.npy" , weights2 )
    np.save(new_folder / "weights3.npy" , weights3 )
    np.save(new_folder / "b1.npy" , b1 )
    np.save(new_folder / "b2.npy" , b2 )
    np.save(new_folder / "b3.npy" , b3 )

def load_model():

    name = input("enter model path: ")
    return np.load( name + "\\" +"weights1.npy") , np.load( name + "\\" +"weights2.npy" ) , np.load( name + "\\" +"weights3.npy" ) , np.load( name + "\\" +"b1.npy" ) ,np.load( name + "\\" +"b2.npy" ) , np.load( name + "\\" +"b3.npy" ) 
    


def sigmoid(z):
    #z = np.clip(z, -50, 50)  # Prevents overflow
    #1 / (1 + np.exp(-z))
    return np.maximum(0,z) 

def evaluate( input , layer1 , layer2 , weights1 , weights2 , weights3 , b1 , b2 , b3  ):
    z1=np.dot( weights1 , input  ) + b1
    layer1= sigmoid(z1)

    z2=np.dot( weights2 , layer1 ) + b2
    layer2= sigmoid(z2)

    z3=np.dot( weights3 , layer2 ) + b3
     

    return np.softmax(z3)

def evaluate_t( input , layer1 , layer2 , weights1 , weights2 , weights3 , b1 , b2 , b3  ):

    z1=np.dot( weights1 , input  ) + b1[: , np.newaxis]

    layer1 = sigmoid(z1)

    z2=np.dot( weights2 , layer1 ) + b2[: , np.newaxis]

    layer2 = sigmoid(z2)

    z3=np.dot( weights3 , layer2 ) + b3[: , np.newaxis]

    return layer1 , layer2 , np.softmax(z3)



def get_example_p(img_path):
    img = Image.open(img_path)
    img_arr = np.zeros((28,28))
    for x in range(28):
        for y in range(28):
            img_arr [x , y ] = max(255-(np.array(img)[x , y , : ] ))
    return img_arr.flatten()

def get_example(img_path):
    img = Image.open(img_path)
    img_arr = np.array(img)[:,:,3].flatten() / 255.0
    return img_arr

def get_examples(dataset):
    
    data = np.zeros((28*28,batch_size))
    data_label = np.zeros(batch_size)
    for i in range(batch_size):
        number = np.random.randint(0,9)
        data[:,i] = get_example( dataset / (str(number) +"\\"+ str(number) + "\\" + str(np.random.randint(0,10000))+ ".png"))
        data_label [i] = number  
        print( "data loaded percentage: " +str(i/(batch_size/100))) 
    return data , data_label    

def delta_b(pre_deltas , cur_layer , weights ):

    prod=np.dot( pre_deltas.T , weights )

    return prod.T * cur_layer * ( 1 - cur_layer )
        

def delta(out_layer, ideal):
    return (out_layer - ideal)

def derive(deltas , pre_activations , act_num ) :
    L=len(deltas)
    weight_clac = np.zeros((L , act_num , training_batch ))
    weights_dersss = np.zeros( (L , act_num) )
    for k in range(act_num):
        for j in range(L):
            weight_clac[j , k , :] = pre_activations[k , :]*deltas[j , :]
            weights_dersss[ j , k ] = np.sum(weight_clac [ j , k , : ])/training_batch
    return weights_dersss





learning_rate = 0.01



data = np.zeros((28*28 , batch_size))

input_num= 28*28

layer1_num= 128
layer1=np.zeros((layer1_num , training_batch))

layer2_num= 64
layer2=np.zeros((layer2_num ,training_batch))

num=10
output=np.zeros((num ,training_batch))

weights1=np.random.randn(layer1_num , input_num)
weights2=np.random.randn(layer2_num , layer1_num)
weights3=np.random.randn(num , layer2_num)

b1=np.random.randn(layer1_num)/100
b2=np.random.randn(layer2_num)/100
b3=np.random.randn(num)/100

print("wanna load a model?")

savemaybe = input("enter 'y' if yes: ")
if savemaybe == 'y' :
    weights1 , weights2 , weights3 , b1 , b2 , b3 = load_model()



predection= np.zeros((num,training_batch))
pictures = np.zeros((28*28 ,training_batch))
batch = 0

(x_train, y_train), (x_test, y_test) = loader.load_data()

x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = np.array(x_train)
y_train = np.array(y_train)
for v in range(60000):
    mean = np.mean(x_train[ v , : ])
    std = np.std(x_train[ v , : ])
    x_train[ v , : ] = -(mean - x_train[ v , : ] )/ std

if savemaybe != "y":
    
    
    
    
    


    data = x_train[:batch_size].reshape(batch_size, 28*28)
    labels = y_train[:batch_size]
    
    weights1_ders = np.zeros((layer1_num, input_num))
    weights2_ders = np.zeros((layer2_num, layer1_num))
    weights3_ders = np.zeros((num, layer2_num))
    b1_ders = np.zeros(layer1_num)
    b2_ders = np.zeros(layer2_num)
    b3_ders = np.zeros(num)
    
    

    out_deltas = np.zeros((num,training_batch))
    layer2_deltas = np.zeros((layer2_num,training_batch))
    layer1_deltas = np.zeros((layer1_num,training_batch))
    error = np.zeros((num,training_batch))
    training_batch_labels = np.zeros(training_batch)
    ideal = np.zeros((num , training_batch))


    while i != 10 :

        clac_percent = 0
        ideal.fill(0)

        count = 0
        for pic in range(batch*training_batch ,(batch+1)*training_batch):
            pictures[: ,count] = data[pic, : ]
            ideal [y_train[pic],count] = 1
            count += 1
            if pic == 59999:
                batch = 0
        layer1 , layer2 , predection=evaluate_t(pictures,layer1, layer2 , weights1 , weights2 , weights3 , b1 , b2 , b3)
         
        error = (ideal - predection)**2
        
        out_deltas = delta(predection,ideal)
        layer2_deltas = delta_b(out_deltas , layer2 , weights3 ) 
        layer1_deltas = delta_b(layer2_deltas , layer1,weights2)
        weights1_ders = derive(layer1_deltas , pictures , input_num ) 
        weights2_ders = derive(layer2_deltas , layer1 , layer1_num )
        weights3_ders = derive(out_deltas , layer2 , layer2_num )
        b1_ders =np.sum(layer1_deltas, axis=1, keepdims=False)/training_batch
        b2_ders = np.sum(layer2_deltas, axis=1, keepdims=False)/training_batch
        b3_ders = np.sum(out_deltas, axis=1, keepdims=False)/training_batch
        batch += 1

        percentage = 0
        for test_sub in range(training_batch):
            if np.where(predection[:,test_sub]==max(predection[:,test_sub]))[0][0] == np.where(ideal[:,test_sub ] == 1)[0][0]:
                #print(ideal[:,test_sub])
                #print(predection[:,test_sub])
                
                percentage += 1
                #print(percentage)
                #input()
        avg_error = np.sum(error)/training_batch
        print( "cost value: " + str(avg_error))
        print("model accuracy percentage: "+str(percentage))
        
        if  percentage > 95 : 
            i +=1
        else :
            weights1 = weights1 -(learning_rate * weights1_ders) 
            weights2 = weights2 -(learning_rate * weights2_ders)
            weights3 = weights3 -(learning_rate * weights3_ders)
            b1 -= (learning_rate * b1_ders)
            b2 -= (learning_rate * b2_ders)
            b3 -= (learning_rate * b3_ders)

    if avg_error < 0.1 and learning_handler==0:
        learning_rate = learning_rate/100
        learning_handler+=1     


if savemaybe != 'y':
    save_model()

while exit != '0' :    
    print("enter picture")
    picture_test = input("enter number path")
    predection = np.zeros(num)
    test_n = np.random.randint(1,1000)
    #test=x_test[ test_n , : ].reshape(28*28)
    test = get_example_p(picture_test)
    visual = test
    test = (test-np.mean(test))/np.std(test)
    predection=evaluate(test,layer1, layer2 , weights1 , weights2 , weights3 , b1 , b2 , b3)
    print(predection)
    decision = 0
    v = 0
    for number in predection :
        if max(predection) == number:
            decision = v
        v +=1
    print(decision)
    visualize_1d_pixels(test , (28,28) )
    visualize_1d_pixels(visual , (28,28) )
    exit=input("type 0 to exit")

    