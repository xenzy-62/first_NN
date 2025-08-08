import numpy as np
from PIL import Image

i = 0
exit = "c"
clac_percent = 0
savemaybe = "b"

def save_model():
    name = input("enter model name: ")
    np.save(name + "weights1.npy" , weights1 )
    np.save(name + "weights2.npy" , weights2 )
    np.save(name + "weights3.npy" , weights3 )
    np.save(name + "b1.npy" , b1 )
    np.save(name + "b2.npy" , b2 )
    np.save(name + "b3.npy" , b3 )

def load_model():

    name = input("enter model name: ")
    return np.load("C:\\code\\py\\"+ name +"weights1.npy") , np.load("C:\\code\\py\\"+ name +"weights2.npy" ) , np.load("C:\\code\\py\\"+ name +"weights3.npy" ) , np.load("C:\\code\\py\\"+ name +"b1.npy" ) ,np.load("C:\\code\\py\\"+ name +"b2.npy" ) , np.load("C:\\code\\py\\"+ name +"b3.npy" ) 
    


def sigmoid(z):
    z = np.clip(z, -50, 50)  # Prevents overflow
    return 1 / (1 + np.exp(-z))

def evaluate( input , layer1 , layer2 , weights1 , weights2 , weights3 , b1 , b2 , b3  ):
    z1=np.dot( weights1 , input  ) + b1
    layer1= sigmoid(z1)

    z2=np.dot( weights2 , layer1 ) + b2
    layer2= sigmoid(z2)

    z3=np.dot( weights3 , layer2 ) + b3
     

    return layer1 , layer2 , sigmoid(z3)


def get_example_p(img_path):
    img = Image.open(img_path)
    img_arr = 1-(np.array(img)[:,:,0].flatten() / 255.0)
    return img_arr

def get_example(img_path):
    img = Image.open(img_path)
    img_arr = np.array(img)[:,:,3].flatten() / 255.0
    return img_arr

def get_examples(dataset):
    
    data = np.zeros((28*28,1000000))
    data_label = np.zeros(1000000)
    for i in range(1000000):
        number = np.random.randint(0,9)
        data[:,i] = get_example( dataset +"\\"+ str(number) +"\\"+ str(number) + "\\" + str(np.random.randint(0,10000))+ ".png")
        data_label [i] = number  
        print( "data loaded percentage: " +str(i/10000)) 
    return data , data_label    

def delta_b(pre_deltas , cur_layer , weights ):

    prod=np.dot(pre_deltas , weights )

    return prod * cur_layer * ( 1 - cur_layer )
        

def delta(out_layer, data_label):
    example = np.zeros(len(out_layer))
    example[int(data_label)] = 1
    return (out_layer - example)

def derive(deltas , pre_activations , act_num ) :
    L=len(deltas)
    weights_dersss = np.zeros( (L , act_num) )
    for k in range(act_num):
        for j in range(L):
            weights_dersss[ j , k ] = pre_activations[k]*deltas[j]
    return weights_dersss



batch_size = 1000000

learning_rate = 0.1



data = np.zeros(( 28*28 , batch_size ))
data_label= np.zeros(batch_size)

input_num= 28*28

layer1_num= 8
layer1=np.zeros(layer1_num)

layer2_num= 8
layer2=np.zeros(layer2_num)

num=10
output=np.zeros(num)

weights1=np.random.randn( layer1_num , input_num )
weights2=np.random.randn(layer2_num , layer1_num )
weights3=np.random.randn(num , layer2_num )

b1=np.random.randn(layer1_num)
b2=np.random.randn(layer2_num)
b3=np.random.randn(num)

print("wanna load a model?")

savemaybe = input("enter 'y' if yes: ")
if savemaybe == 'y' :
    weights1 , weights2 , weights3 , b1 , b2 , b3 = load_model()



predection= np.zeros(num)
picture = np.zeros(28*28)
batch = 1
if savemaybe != "y":
    data , data_label = get_examples("C:\\Users\\hassa\\.cache\\kagglehub\\datasets\\jcprogjava\\handwritten-digits-dataset-not-in-mnist\\versions\\4\\dataset")
    while i == 0 :

        clac_percent = 0
        error = 0
        weights1_ders = np.zeros((layer1_num, input_num))
        weights2_ders = np.zeros((layer2_num, layer1_num))
        weights3_ders = np.zeros((num, layer2_num))
        b1_ders = np.zeros(layer1_num)
        b2_ders = np.zeros(layer2_num)
        b3_ders = np.zeros(num)
    
    

        out_deltas = np.zeros(num)
        layer2_deltas = np.zeros(layer2_num)
        layer1_deltas = np.zeros(layer1_num)
        weights1_ders = np.zeros((layer1_num , input_num) )
        weights2_ders = np.zeros((layer2_num , layer1_num) )
        weights3_ders = np.zeros((num , layer2_num))
        
        error = np.zeros(num)

        for ex_num in range((batch-1)*100,(batch*100)-1) :
            if ex_num == batch_size - 10 :
                data , data_label = get_examples("C:\\Users\\hassa\\.cache\\kagglehub\\datasets\\jcprogjava\\handwritten-digits-dataset-not-in-mnist\\versions\\4\\dataset")
                batch = 1
            picture = data[:, ex_num ]
            layer1 , layer2 , predection=evaluate(picture,layer1, layer2 , weights1 , weights2 , weights3 , b1 , b2 , b3)
            ideal = np.zeros(num)
            example= int(data_label[ex_num])
            ideal [example] = 1 
            error += (ideal - predection)**2
            if max(predection) != predection[example]:
                clac_percent += 1
            out_deltas = delta(predection,int(data_label[ex_num]))
            layer2_deltas = delta_b(out_deltas , layer2 , weights3 ) 
            layer1_deltas = delta_b(layer2_deltas , layer1,weights2)
            weights1_ders += derive(layer1_deltas , picture , input_num ) 
            weights2_ders += derive(layer2_deltas , layer1 , layer1_num )
            weights3_ders += derive(out_deltas , layer2 , layer2_num )
            b1_ders += layer1_deltas
            b2_ders += layer2_deltas
            b3_ders += out_deltas
        batch += 1
        percentage = 100 - clac_percent
        print( "cost value: " + str(np.sum(error/100)))
        print( "model accuracy percentage: " + str(percentage))
        avg_error = np.sum(error)/100
        if np.sum(avg_error) < 0.1 or percentage > 97 : 
            i +=1
        else :
            weights1 = weights1 -(learning_rate * weights1_ders/100) 
            weights2 = weights2 -(learning_rate * weights2_ders/100)
            weights3 = weights3 -(learning_rate * weights3_ders/100)
            b1 -= (learning_rate * b1_ders / 100)
            b2 -= (learning_rate * b2_ders / 100)
            b3 -= (learning_rate * b3_ders / 100)
        
        if avg_error < 0.3 :
            learning_rate = learning_rate/10


if savemaybe != 'y':
    save_model()

while exit != '0' :    
    print("enter picture")
    picture_test = input("enter number path")


    test=get_example(picture_test)
    good = get_example("C:\\Users\\hassa\\.cache\\kagglehub\\datasets\\jcprogjava\\handwritten-digits-dataset-not-in-mnist\\versions\\4\\dataset\\4\\4\\8.png")
    layer1 , layer2 , predection=evaluate(test,layer1, layer2 , weights1 , weights2 , weights3 , b1 , b2 , b3)
    print(predection)
    decision = 0
    v = 0
    for number in predection :
        if max(predection) == number:
            decision = v
        v +=1
    print(decision)

    exit=input("type 0 to exit")

    