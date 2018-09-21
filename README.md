# Tensorflow GAN on MNIST Data Generation

A GAN network that generates MNIST digits data

## Code Files:
loader.py: load MNIST real images  
generator.py: a generator network with three convolutional layers    
discriminator.py: a discriminator network with two convolutional layers and two dense layers  
train.py: training the gan  
inference.py: generate MNIST images  

## Dataset:
Tensorflow Tutorial MNIST dataset  
## Result:
### Real MNIST Images  
![Real Image 1](result/real/0.png)
![Real Image 2](result/real/1.png)
![Real Image 3](result/real/2.png)  
### Generated MNIST Images Before Training
![Noise Image 1](result/noise/0.png)
![Noise Image 2](result/noise/1.png)
![Noise Image 3](result/noise/2.png)    
### Generated MNIST Images After Training
![Generate Image 1](result/generated/0.png)
![Generate Image 2](result/generated/1.png)
![Generate Image 3](result/generated/2.png)  
