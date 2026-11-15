import matplotlib.pyplot as plt
from cupy.lib.stride_tricks import as_strided, sliding_window_view
from PIL import Image
import cupy as cp

# Open the image file
img = Image.open('images//cat_1.jpg')

# Convert the image to a NumPy array
img_array = cp.asarray(img)

# Check the properties
print(img_array.shape)  # Output layout: (Height, Width, Channels)
print(img_array.dtype)  # Output typically: uint8 (values from 0 to 255)
plt.imshow(img_array.get())
#plt.show()
class image_kernel:
    def input(self, img:cp.ndarray, num_kernels:int):
        self.image = img
        if self.image.ndim == 2:
            W,H = img.shape
            C = 1
        elif self.image.ndim == 3:
            W,H,C = img.shape
        self.image_flat = self.image.reshape (W,H*C)
        rng = cp.random.default_rng()
        self.biases=rng.uniform(-0.2,0.2, size =(W, num_kernels))
        self.weights=rng.standard_normal(size=(H*C,num_kernels))
    def image_layer (self):
        print (f" image shape {self.image_flat.shape}, weights = {self.weights.shape}")
        Hidden_tensor = cp.dot(self.image_flat,self.weights)+self.biases  
        print (Hidden_tensor.shape)
        activation_output = cp.maximum(0, Hidden_tensor)
        W,H = activation_output.shape
        return activation_output.reshape(W,H)




        

layer = image_kernel()
layer1 = image_kernel()
layer2 = image_kernel()
layer.input(img_array,num_kernels=225)
output = layer.image_layer()
layer1.input(output,num_kernels=100)
output = layer1.image_layer()
layer2.input(output,num_kernels=50)
output = layer2.image_layer()
print(output.shape)