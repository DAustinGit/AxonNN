import matplotlib.pyplot as plt
from cupy.lib.stride_tricks import as_strided, sliding_window_view
from PIL import Image
import cupy as cp

# Open the image file
img = Image.open('cat_1.jpg')

# Convert the image to a NumPy array
img_array = cp.asarray(img)

# Check the properties
print(img_array.shape)  # Output layout: (Height, Width, Channels)
print(img_array.dtype)  # Output typically: uint8 (values from 0 to 255)
#plt.imshow(img_array.get())
#plt.show()
img_2d = img_array.reshape(img_array.shape[2], -1)




class convelution:
    def __init__(self, num_kernels:int, kernel_size:int):
        self.m = (kernel_size**2)*3
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        rng = cp.random.default_rng()
        self.biases=rng.uniform(-0.2,0.2, size =(num_kernels, 1))
        self.weights=rng.standard_normal(size=(num_kernels, self.m)) * cp.sqrt(2.0 / self.m)
        print (self.biases.shape)

    def im2col(self,img:cp.ndarray,stride=1, kernel_size=4):
        windows = sliding_window_view(img, (kernel_size,kernel_size), axis=(0,1))
        windows = windows[:, ::stride, ::stride, :, :]
        print (f"windows output: {windows.shape}")
        self.H_out, self.W_out,C_out, kh, kw = windows.shape
        self.im2col = windows.reshape(self.H_out*self.W_out,(kernel_size**2)*C_out).T 
        print (f"im2col new output: {self.im2col.shape}")

        
  
    def forward(self):
        print(f"weights = {self.weights.shape}, self.im2col {self.im2col.shape}")
        self.output_tensor = cp.dot(self.weights, self.im2col)+self.biases
        print (f"forward pass tensor {self.output_tensor.shape}")
        return self.output_tensor


    def pooling(self, window:int,stride:int):
        pool_input = self.output_tensor.reshape(self.num_kernels, self.H_out, self.W_out)
        self.C,self.H,self.W=pool_input.shape
        self.H_out =(H-window)//stride+1
        self.W_out = (W-window)//stride+1
        s_char,s_row,s_col = pool_input.strides
        new_shape = (C, self.H_out, self.W_out, window,window)
        new_strides = (s_char, s_row, s_col, s_row * stride, s_col * stride)
        pool = as_strided(pool_input, shape=new_shape, strides=new_strides)
        flat_windows = pool.reshape(C, self.H_out * self.W_out, window * window)
        pooled_flat = cp.max(flat_windows, axis=2)
        bool_ind = cp.argmax(flat_windows, axis=2)
        grid = cp.arange(4)[None, None, :]
        self.mask = (bool_ind[:, :, None] == grid)   #creates a mask indicating which pixel gtot selected by cp.max stored for backpropergation
        pooled_output = pooled_flat.reshape(C, self.H_out, self.W_out)
        print(f"pooled output tensor {pooled_output.shape}")
    def pooling_backpass(self, incoming_gradient):
        m,n = incoming_gradient.shape
        dY = incoming_gradient.reshape(self.C, self.H,self.W)


    
kernel = convelution(num_kernels=4,kernel_size=4)
kernel.im2col(img=img_array, stride=1)
kernel.forward()
kernel.pooling(window=4,stride=4)


