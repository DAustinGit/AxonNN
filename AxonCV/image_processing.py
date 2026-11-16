import matplotlib.pyplot as plt
from cupy.lib.stride_tricks import as_strided, sliding_window_view
from PIL import Image
import cupy as cp



def im2col(img:cp.ndarray,stride=1, kernel_size=4)->cp.ndarray:
    idx = cp.arange(img.size).reshape(img.shape)
    print(f"idx {idx.shape}")
    idx_window = sliding_window_view(idx, (kernel_size,kernel_size), axis=(0,1))
    windows = sliding_window_view(img, (kernel_size,kernel_size), axis=(0,1))
    idx_window = idx_window[:, ::stride, ::stride, :, :]
    windows = windows[:, ::stride, ::stride, :, :]
    print(f"idx shape {idx.shape}")
    print (f"windows output: {windows.shape}")
    H_out, W_out,C_out, kh, kw = windows.shape
    im2col = windows.reshape(H_out*W_out,(kernel_size**2)*C_out).T
    index = idx_window.reshape(H_out*W_out,(kernel_size**2)*C_out).T
    #print(f"im2col {im2col.shape}")
    #print(f"idx windows {index}")
    return im2col,index

def col2im(img:cp.array,col:cp.array, idx: cp.array):
    H,W,C =img.shape
    total_elements = H * W * C
    output = cp.zeros_like(img,dtype=cp.float32)
    x,y,z = cp.unravel_index(idx, (H,W,C))
    cp.add.at(output, (x,y,z,), col.astype(cp.float32))
    counts = cp.zeros(total_elements, dtype=cp.float32)
    cp.add.at(counts, idx, 1.0)
    output /= cp.maximum(counts, 1.0).reshape(H,W,C)
    reconstructed = cp.clip(output, 0, 255).astype(cp.uint8)
    return reconstructed

