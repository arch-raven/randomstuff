from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path



def compare_images(img, tgt):
    diff = np.array(ImageChops.difference(img, tgt))
    diff = diff * (255.0 / diff.max())
    Image.fromarray(diff.astype(np.uint8)).show()
    # diff.flatten() - np.abs(np.array(img) - np.array(tgt)).flatten() // should be all 0 right?
    # plt.hist(diff.flatten() - np.abs(np.array(img) - np.array(tgt)).flatten())


def rgb_to_ycbcr(original):
    """RGB -> YCbCr Color Space conversion"""
    Y = 0.299 * original[..., 0] + 0.587 * original[..., 1] + 0.114 * original[..., 2]
    Cb = -0.1687 * original[..., 0] - 0.3313 * original[..., 1] + 0.5 * original[..., 2] + 128
    Cr = 0.5 * original[..., 0] - 0.4187 * original[..., 1] - 0.0813 * original[..., 2] + 128
    return Y, Cb, Cr


def ycbcr_to_rgb_with_upsample(Y, Cb, Cr, upscale_chrominance_factor=2):
    """Convert YCbCr to RGB and upsample chrominance channels if needed."""
    # Upsample chrominance channels if upscale_chrominance_factor is greater than 1
    if upscale_chrominance_factor > 1:
        Cb = np.kron(Cb, np.ones((upscale_chrominance_factor, upscale_chrominance_factor)))
        Cr = np.kron(Cr, np.ones((upscale_chrominance_factor, upscale_chrominance_factor)))

    # Ensure the upsampled chrominance channels match the dimensions of the luminance channel
    assert Y.shape == Cb.shape and Y.shape == Cr.shape
    # Convert YCbCr back to RGB
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    rgb = np.stack((R, G, B), axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def avgpool(Cb, Cr, window):
    """By taking mean over a 2x2 grid"""
    h, w = Cb.shape[0] // window, Cb.shape[1] // window
    Cb = Cb.reshape(h, window, w, window).mean(axis=(1,3))
    Cr = Cr.reshape(h, window, w, window).mean(axis=(1,3))
    return Cb, Cr


class CosineTransform2D:
    def __init__(self, k : int):
        self.k = k
        self.normalize_mat = self.get_normalize_mat(k)
        self.harmonics = self.get_harmonics(k)
    
    @staticmethod
    def get_normalize_mat(n):
        mat = np.ones(n).reshape(n,1)
        mat[0, 0] = 1 / np.sqrt(2)
        return (mat @ mat.T)

    
    @staticmethod
    def get_harmonics(n):
        spatial = np.arange(n).reshape((n, 1))
        spectral = np.arange(n).reshape((1, n))
        
        spatial = 2 * spatial + 1
        spectral = (spectral * np.pi) / (2 * n)
        
        return np.cos(spatial @ spectral)
    

    def dct_kxk(self, in_mat_2d):
        assert all([n == self.k for n in in_mat_2d.shape]), f"Expected square matrix of length {self.k}, received {in_mat_2d.shape}"
        return (1 / np.sqrt(2 * self.k)) * self.normalize_mat * (self.harmonics.T @ in_mat_2d @ self.harmonics)

    def idct_kxk(self, in_mat_2d):
        assert all([n == self.k for n in in_mat_2d.shape]), f"Expected square matrix of length {self.k}, received {in_mat_2d.shape}"
        return (1 / np.sqrt(2 * self.k)) * (self.harmonics @ (self.normalize_mat * in_mat_2d) @ self.harmonics.T)


    def dct(self, img):
        h, w = img.shape
        k = self.k
        output = np.zeros_like(img)
        
        for i,j in product(range(0, h, k), range(0, w, k)):
            subimage = img[i:i+k, j:j+k]
            output[i:i+k, j:j+k] = self.dct_kxk(subimage)
        return output
    
    def idct(self, img):
        h, w = img.shape
        k = self.k
        output = np.zeros_like(img)
        
        for i,j in product(range(0, h, k), range(0, w, k)):
            subimage = img[i:i+k, j:j+k]
            output[i:i+k, j:j+k] = self.idct_kxk(subimage)
        return output
    


def main(imgpath : Path):
    orig_img = Image.open(imgpath).convert('RGB')
    rgb_array = np.array(orig_img)
    print(f"original is RGB image of size {rgb_array.shape}, dtype: {rgb_array.dtype}")
    
    h, w, c = rgb_array.shape
    assert c == 3
    if h%16!=0 or w%16!=0: # 8, after downsampling another factor of 2 => 8*2
        rgb_array = rgb_array[:(h//16)*16, :(w//16)*16]
        print(f"rgb_array cropped to dim = {rgb_array.shape}")

    Y, Cb, Cr = rgb_to_ycbcr(rgb_array)
    Cb, Cr = avgpool(Cb, Cr, window=2)

    transform_2d = CosineTransform2D(8)
    
    Y_freq = transform_2d.dct(Y)
    Cb_freq = transform_2d.dct(Cb)
    Cr_freq = transform_2d.dct(Cr)

    Y_i = transform_2d.idct(Y_freq)
    Cb_i = transform_2d.idct(Cb_freq)
    Cr_i = transform_2d.idct(Cr_freq)

    print(f"[Y] passing= {np.allclose(Y, Y_i)}")
    print(f"[Cb] passing={np.allclose(Cb, Cb_i)}")
    print(f"[Cr] passing={np.allclose(Cr, Cr_i)}")

    recon_rgb_array = ycbcr_to_rgb_with_upsample(Y_i, Cb_i, Cr_i)
    recon_rgb_img = Image.fromarray(recon_rgb_array)
    recon_rgb_img.save(imgpath.with_name("recon-"+imgpath.stem+".png"))

    compare_images(orig_img, recon_rgb_img)

if __name__ == "__main__":
    main(Path("files/hunger-games-0.png"))