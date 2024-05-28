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




class QuantizeAndEncode2D:
    def __init__(self):
        self.luminance_qt, self.chrominance_qt = self.get_quantization_table()
        self.zigzag_indices = self.get_zizag_indices()

    @staticmethod
    def get_quantization_table():
        luminance_qt =  np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

        chrominance_qt = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ])

        return luminance_qt, chrominance_qt


    def quantize(self, Y, Cb, Cr):
        Y_q = np.zeros_like(Y, dtype=np.int8)
        Cb_q = np.zeros_like(Cb, dtype=np.int8)
        Cr_q = np.zeros_like(Cr, dtype=np.int8)
        
        k = 8
        h, w = Y.shape
        for i,j in product(range(0, h, k), range(0, w, k)):
            Y_q[i:i+k, j:j+k] = np.round(Y[i:i+k, j:j+k] / self.luminance_qt).astype(np.int8)

        h, w = Cb.shape
        for i,j in product(range(0, h, k), range(0, w, k)):
            Cb_q[i:i+k, j:j+k] = np.round(Cb[i:i+k, j:j+k] / self.chrominance_qt).astype(np.int8)
            Cr_q[i:i+k, j:j+k] = np.round(Cr[i:i+k, j:j+k] / self.chrominance_qt).astype(np.int8)

        return Y_q, Cb_q, Cr_q
    
    def dequantize(self, Y_q, Cb_q, Cr_q):
        Y = np.zeros_like(Y_q, dtype=np.float32)
        Cb = np.zeros_like(Cb_q, dtype=np.float32)
        Cr = np.zeros_like(Cr_q, dtype=np.float32)
        
        k = 8
        h, w = Y_q.shape
        for i,j in product(range(0, h, k), range(0, w, k)):
            Y[i:i+k, j:j+k] = Y_q[i:i+k, j:j+k] * self.luminance_qt

        h, w = Cb_q.shape
        for i,j in product(range(0, h, k), range(0, w, k)):
            Cb[i:i+k, j:j+k] = Cb_q[i:i+k, j:j+k] * self.chrominance_qt
            Cr[i:i+k, j:j+k] = Cr_q[i:i+k, j:j+k] * self.chrominance_qt

        return Y, Cb, Cr

    @staticmethod
    def get_zizag_indices():
        zigzag_indices = [np.diagonal(np.fliplr(np.arange(8*8).reshape(8, 8)), k)for k in range(1-8, 8)]
        zigzag_indices = [x if i%2==0 else x[::-1] for i , x in enumerate(zigzag_indices)]
        return np.concatenate(zigzag_indices)[::-1]

    def encode_8x8(self, signal : np.ndarray):
        """Run-length encodes a 1D signal (e.g., quantized DCT coefficients) with zigzag ordering."""
        # Apply zigzag ordering to the signal
        signal = signal.flatten()[self.zigzag_indices]

        encoded = []
        count = 1
        for i in range(1, len(signal)):
            if signal[i] == signal[i - 1]:
                count += 1
            else:
                encoded.append((signal[i - 1], count))
                count = 1
        encoded.append((signal[-1], count))
        return encoded

    
    def decode_8x8(self, code : list):
        output = np.zeros(64, dtype=np.float32)
        idx = 0
        for val, count in code:
            output[idx:idx+count] = val
            idx += count
        return output[self.zigzag_indices].reshape(8,8)
    
    def encode_image(self, img) -> list[list]:
        h, w = img.shape
        assert h%8==0 and w%8==0
        encoded = []
        for i, j in product(range(0, h, 8), range(0, w, 8)):
            encoded.append(self.encode_8x8(img[i:i+8, j:j+8]))
        return encoded
    
    def decode_image(self, code : list[list], h, w) -> np.ndarray:
        output = np.zeros((h, w), dtype=np.float32)
        idx = 0
        for i, j in product(range(0, h, 8), range(0, w, 8)):
            output[i:i+8, j:j+8] = self.decode_8x8(code[idx])
            idx += 1
        return output
    
    def forward(self, Y, Cb, Cr):
        Y_q, Cb_q, Cr_q = self.quantize(Y, Cb, Cr)
        return self.encode_image(Y_q), self.encode_image(Cb_q), self.encode_image(Cr_q)
    
    def backward(self, Y_q, Cb_q, Cr_q, h, w):
        Y = self.decode_image(Y_q, h, w)
        Cb = self.decode_image(Cb_q, h//2, w//2)
        Cr = self.decode_image(Cr_q, h//2, w//2)
        return self.dequantize(Y, Cb, Cr)




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


    qe = QuantizeAndEncode2D()
    h,w = Y.shape

    Y_q, Cb_q, Cr_q = qe.forward(Y_freq, Cb_freq, Cr_freq)
    Y_i, Cb_i, Cr_i = qe.backward(Y_q, Cb_q, Cr_q, h, w)

    Y_i = transform_2d.idct(Y_i)
    Cb_i = transform_2d.idct(Cb_i)
    Cr_i = transform_2d.idct(Cr_i)

    print(f"[Y] passing= {np.allclose(Y, Y_i)}")
    print(f"[Cb] passing={np.allclose(Cb, Cb_i)}")
    print(f"[Cr] passing={np.allclose(Cr, Cr_i)}")

    recon_rgb_array = ycbcr_to_rgb_with_upsample(Y_i, Cb_i, Cr_i)
    recon_rgb_img = Image.fromarray(recon_rgb_array)
    recon_rgb_img.save(imgpath.with_name("recon-"+imgpath.stem+".png"))

    compare_images(orig_img, recon_rgb_img)

if __name__ == "__main__":
    main(Path("files/hunger-games-0.png"))