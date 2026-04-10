import cv2
import numpy as np
from numpy import uint8, clip, where
from pywt import dwt2, waverec2
from scipy.ndimage import median_filter
from scipy.signal import convolve2d


class Fusion:
    @classmethod
    def fuse(cls, img1_path, img2_path, wavelet, level, approx_rule, detail_rule, channel=1):
        if channel == 1:
            return cls.singleChFuse(img1_path, img2_path, wavelet, level, approx_rule, detail_rule)
        elif channel == 3:
            return cls.tripleChFuse(img1_path, img2_path, wavelet, level, approx_rule, detail_rule)
        else:
            raise ValueError("Channel and be only '1' for single channel and '3' for triple channel")

    @classmethod
    def singleChFuse(cls, img1_path, img2_path, wavelet, level, approx_rule, detail_rule):
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        img1_coeff = [img1]
        img2_coeff = [img2]

        for _ in range(level):
            img1_coeff = list(dwt2(img1_coeff.pop(0), wavelet)) + img1_coeff
            img2_coeff = list(dwt2(img2_coeff.pop(0), wavelet)) + img2_coeff

        fused_coeff = []
        for c1, c2 in zip(img1_coeff, img2_coeff):
            if type(c1) == type(tuple()):
                tup = []
                for band1, band2 in zip(c1, c2):
                    temp = cls.fuseBandbyRule(c1, c2, detail_rule)
                    tup.append(temp)
                fused_coeff.append(tuple(tup))
            else:
                temp = cls.fuseBandbyRule(c1, c2, approx_rule)
                fused_coeff.append(temp)

        fused_img = waverec2(fused_coeff, wavelet)
        fused_img = clip(fused_img, 0, 255).astype(uint8)

        # TODO add a path below for fusion
        fused_path = "add/a/path/here.img_ext"
        cv2.imwrite(fused_path, fused_img)
        return fused_path

    @classmethod
    def tripleChFuse(cls, img1_path, img2_path, wavelet, level, approx_rule, detail_rule):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1_R, img1_G, img1_B = cv2.split(img1)
        img2_R, img2_G, img2_B = cv2.split(img2)

        img1_R_coeff = [img1_R]
        img1_G_coeff = [img1_G]
        img1_B_coeff = [img1_B]
        img2_R_coeff = [img2_R]
        img2_G_coeff = [img2_G]
        img2_B_coeff = [img2_B]

        for _ in range(level):
            img1_R_coeff = list(dwt2(img1_R_coeff.pop(0), wavelet)) + img1_R_coeff
            img1_G_coeff = list(dwt2(img1_G_coeff.pop(0), wavelet)) + img1_G_coeff
            img1_B_coeff = list(dwt2(img1_B_coeff.pop(0), wavelet)) + img1_B_coeff
            img2_R_coeff = list(dwt2(img2_R_coeff.pop(0), wavelet)) + img2_R_coeff
            img2_G_coeff = list(dwt2(img2_G_coeff.pop(0), wavelet)) + img2_G_coeff
            img2_B_coeff = list(dwt2(img2_B_coeff.pop(0), wavelet)) + img2_B_coeff

        img1_coeffs = [img1_R_coeff, img1_G_coeff, img1_B_coeff]
        img2_coeffs = [img2_R_coeff, img2_G_coeff, img2_B_coeff]
        fused_coeff = []

        for channel in zip(img1_coeffs, img2_coeffs):
            fused_ch = []
            c1, c2 = channel
            for i in range(level + 1):
                if i == 0:
                    temp = cls.fuseBandbyRule(c1[0], c2[0], approx_rule)
                    fused_ch.append(temp)
                else:
                    fused_db = []
                    for j in range(3):
                        temp = cls.fuseBandbyRule(c1[i][j], c2[i][j], detail_rule)
                        fused_db.append(temp)
                    fused_ch.append(fused_db)
            fused_coeff.append(fused_ch)

        fused_R = waverec2(fused_coeff[0], wavelet)
        fused_G = waverec2(fused_coeff[1], wavelet)
        fused_B = waverec2(fused_coeff[2], wavelet)
        fused_img = cv2.merge([fused_R, fused_G, fused_B])
        fused_img = clip(fused_img, 0, 255).astype(uint8)

        # TODO add a path below for fusion
        fused_path = "add/a/path/here.img_ext"
        cv2.imwrite(fused_path, fused_img)
        return fused_path

    @classmethod
    def fuseBandbyRule(cls, band1, band2, rule:str):
        match rule.upper():
            case "LV":
                return cls.LVRule(band1, band2)
            case "SML":
                return cls.SMLRule(band1, band2)
            case "MIN":
                return np.minimum(band1, band2)
            case "MAX":
                return np.maximum(band1, band2)
            case "AVG":
                return (band1 + band2)/2
            case _:
                raise ValueError(f"Rule: {rule} is not a valid fusion rule")

    @classmethod
    def LVRule(cls, band1, band2):
        ksize = (5, 5)

        def __lv(band):
            band = band.astype(np.float32)
            mean = cv2.boxFilter(band, ddepth=-1, ksize=ksize)
            mean_sq = cv2.boxFilter(band ** 2, ddepth=-1, ksize=ksize)
            return mean_sq - mean ** 2

        var_band1 = __lv(band1)
        var_band2 = __lv(band2)
        decision_map = var_band1 >= var_band2
        decision_map = median_filter(decision_map.astype(np.uint8), size=3).astype(bool)
        return where(decision_map, band1, band2).astype(band1.dtype)

    @classmethod
    def SMLRule(cls, band1, band2):
        def __sml(band):
            f1 = np.array([[1, -2, 1]])
            f2 = np.array([[1], [-2], [1]])
            return (np.abs(convolve2d(band, f1, mode='same', boundary='symm')) +
                    np.abs(convolve2d(band, f2, mode='same', boundary='symm')))

        return where(__sml(band2) >= __sml(band1), band2, band1)
