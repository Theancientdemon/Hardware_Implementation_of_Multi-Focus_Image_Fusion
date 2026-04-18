from datetime import datetime

import cv2
import numpy as np
from pywt import wavedec
from skimage.registration import phase_cross_correlation


class Registration:
    @classmethod
    def register(cls, ref, query, wavelet, level):
        query = cv2.imread(query, cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(ref, cv2.IMREAD_GRAYSCALE)

        q_wt = wavedec(query, wavelet, level=level)[0]
        r_wt = wavedec(ref, wavelet, level=level)[0]

        (x, y), err, _ = phase_cross_correlation(r_wt, q_wt)

        M = np.float32([[1, 0, y * (2 ** level)],
                        [0, 1, x * (2 ** level)]])

        reg_img,_,_ = cv2.warpAffine(query, M, (query.shape[1], query.shape[0])), x * (2 ** level), y * (2 ** level)

        now = datetime.now()
        reg_path = f"photos/registered/{now.year}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}{now.second:02}.png"
        cv2.imwrite(reg_path, reg_img)

        return reg_path
