from datetime import datetime
import cv2
import numpy as np

class Registration:
    @classmethod
    def register(cls, ref_path, query_path):
        query = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
        query_Color = cv2.imread(query_path)
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

        warp = np.eye(2, 3, dtype=np.float32)

        cc, warp = cv2.findTransformECC(
            query,
            ref,
            warp,
            cv2.MOTION_AFFINE
        )
        reg_img = cv2.warpAffine(query_Color, warp, (query.shape[1], query.shape[0]))

        now = datetime.now()
        reg_path = f"photos/registered/REG{now.year}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}{now.second:02}.png"
        cv2.imwrite(reg_path, reg_img)

        return reg_path