import os

import numpy as np


def load_det(file_dir):
    det_list = []
    for index in range(len(os.listdir(file_dir))):
        List = []
        file_path = os.path.join(file_dir, "road_{}.txt".format(index + 1))
        with open(file_path, "r") as f:
            for _, det in enumerate(f.readlines()):
                det = det.replace('\n', "").split(" ")
                List.append(np.array(det[1:5], dtype="int"))
        det_list.append(List)
    return det_list
