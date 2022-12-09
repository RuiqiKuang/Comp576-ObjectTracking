import os

import numpy as np


def load_det(file_dir):
    """
    根据file目录，产生观测list
    :param file_dir: 每帧观测分别对应一个txt文件，每个文件中多个目标观测逐行写入
    :return: 所有观测list，[[帧1所有观测],[帧2所有观测]]
    """
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
