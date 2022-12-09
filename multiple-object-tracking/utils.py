import numpy as np

from match import Matcher


def box2state(box):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return np.array([[center_x, center_y, w, h, 0, 0]]).T  # 定义为 [中心x,中心y,宽w,高h,dx,dy]


def state2box(state):
    center_x = state[0]
    center_y = state[1]
    w = state[2]
    h = state[3]
    return [int(i) for i in [center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2]]


def box2det(box):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return np.array([[center_x, center_y, w, h]]).T  # 定义为 [中心x,中心y,宽w,高h]


def det2box(mea):
    center_x = mea[0]
    center_y = mea[1]
    w = mea[2]
    h = mea[3]
    return [int(i) for i in [center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2]]


def det2state(mea):
    return np.row_stack((mea, np.zeros((2, 1))))


def state2det(state):
    return state.X_prior[0:4]


def cal_iou(state, measure):
    state = det2box(state)  # [lt_x, lt_y, rb_x, rb_y].T
    measure = det2box(measure)  # [lt_x, lt_y, rb_x, rb_y].T
    s_tl_x, s_tl_y, s_br_x, s_br_y = state[0], state[1], state[2], state[3]
    m_tl_x, m_tl_y, m_br_x, m_br_y = measure[0], measure[1], measure[2], measure[3]
    x_min = max(s_tl_x, m_tl_x)
    x_max = min(s_br_x, m_br_x)
    y_min = max(s_tl_y, m_tl_y)
    y_max = min(s_br_y, m_br_y)
    inter_h = max(y_max - y_min + 1, 0)
    inter_w = max(x_max - x_min + 1, 0)
    inter = inter_h * inter_w
    if inter == 0:
        return 0
    else:
        return inter / ((s_br_x - s_tl_x) * (s_br_y - s_tl_y) + (m_br_x - m_tl_x) * (m_br_y - m_tl_y) - inter)


def association(kalman_list, det_list):
    """
    多目标关联，使用最大权重匹配
    :param kalman_list: 状态列表，存着每个kalman对象，已经完成预测外推
    :param det_list: 量测列表，存着矩阵形式的目标量测 ndarray [c_x, c_y, w, h].T
    :return:
    """

    state_rec = {i for i in range(len(kalman_list))}
    det_rec = {i for i in range(len(det_list))}
    state_list = list()
    for kalman in kalman_list:
        state = kalman.X_prior
        state_list.append(state[0:4])

    # 进行匹配得到一个匹配字典
    M = Matcher()
    match_dict = M.match(state_list, det_list)

    # 根据匹配字典，将匹配上的直接进行更新，没有匹配上的返回
    state_used = set()
    det_used = set()
    match_list = list()
    for state, measure in match_dict.items():
        state_index = int(state.split('_')[1])
        det_index = int(measure.split('_')[1])
        match_list.append([state2box(state_list[state_index]), det2box(det_list[det_index])])
        kalman_list[state_index].update(det_list[det_index])
        state_used.add(state_index)
        det_used.add(det_index)

    # 求出未匹配状态和量测，返回
    return list(state_rec - state_used), list(det_rec - det_used), match_list
