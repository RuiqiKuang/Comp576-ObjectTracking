import argparse
import time

import cv2
import numpy as np

import const
import detection
import utils
from kalman import Kalman

# 状态转移矩阵
A = np.array([[1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
# 控制输入矩阵B
B = None
# 过程噪声协方差矩阵Q，p(w)~N(0,Q)
Q = np.eye(A.shape[0]) * 0.1
# 状态观测矩阵
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]])
# 观测噪声协方差矩阵R，p(v)~N(0,R)
R = np.eye(H.shape[0]) * 1
# 状态估计协方差矩阵P
P = np.eye(A.shape[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOT demo')
    parser.add_argument('--display', action='store_true', help='Display  output')
    parser.add_argument('--save_img', action='store_true', help='record output frames')
    args = parser.parse_args()
    # print(args)
    dis = args.display
    save = args.save_img
    t1 = time.time()

    cap = cv2.VideoCapture(const.VIDEO_PATH)
    det_list_all = detection.load_det(const.FILE_DIR)
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(const.VIDEO_OUTPUT_PATH, fourcc, fps, sz, True)

    state_list = []
    frame_cnt = 1
    for det_list_frame in det_list_all:
        ret, frame = cap.read()
        if not ret:
            break
        for s in state_list:
            s.predict()
        # 匹配
        det_list = [utils.box2det(detection) for detection in det_list_frame]
        state_rem_list, det_rem_list, match_list = utils.association(state_list, det_list)
        # 状态没匹配上,超过一定时间就删除
        state_del = list()
        for idx in state_rem_list:
            status, _, _ = state_list[idx].update()
            if not status:
                state_del.append(idx)
        state_list = [state_list[i] for i in range(len(state_list)) if i not in state_del]
        # 测量没匹配上的，作为新生目标进行航迹起始
        for idx in det_rem_list:
            state_list.append(Kalman(A, B, H, Q, R, utils.det2state(det_list[idx]), P))
        # Visualization
        # 显示所有det到图像上
        for det in det_list_frame:
            cv2.rectangle(frame, tuple(det[:2]), tuple(det[2:]), const.COLOR_DET, thickness=1)
        # 显示所有的state到图像上
        for kalman in state_list:
            pos = utils.state2box(kalman.X_posterior)
            cv2.rectangle(frame, tuple(pos[:2]), tuple(pos[2:]), const.COLOR_STA, thickness=2)
        # 绘制轨迹
        for kalman in state_list:
            tracks_list = kalman.track
            for idx in range(len(tracks_list) - 1):
                last_frame = tracks_list[idx]
                cur_frame = tracks_list[idx + 1]
                cv2.line(frame, last_frame, cur_frame, kalman.track_color, 2)

        cv2.putText(frame, str(frame_cnt), (0, 50), color=(255, 0, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)
        if dis:
            cv2.imshow('Road', frame)
        if save:
            cv2.imwrite("./record/{}/{}_{}.jpg".format(const.cache, const.cache, frame_cnt), frame)
        video_writer.write(frame)
        cv2.waitKey(1)
        frame_cnt += 1
    t2 = time.time()
    print("Using {} second".format(t2 - t1))
    print("Using {} second for kalman".format(const.time_kalman))
    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()
