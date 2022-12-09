import os

import cv2
import numpy as np

from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace, \
    show_image_from_matrix

# 目标的状态X = [x,y,h,w,delta_x,delta_y],中心坐标，宽高，中心坐标速度

IOU_Threshold = 0.3  # 匹配时的阈值

# 状态转移矩阵
A = np.array([[1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

# 状态观测矩阵
H = np.eye(6)
# 过程噪声协方差矩阵Q
Q = np.eye(6) * 0.1

# 观测噪声协方差矩阵R，p(v)~N(0,R)
R = np.eye(6) * 1

# 控制输入矩阵B
B = None
# 状态估计协方差矩阵P初始化
P = np.eye(6)

if __name__ == "__main__":

    video_path = "./data/road.mp4"
    label_path = "./data/labels"
    file_name = "road"
    cap = cv2.VideoCapture(video_path)
    rets, frame = cap.read()
    cap.release()
    show_image_from_matrix(frame)
    ROI = cv2.selectROI("select initial target box", frame)
    print(ROI)
    initial_target_box = [ROI[0], ROI[1], ROI[0] + ROI[2], ROI[1] + ROI[3]]
    initial_box_state = xyxy_to_xywh(initial_target_box)
    initial_state = np.array([[initial_box_state[0], initial_box_state[1], initial_box_state[2], initial_box_state[3],
                               0, 0]]).T

    cap = cv2.VideoCapture(video_path)
    out = None
    SAVE_VIDEO = False
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('kalman_output.mp4', fourcc, 20, (768, 576))

    # ---------状态初始化----------------------------------------
    frame_counter = 1
    X_posterior = np.array(initial_state)
    P_posterior = np.array(P)
    Z = np.array(initial_state)
    trace_list = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        last_box_posterior = xywh_to_xyxy(X_posterior[0:4])
        plot_one_box(last_box_posterior, frame, color=(255, 255, 255), target=False)
        if not ret:
            break
        # print(frame_counter)
        label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
        with open(label_file_path, "r") as f:
            content = f.readlines()
            max_iou = IOU_Threshold
            max_iou_matched = False
            # ---------使用最大IOU来寻找观测值------------
            for j, data_ in enumerate(content):
                data = data_.replace('\n', "").split(" ")
                xyxy = np.array(data[1:5], dtype="float")
                plot_one_box(xyxy, frame)
                iou = cal_iou(xyxy, xywh_to_xyxy(X_posterior[0:4]))
                if iou > max_iou:
                    target_box = xyxy
                    max_iou = iou
                    max_iou_matched = True
            if max_iou_matched:
                # 如果找到了最大IOU BOX,则认为该框为观测值
                plot_one_box(target_box, frame, target=True)
                xywh = xyxy_to_xywh(target_box)
                box_center = (int((target_box[0] + target_box[2]) // 2), int((target_box[1] + target_box[3]) // 2))
                trace_list = updata_trace_list(box_center, trace_list, 100)
                cv2.putText(frame, "Tracking", (int(target_box[0]), int(target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0), 2)
                dx = xywh[0] - X_posterior[0]
                dy = xywh[1] - X_posterior[1]
                Z[0:4] = np.array([xywh]).T
                Z[4::] = np.array([dx, dy])
        if max_iou_matched:
            # -----进行先验估计-----------------
            X_prior = np.dot(A, X_posterior)
            box_prior = xywh_to_xyxy(X_prior[0:4])
            # -----计算状态估计协方差矩阵P--------
            P_prior_1 = np.dot(A, P_posterior)
            P_prior = np.dot(P_prior_1, A.T) + Q
            # ------计算卡尔曼增益---------------------
            k1 = np.dot(P_prior, H.T)
            k2 = np.dot(np.dot(H, P_prior), H.T) + R
            K = np.dot(k1, np.linalg.inv(k2))
            # --------------后验估计------------
            X_posterior_1 = Z - np.dot(H, X_prior)
            X_posterior = X_prior + np.dot(K, X_posterior_1)
            box_posterior = xywh_to_xyxy(X_posterior[0:4])
            # ---------更新状态估计协方差矩阵P-----
            P_posterior_1 = np.eye(6) - np.dot(K, H)
            P_posterior = np.dot(P_posterior_1, P_prior)
        else:
            # 此时直接迭代，不使用卡尔曼滤波
            X_posterior = np.dot(A, X_posterior)
            box_posterior = xywh_to_xyxy(X_posterior[0:4])
            box_center = (
                (int(box_posterior[0] + box_posterior[2]) // 2), int((box_posterior[1] + box_posterior[3]) // 2))
            trace_list = updata_trace_list(box_center, trace_list, 20)
            cv2.putText(frame, "Lost", (box_center[0], box_center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)

        draw_trace(frame, trace_list)
        cv2.putText(frame, "TRACKER", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Last frame's best estimation", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow('track', frame)
        if SAVE_VIDEO:
            out.write(frame)

        frame_counter = frame_counter + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
