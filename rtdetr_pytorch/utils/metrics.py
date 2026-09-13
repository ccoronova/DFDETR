import numpy as np

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """
    计算每个类别的AP、precision、recall、F1和PR曲线
    tp: [N]，每个预测是否为TP
    conf: [N]，每个预测的置信度
    pred_cls: [N]，每个预测的类别
    target_cls: [M]，所有GT的类别
    返回:
        precision, recall, ap, f1, ap_class, pr_curve
    """
    # 排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls)))
    ap, precision, recall, f1, pr_curve = [], [], [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()
        if n_p == 0 or n_gt == 0:
            ap.append(0)
            precision.append(0)
            recall.append(0)
            f1.append(0)
            pr_curve.append(np.zeros((101, 2)))
            continue
        # 累加TP/FP
        fpc = (1 - tp[i]).cumsum()
        tpc = tp[i].cumsum()
        recall_curve = tpc / (n_gt + 1e-16)
        precision_curve = tpc / (tpc + fpc + 1e-16)
        # 插值
        r = np.linspace(0, 1, 101)
        p = np.interp(r, recall_curve, precision_curve, left=0, right=0)
        ap_c = np.trapz(p, r)
        f1_c = 2 * p * r / (p + r + 1e-16)
        ap.append(ap_c)
        precision.append(p[50])  # recall=0.5处的precision
        recall.append(r[50])
        f1.append(f1_c[50])
        pr_curve.append(np.stack([r, p], axis=1))
    ap = np.array(ap)
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.array(f1)
    pr_curve = np.stack(pr_curve, axis=0)
    return precision, recall, ap, f1, unique_classes, pr_curve
