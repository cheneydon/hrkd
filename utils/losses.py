import torch.nn.functional as F


def soft_cross_entropy(predicts, targets, t=1, mean=True):
    student_prob = F.log_softmax(predicts / t, dim=-1)
    teacher_prob = F.softmax(targets / t, dim=-1)
    out = -teacher_prob * student_prob
    if mean:
        out = out.mean()
    return out
