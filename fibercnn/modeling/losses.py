import torch


def euclidean_distance(pt1, pt2):
    return torch.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def _c(ca, i, j, P, Q, device):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = euclidean_distance(P[0], Q[0])
    elif i > 0 and j == 0:
        ca[i, j] = torch.tensor(
            [_c(ca, i - 1, 0, P, Q, device), euclidean_distance(P[i], Q[0])],
            device=device,
            requires_grad=True,
        ).max()
    elif i == 0 and j > 0:
        ca[i, j] = torch.tensor(
            [_c(ca, 0, j - 1, P, Q, device), euclidean_distance(P[0], Q[j])],
            device=device,
            requires_grad=True,
        ).max()
    elif i > 0 and j > 0:
        ca[i, j] = torch.tensor(
            [
                torch.tensor(
                    [
                        _c(ca, i - 1, j, P, Q, device),
                        _c(ca, i - 1, j - 1, P, Q, device),
                        _c(ca, i, j - 1, P, Q, device),
                    ],
                    device=device,
                    requires_grad=True,
                ).min(),
                euclidean_distance(P[i], Q[j]),
            ],
            device=device,
            requires_grad=True,
        ).max()
    else:
        ca[i, j] = torch.tensor([float("inf")], device=device, requires_grad=True)
    return ca[i, j]


def frechet_distance(P, Q, device=None):
    """Computes the discrete frechet distance between two polygonal lines.
    Algorithm:      http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
    Original Code:  https://gist.github.com/MaxBareiss/ba2f9441d9455b56fbc9

    P and Q are arrays of 2-element arrays (points)
    """
    if device is None:
        device = P.device

    ca = torch.ones((len(P), len(Q)), device=device, requires_grad=True)
    ca = -ca
    return _c(ca, len(P) - 1, len(Q) - 1, P, Q, device)
