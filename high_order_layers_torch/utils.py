import torch


def make_periodic(x, periodicity: float):
    xp = x+0.5*periodicity
    xp = torch.remainder(xp, 2*periodicity)  # always positive
    xp = torch.where(xp > periodicity, 2*periodicity-xp, xp)
    xp = xp - 0.5*periodicity
    return xp
