import numpy as np
import copy

# def norm_sub(estimates: list, n: int):
#     if not np.any(estimates):
#         # if all zeros, then assign average to all
#         return np.ones(len(estimates)) * (n / len(estimates))
#     estimates = copy.deepcopy(estimates)
#     while (np.fabs(sum(estimates) - n) > 1) or (estimates < 0).any():
#         estimates[estimates < 0] = 0
#         total = sum(estimates)
#         mask = estimates > 0
#         diff = (n - total) / sum(mask)
#         estimates[mask] += diff
#     return estimates


def norm_sub(estimates: np.array, n: int):
    if not np.any(estimates):
        # if all zeros, then assign average to all
        return np.ones(len(estimates)) * (n / len(estimates))
    print(f"before normsub- min:{np.min(estimates)}, max: {np.max(estimates)}, sum: {np.sum(estimates)}")
    adjusted_estimates = copy.deepcopy(estimates)
    sorted_estimates = -np.sort(-estimates)
    cumsum = np.cumsum(sorted_estimates)
    range = np.arange(start=1, stop=len(estimates)+1)
    diffs = sorted_estimates - (cumsum - n) / range
    rho = np.max(np.argwhere(diffs > 0))
    theta = (cumsum[rho] - n) / (rho + 1)
    adjusted_estimates -= theta
    adjusted_estimates[adjusted_estimates < 0] = 0
    # print(sorted_estimates)
    # print(cumsum)
    # print(diffs)
    # print(rho)
    # print(f"after normsub- min:{np.min(adjusted_estimates)}, max: {np.max(adjusted_estimates)}, sum: {np.sum(adjusted_estimates)}")
    # exit()
    return adjusted_estimates
