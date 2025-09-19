from bisect import bisect_left

import numpy as np


def local_k_choose(max_k: int, n: int, number_sketches: int, epsilon: float, delta: float) -> int:
    k_range = np.arange(start=2, stop=max_k)
    for i, k in enumerate(k_range):
        # std = 0.649 / np.sqrt(number_sketches) * n / k * 2 * (k - 1) + 4 * (k - 1) * np.log(1 / delta) / epsilon
        std = 0.649 / np.sqrt(number_sketches) * n / (k**2) * (k**2 - 1) + 4 * 0.649 * (k - 1) * np.log(1 / delta) / epsilon
        avg_count = n / (k ** 2)
        if std * 2 > avg_count:
            break
    return k


if __name__ == '__main__':
    n = 20000
    S = 4
    epss = np.array([0.5, 1, 2, 3, 4, 5])
    for eps in epss:
        # print(f"==== epsilon: {eps}")
        k = local_k_choose(10, n=n, number_sketches=4096, epsilon=eps / (S * 2), delta=1/n)
        # print(f"choose k:{k}")