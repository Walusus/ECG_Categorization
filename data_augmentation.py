from scipy.signal import resample
import numpy as np


def stretch_hor(vec, intensity):
    """
    Applies horizontal stretching to a copy of the given vector.
    :returns: Modified copy of given vector.
    """
    low = int(len(vec) - len(vec) * intensity)
    high = int(len(vec) + len(vec) * intensity)
    res_vec = resample(vec, np.random.randint(low, high))

    if len(res_vec) < len(vec):
        result = np.zeros(len(vec))
        result[:len(res_vec)] = res_vec
    else:
        result = res_vec[:len(vec)]

    return result


def stretch_ver(vec, intensity):
    """
    Applies vertical stretch to a copy of the given vector.
    :returns: Modified copy of given vector.
    """
    result = vec + vec * intensity * (.5 - np.random.rand())
    return result


def modify_vector(vec, intensity):
    """
    Applies amplification and/or stretching to a copy of the given vector.
    :returns: Modified copy of given vector.
    """
    mod_type = np.random.randint(0, 3)
    if mod_type is 0:
        return stretch_ver(vec, intensity)
    elif mod_type is 1:
        return stretch_hor(vec, intensity)
    else:
        return stretch_ver(stretch_hor(vec, intensity), intensity)
