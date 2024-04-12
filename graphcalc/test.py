



def clamp0(v, v_min, v_max):
    if v < v_min:
        return v_min
    if v > v_max:
        return v_max
    return v

def clamp1(v, v_min, v_max):
    b0 = int(v > v_min)
    b1 = int(v < v_max)
    return v * b0 * b1 + v_min * (1 - b0) + v_max * (1 - b1)


from timeit import timeit

print(timeit('clamp1(0.5, 0, 1)', setup='from test import clamp1', number=10000000))

