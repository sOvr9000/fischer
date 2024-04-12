




def weighted_division(items: list[float], total: float) -> list[float]:
    '''
    Divide `total` by the number of items in `items`, returning each quotient that adds up to `total` based on the weights of each item in `items`.
    '''
    v = total / sum(items)
    return [
        v * i
        for i in items
    ]


if __name__ == '__main__':
    wd = weighted_division(
        items=[
            6 / 50,
            2.2 / 50,
            1.5 / 100,
            .744 / 50,
            .528 / 50,
        ],
        total=77,
    )
    print(wd)
    wdr = [int(.5 + q) for q in wd]
    print(wdr)
    print(sum(wdr))




