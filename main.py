def calc_sum(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    x, y = map(int, input().split())
    print(calc_sum(x, y))