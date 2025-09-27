def calc_sum(a: int, b: int, c: int) -> int:
    return a + b + c

if __name__ == "__main__":
    x, y, z = map(int, input().split())
    print(calc_sum(x, y, z))