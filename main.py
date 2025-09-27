def calc_sum(*args: int) -> int:
    return sum(args)

if __name__ == "__main__":
    nums = map(int, input().split())
    print(calc_sum(*nums))
