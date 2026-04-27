#!/usr/bin/env python3
import sys
import math

def main():
    # 读取stdin全部内容；支持：echo "1 2 3" | python stats.py
    # 也支持：交互粘贴多行后 Ctrl+D(Unix/macOS) / Ctrl+Z 回车(Windows) 结束输入
    text = sys.stdin.read().strip()
    if not text:
        print("No numbers provided on stdin.")
        sys.exit(1)

    try:
        xs = [float(t) for t in text.split()]
    except ValueError as e:
        print(f"Failed to parse numbers: {e}")
        sys.exit(1)

    n = len(xs)
    mean = sum(xs) / n

    # 总体标准差（ddof=0）
    var_pop = sum((x - mean) ** 2 for x in xs) / n
    std_pop = math.sqrt(var_pop)

    # 样本标准差（ddof=1），n=1 时无定义
    if n >= 2:
        var_samp = sum((x - mean) ** 2 for x in xs) / (n - 1)
        std_samp = math.sqrt(var_samp)
    else:
        std_samp = float("nan")

    print(f"count: {n}")
    print(f"mean:  {mean}")
    print(f"std (sample, ddof=1):     {std_samp}")
    print(f"std (population, ddof=0): {std_pop}")

if __name__ == "__main__":
    main()
