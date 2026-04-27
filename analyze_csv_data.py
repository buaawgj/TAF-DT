#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd

DEFAULT_CSV_PATH = "/home/wgj/Downloads"


def main(csv_name: str):
    # 读取 csv 文件
    csv_path = os.path.join(DEFAULT_CSV_PATH, csv_name)
    df = pd.read_csv(csv_path)

    # 找到 value 列（忽略大小写匹配 "value"）
    value_col = None
    for col in df.columns:
        if col.lower() == "value":
            value_col = col
            break
    if value_col is None:
        raise ValueError("没有找到名为 'value'（大小写不敏感）的列，请检查文件。")

    # 第一列用于时间（本文件是 'Wall time'）
    time_col = df.columns[0]

    values = df[value_col].astype(float)
    times = df[time_col].astype(float)

    # 1. 整体 Value 最大值
    overall_max = values.max()
    print(f"整体 {value_col} 最大值: {overall_max:.6f}")

    # 2. 最后 3 / 5 / 10 行的统计量
    for k in (10, 5, 3):
        if len(values) >= k:
            tail_vals = values.tail(k)
        else:
            # 如果总行数少于 k，就用全部数据
            tail_vals = values

        max_k = tail_vals.max()
        mean_k = tail_vals.mean()
        # 方差这里采用总体方差（ddof=0），如果你想要样本方差改为 ddof=1
        var_k = tail_vals.std(ddof=1)

        print(
            f"最后 {k} 行（实际使用 {len(tail_vals)} 行）{value_col}："
            f"最大值 = {max_k:.6f}，平均值 = {mean_k:.6f}，标准差 = {var_k:.6f}"
        )

    # 3. 通过第一列时间估算运行时长（单位：小时）
    # 假设第一列是 Unix 时间戳（秒）
    duration_seconds = float(times.iloc[-1]) - float(times.iloc[0])
    duration_hours = duration_seconds / 3600.0
    
    # 4. Value 最后一个值
    final_val = values.iloc[-1]
    print(f"{value_col} 最后值: {final_val:.6f}")

    print(
        f"\n根据第一列时间估算的程序运行时间：约 {duration_hours:.3f} 小时 "
        f"（{duration_seconds:.1f} 秒）"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法：python {sys.argv[0]} your_file.csv")
        sys.exit(1)

    main(sys.argv[1])
