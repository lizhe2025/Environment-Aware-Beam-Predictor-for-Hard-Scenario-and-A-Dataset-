#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Inspector — 检查 6G 波束预测数据集完整性
"""

import os
import sys
import json
import numpy as np
from collections import Counter


def inspect_dataset(data_dir='./dataset'):
    print("=" * 70)
    print(f"数据集目录: {os.path.abspath(data_dir)}")
    print("=" * 70)

    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return False

    files = os.listdir(data_dir)
    npz_files = [f for f in files if f.endswith('.npz')]

    print(f"\n📁 发现 {len(npz_files)} 个 NPZ 文件:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(data_dir, f))
        size_mb = size / (1024 * 1024)
        print(f"   {f:35s}  {size_mb:10.2f} MB")

    # meta.json
    meta_path = os.path.join(data_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"\n📋 meta.json:")
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    else:
        print("\n⚠️ meta.json 不存在")

    print("\n" + "=" * 70)
    print("NPZ 文件详细分析")
    print("=" * 70)

    for npz_name in sorted(npz_files):
        path = os.path.join(data_dir, npz_name)
        print(f"\n🔍 {npz_name}")
        print("-" * 50)

        try:
            data = np.load(path, allow_pickle=False)
            print(f"   字段数: {len(data.files)}")

            # 样本数
            n_samples = None
            for key in ['position', 'sector', 'los']:
                if key in data:
                    n_samples = data[key].shape[0]
                    break
            print(f"   样本数: {n_samples}")

            if n_samples is None:
                print("   ⚠️ 无法确定样本数")
                data.close()
                continue

            # sector 分布
            if 'sector' in data:
                sectors = data['sector']
                unique, counts = np.unique(sectors, return_counts=True)
                print(f"   Sector 分布:")
                for u, c in zip(unique, counts):
                    pct = c / len(sectors) * 100
                    print(f"      Sector {int(u):2d}: {c:7d}  ({pct:6.2f}%)")

            # quadrant 分布
            if 'quadrant_idx' in data:
                quads = data['quadrant_idx']
                unique, counts = np.unique(quads, return_counts=True)
                print(f"   Quadrant 分布:")
                for u, c in zip(unique, counts):
                    pct = c / len(quads) * 100
                    print(f"      Quadrant {int(u)}: {c:7d}  ({pct:6.2f}%)")

            # LoS 分布
            if 'los' in data:
                los = data['los']
                true_count = int(los.sum())
                false_count = len(los) - true_count
                print(f"   LoS: True={true_count:6d} ({true_count/len(los)*100:5.1f}%)  "
                      f"False={false_count:6d} ({false_count/len(los)*100:5.1f}%)")

            # channel_valid
            if 'channel_valid' in data:
                cv = data['channel_valid']
                valid = int(cv.sum())
                print(f"   Channel Valid: {valid:6d}/{len(cv):6d} ({valid/len(cv)*100:5.1f}%)")

            # RSRP 字段
            rsrp_keys = [k for k in data.files if k.startswith('rsrp_')]
            print(f"   RSRP 字段: {len(rsrp_keys)} 个")
            for k in sorted(rsrp_keys):
                arr = data[k]
                print(f"      {k:25s} shape={str(arr.shape):20s}  "
                      f"range=[{arr.min():.2e}, {arr.max():.2e}]")

            # best 字段
            best_keys = [k for k in data.files if k.startswith('best_')]
            print(f"   Best 字段: {len(best_keys)} 个")
            for k in sorted(best_keys):
                arr = data[k]
                # 检查是否有异常值
                invalid = np.sum((arr < 0) | (arr >= 256))
                print(f"      {k:20s} shape={str(arr.shape):15s}  "
                      f"range=[{arr.min()}, {arr.max()}]  "
                      f"invalid={invalid}")

            # 位置范围
            if 'position' in data:
                pos = data['position']
                print(f"   位置范围:")
                print(f"      X: [{pos[:,0].min():8.2f}, {pos[:,0].max():8.2f}]")
                print(f"      Y: [{pos[:,1].min():8.2f}, {pos[:,1].max():8.2f}]")
                print(f"      Z: [{pos[:,2].min():8.2f}, {pos[:,2].max():8.2f}]")

            # stratified_group
            if 'stratified_group' in data:
                sg = data['stratified_group']
                unique_sg = np.unique(sg)
                print(f"   Stratified Group: {len(unique_sg)} 个唯一值")
                # 显示前 10 个
                for g in unique_sg[:10]:
                    count = int((sg == g).sum())
                    print(f"      '{g}': {count:6d} 样本")
                if len(unique_sg) > 10:
                    print(f"      ... 还有 {len(unique_sg)-10} 个")

            # h_narrow 检查
            if 'h_narrow' in data:
                h = data['h_narrow']
                print(f"   h_narrow: shape={h.shape}, dtype={h.dtype}")
                # 检查是否全零
                zero_count = np.sum(np.abs(h).max(axis=(1,2)) < 1e-12)
                print(f"      全零信道: {zero_count}/{len(h)} ({zero_count/len(h)*100:.1f}%)")

            data.close()

        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("检查完成")
    print("=" * 70)
    return True


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './dataset'
    inspect_dataset(data_dir)
