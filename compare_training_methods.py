#!/usr/bin/env python
"""
分段训练 vs 全局训练对比脚本
可视化两种方法的性能差异
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 性能数据
segments = ['0-90', '90-180', '180-270', '270-365', '365-545', '545-730']

# 验证集 MAE
val_mae_global = [419.43, 30.43, 71.57, 121.44, 207.30, 427.20]
val_mae_segmented = [19.05, 18.20, 44.12, 32.41, 64.68, 54.67]

# 训练集 MAE（最终 epoch）