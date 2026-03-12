#!/usr/bin/env python3
"""
CATLASS JIT BO Tuner V2 - 按照论文方式实现
使用GPyOpt内置约束机制，避免污染GP模型

参考: High Performance OpenCL-Based GEMM Kernel Auto-Tuned by Bayesian Optimization
源码: https://github.com/lsl036/CL-DB-GEMM
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import GPy
    import GPyOpt
    from GPyOpt.methods import BayesianOptimization
except ImportError:
    print("请安装GPyOpt: pip install GPyOpt GPy")
    sys.exit(1)

from jit_runner import CATLASSJITRunner


class JITBOTunerV2:
    """
    按照论文方式实现的Bayesian Optimization调参器
    
    关键改进:
    1. 使用GPyOpt的constraints参数，在采样时过滤不可行配置
    2. 添加acquisition_jitter提高探索
    3. 使用exact_feval=True（假设无噪声）
    4. 正确处理编译失败（不污染GP）
    """
    
    def __init__(self, matrix_size: int = 4096, max_iter: int = 50, seed: int = None):
        self.matrix_size = matrix_size
        self.max_iter = max_iter
        self.seed = seed
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            print(f"随机种子: {seed}")
        self.runner = CATLASSJITRunner()
        
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = self.base_dir / "results"
        self.log_dir = self.base_dir / "log"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 搜索历史
        self.history: List[Dict] = []
        self.best_config = None
        self.best_gflops = 0.0
        self.failed_count = 0
        
        # ========== 搜索空间定义 ==========
        # 使用discrete类型（与论文一致）
        self.space = [
            # Tile维度
            {'name': 'tile_m', 'type': 'discrete', 'domain': (64, 128, 256)},      # 0
            {'name': 'tile_n', 'type': 'discrete', 'domain': (64, 128, 256)},      # 1
            # K维度
            {'name': 'l1_k', 'type': 'discrete', 'domain': (64, 128, 256, 512)},   # 2
            {'name': 'l0_k', 'type': 'discrete', 'domain': (16, 32, 64, 128, 256)},# 3
            # GemmAtlasA2 flags
            {'name': 'enable_unit', 'type': 'discrete', 'domain': (0, 1)},          # 4
            {'name': 'enable_shuffle_k', 'type': 'discrete', 'domain': (0, 1)},     # 5
            {'name': 'enable_abba', 'type': 'discrete', 'domain': (0, 1)},          # 6
            # Swizzle
            {'name': 'swizzle_offset', 'type': 'discrete', 'domain': (1, 2, 3, 4)}, # 7
            {'name': 'swizzle_direction', 'type': 'discrete', 'domain': (0, 1)},    # 8
        ]
        
        # ========== 约束定义 (论文格式) ==========
        # 约束格式: 'expression - 0.1' 
        # 当expression <= 0时，约束满足
        # 使用-0.1是为了让等于0的情况也满足约束
        self.constraints = [
            # 约束1: L1_K必须能被L0_K整除
            # x[:,2] % x[:,3] == 0  =>  x[:,2] % x[:,3] - 0.1 <= 0
            {'name': 'l1k_divisible_by_l0k', 
             'constraint': 'x[:,2] % x[:,3] - 0.1'},
            
            # 约束2: 256x256超过L0C空间
            # NOT (tile_m == 256 AND tile_n == 256)
            # 实现: (x[:,0] == 256) * (x[:,1] == 256) - 0.5 <= 0
            # 当tile_m=256且tile_n=256时，结果=0.5>0，不满足
            {'name': 'not_256x256',
             'constraint': '(x[:,0] == 256) * (x[:,1] == 256) - 0.5'},
            
            # 约束3: L0_K不能大于L1_K
            # x[:,3] <= x[:,2]  =>  x[:,3] - x[:,2] - 0.1 <= 0
            {'name': 'l0k_not_greater_than_l1k',
             'constraint': 'x[:,3] - x[:,2] - 0.1'},
            
            # 约束4: 对于小tile(64x64)，L1_K不应过大（可能超L1空间）
            # 当tile=64x64时，L1_K应<=256
            # (x[:,0] == 64) * (x[:,1] == 64) * (x[:,2] > 256) - 0.5 <= 0
            {'name': 'small_tile_k_limit',
             'constraint': '(x[:,0] == 64) * (x[:,1] == 64) * (x[:,2] > 256) - 0.5'},
        ]
    
    def _decode_params(self, x: np.ndarray) -> Dict:
        """将BO参数解码为配置字典"""
        return {
            'tile_m': int(x[0]),
            'tile_n': int(x[1]),
            'l1_k': int(x[2]),
            'l0_k': int(x[3]),
            'enable_unit': int(x[4]),
            'enable_shuffle_k': int(x[5]),
            'enable_abba': int(x[6]),
            'swizzle_offset': int(x[7]),
            'swizzle_direction': int(x[8]),
        }
    
    def _objective(self, x: np.ndarray) -> float:
        """
        BO目标函数
        
        返回GFLOPS（最大化）
        编译失败时返回0（而不是惩罚值）
        """
        config = self._decode_params(x[0])
        
        # 运行kernel
        m = n = k = self.matrix_size
        exec_time, gflops, error_msg = self.runner.run(m, n, k, config, warmup=3, repeat=10)
        
        # 处理编译/执行失败
        if gflops == 0 or exec_time > 1e9:
            self.failed_count += 1
            # 论文做法: 返回0或很小的值，而不是极大惩罚
            # 这样不会过度影响GP对附近区域的预测
            gflops = 0.0
        
        # 记录历史
        tflops = gflops / 1000
        efficiency = tflops / 352 * 100 if tflops > 0 else 0
        
        self.history.append({
            'config': config,
            'time_ms': exec_time if exec_time < 1e9 else -1,
            'gflops': gflops,
            'tflops': tflops,
            'efficiency': efficiency,
            'success': gflops > 0,
        })
        
        # 更新最优
        if gflops > self.best_gflops:
            self.best_config = config.copy()
            self.best_gflops = gflops
        
        # 打印进度
        idx = len(self.history)
        tile_str = f"Tile[{config['tile_m']},{config['tile_n']}]"
        k_str = f"K[L1={config['l1_k']},L0={config['l0_k']}]"
        flag_str = f"U{config['enable_unit']}S{config['enable_shuffle_k']}A{config['enable_abba']}"
        sw_str = f"Sw{config['swizzle_offset']},{config['swizzle_direction']}"
        
        if gflops > 0:
            print(f"[{idx:3d}] {tile_str} {k_str} {flag_str} {sw_str} -> "
                  f"{exec_time:.2f}ms, {tflops:.1f}TFLOPS ({efficiency:.1f}%)")
        else:
            print(f"[{idx:3d}] {tile_str} {k_str} {flag_str} {sw_str} -> FAILED")
        
        # 返回GFLOPS（GPyOpt会最大化）
        return gflops
    
    def run(self) -> Dict:
        """运行BO优化"""
        print("=" * 80)
        print(f"CATLASS JIT BO优化 V2 (按论文方式实现)")
        print(f"矩阵大小: {self.matrix_size}x{self.matrix_size}")
        print(f"最大迭代: {self.max_iter}")
        print(f"约束数量: {len(self.constraints)}")
        print("=" * 80)
        print("\n约束列表:")
        for c in self.constraints:
            print(f"  - {c['name']}: {c['constraint']}")
        print("=" * 80)
        
        # 创建BO优化器（按论文方式）
        optimizer = BayesianOptimization(
            f=self._objective,
            domain=self.space,
            constraints=self.constraints,     # ✅ 关键：使用GPyOpt约束
            model_type='GP',
            acquisition_type='EI',
            acquisition_jitter=0.05,          # ✅ 论文使用
            exact_feval=True,                 # ✅ 论文使用
            maximize=True,                    # 最大化GFLOPS
            initial_design_numdata=min(20, self.max_iter),
            verbosity=False,
        )
        
        # 运行优化
        optimizer.run_optimization(max_iter=max(0, self.max_iter - 20))
        
        # 结果
        success_count = sum(1 for h in self.history if h['success'])
        
        print("\n" + "=" * 80)
        print(f"优化完成!")
        print(f"  总评估: {len(self.history)}")
        print(f"  成功: {success_count}")
        print(f"  失败: {self.failed_count}")
        print()
        
        if self.best_config:
            best_tflops = self.best_gflops / 1000
            best_eff = best_tflops / 352 * 100
            
            print("最优参数:")
            print(f"  TileShape (M,N): [{self.best_config['tile_m']}, {self.best_config['tile_n']}]")
            print(f"  K (L1/L0): [{self.best_config['l1_k']}, {self.best_config['l0_k']}]")
            print(f"  GemmAtlasA2: unit={self.best_config['enable_unit']}, "
                  f"shuffle_k={self.best_config['enable_shuffle_k']}, "
                  f"abba={self.best_config['enable_abba']}")
            print(f"  Swizzle: offset={self.best_config['swizzle_offset']}, "
                  f"direction={self.best_config['swizzle_direction']}")
            print()
            print(f"最优性能: {best_tflops:.2f} TFLOPS ({best_eff:.1f}% 效率)")
        
        print("=" * 80)
        
        # 保存结果
        self._save_results()
        
        return {
            'config': self.best_config,
            'gflops': self.best_gflops,
        }
    
    def _save_results(self):
        """保存优化结果"""
        timestamp = datetime.now().isoformat()
        
        if self.best_config:
            best_tflops = self.best_gflops / 1000
            
            # 保存最优配置
            best_config_file = self.results_dir / "jit_best_config_v2.json"
            with open(best_config_file, 'w') as f:
                json.dump({
                    'matrix_size': self.matrix_size,
                    'best_params': self.best_config,
                    'best_gflops': self.best_gflops,
                    'best_tflops': best_tflops,
                    'efficiency_percent': best_tflops / 352 * 100,
                    'total_evaluations': len(self.history),
                    'successful_evaluations': sum(1 for h in self.history if h['success']),
                    'timestamp': timestamp,
                }, f, indent=2)
            print(f"最优配置已保存: {best_config_file}")
        
        # 保存历史
        history_file = self.log_dir / "jit_optimization_history_v2.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"优化历史已保存: {history_file}")
        
        # 保存收敛曲线
        convergence_file = self.results_dir / "jit_convergence_v2.json"
        best_so_far = []
        current_best = 0
        for h in self.history:
            if h['gflops'] > current_best:
                current_best = h['gflops']
            best_so_far.append(current_best / 1000)  # TFLOPS
        
        with open(convergence_file, 'w') as f:
            json.dump({
                'iterations': list(range(1, len(best_so_far) + 1)),
                'best_tflops': best_so_far,
                'constraints_used': [c['name'] for c in self.constraints],
            }, f, indent=2)
        print(f"收敛曲线已保存: {convergence_file}")


def main():
    parser = argparse.ArgumentParser(description='CATLASS JIT BO Tuner V2 (Paper-based)')
    parser.add_argument('--matrix_size', type=int, default=4096,
                        help='矩阵大小 (默认: 4096)')
    parser.add_argument('--max_iter', type=int, default=50,
                        help='最大迭代次数 (默认: 50)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子 (用于可重复实验)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CATLASS JIT BO Tuner V2")
    print("基于论文: High Performance OpenCL-Based GEMM Kernel Auto-Tuned by BO")
    print("=" * 80)
    
    tuner = JITBOTunerV2(
        matrix_size=args.matrix_size,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    
    tuner.run()


if __name__ == "__main__":
    main()
