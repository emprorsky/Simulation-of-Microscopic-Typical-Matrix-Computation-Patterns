from utils import size, Tensor, DataType
from typing import List, Tuple, Dict
from operators import Operator
from hardware import HardwareSpec, HW # 全局硬件实例
from modules import device as _device, L2_CACHE_MGR
from math import ceil, log2, floor
import time
import statistics
import numpy as np
import pandas as pd
import os
import copy

# 读取OUT2L1_efficiency.csv并创建排序后的效率列表（traffic_size(KB)为键，efficiency为值）
def load_efficiency_dict(csv_path):
    # 使用pandas读取CSV文件
    df = pd.read_csv(csv_path)
    # 确保列名正确，选择需要的列并转换为float类型
    eff_df = df[['traffic_size(KB)', 'efficiency']].astype(float)
    # 按traffic_size(KB)从小到大排序
    eff_df_sorted = eff_df.sort_values(by='traffic_size(KB)')
    # 转换为列表形式便于后续查找（[(size1, eff1), (size2, eff2), ...]）
    return list(zip(eff_df_sorted['traffic_size(KB)'], eff_df_sorted['efficiency']))

class Matmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None
        '''---'''
        self.OUT2L1_eff_list = load_efficiency_dict('OUT2L1_efficiency.csv')
        self.L12L0A_eff_list = load_efficiency_dict('l12L0A_efficiency.csv')
        self.L12L0B_eff_list = load_efficiency_dict('l12L0B_efficiency.csv')
        self.OUT2L1_eff_roofline_list = load_efficiency_dict('OUT2L1_efficiency - roofline.csv')
        

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(self.input1_shape[:-1])
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.K, self.data_type
        )
        self.flop_count = 2 * self.M * self.K * self.N
        self.io_count = (self.M * self.K + self.K * self.N + self.M * self.N)*self.data_type.word_size
        # print(f'{self.M}, {self.N}, {self.K}')
        return output
    

    def roofline_model(self, chip_type:'HardwareSpec' = HW):
        io_bw   = chip_type.IO_BW
        macs_pc = chip_type.CUBE_MACS_PER_CORE  # 单核心每周期MAC数（4096）
        core_count = chip_type.AI_CORE_COUNT    # 核心数（24）
        clock_freq = chip_type.CLOCK_FREQ # 主频（1.85e9HZ）
        cube_latency=self.flop_count / (chip_type.CUBE_UTILIZATION*macs_pc*2*core_count*clock_freq)
        '''---'''
        # 转换为KB
        traffic_size_kb = self.io_count / 1024
        # 查找对应的效率值（大于等于当前traffic_size_kb的最小key对应的效率）
        OUT2L1_efficiency = 0.2  # 默认值
        for size, eff in self.OUT2L1_eff_list:
            if traffic_size_kb >= size:
                OUT2L1_efficiency = eff
            else:
                break  # 已排序，后续值更大，直接跳出
        
        io_latency=self.io_count/ (chip_type.IO_BW['MEM1']*OUT2L1_efficiency*clock_freq) #io带宽 #float16所以乘2  self.data_type.word_size！！！
        '''---'''
       
        self.roofline_latency = max(
            # 计算计算受限延迟：总FLOPs / 硬件每周期FLOPs能力 ！！！
            cube_latency,
            # 计算有效内存带宽
            io_latency
        )
        return self.flop_count, chip_type.CUBE_UTILIZATION*macs_pc*2*core_count*clock_freq, cube_latency, io_latency, self.roofline_latency

    def print_latency(self):
        print(
            f"{self.computational_graph.M}, {self.computational_graph.N}, {self.computational_graph.K}, {self.best_latency*1e3:.4f}ms, {self.latency_on_gpu*1e3:.4f}ms, {self.best_latency/self.latency_on_gpu*100:.2f}%",
            flush=True,
        )

    @staticmethod
    def generate_tile_loops(loop_M: int, loop_N: int, loop_K: int, loop_order: str):
        assert loop_order in ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
        if loop_order == "mnk":
            for m in range(loop_M):
                for n in range(loop_N):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "mkn":
            for m in range(loop_M):
                for k in range(loop_K):
                    for n in range(loop_N):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(loop_N):
                for m in range(loop_M):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "nkm":
            for n in range(loop_N):
                for k in range(loop_K):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "knm":
            for k in range(loop_K):
                for n in range(loop_N):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "kmn":
            for k in range(loop_K):
                for m in range(loop_M):
                    for n in range(loop_N):
                        yield m, n, k

    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, word_size(B): {self.data_type.word_size}"
            )

    #暂时先省去L0分块
    class Mapping:
        def __init__(
            self,
            l1_tile_M: int,
            l1_tile_N: int,
            l1_tile_K: int,
            l1_loop_order: str,
        ):
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N
            self.l1_tile_K = l1_tile_K
            self.l1_loop_order = l1_loop_order

        def display(self):
            print(f'{"-"*10} Mapping {"-"*10}')
            print(
                f"l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, l1_tile_K: {self.l1_tile_K}, l1_loop_order: {self.l1_loop_order}"
            )

    @staticmethod
    def find_permutations(n):
        permutations = set()

        for i in range(1, n + 1):
            if n % i == 0:
                for j in range(1, n + 1):
                    if (n // i) % j == 0:
                        k = n // (i * j)
                        permutations.add((i, j, k))

        return list(permutations)
    
    # ================================================================
    # [修改] compile_and_simulate 重构为三模式：
    #   fast      : pow2 候选 [32,64,128,256,512]，快速穷举（来自 new_matmul.py）
    #   exhaustive: 16 的倍数候选（上限=dim 向下取整到 16 的倍数），全量穷举
    #   bayes     : 同 exhaustive 候选空间，贝叶斯优化（scikit-optimize）
    # 新增参数 n_calls：bayes 模式的最大评估次数（默认 80）
    # ================================================================
    def compile_and_simulate(
            self,
            chip_type:'HardwareSpec' = HW,
            compile_mode: str = "exhaustive",
            n_calls: int = 80,          # [新增] bayes 模式专用参数
        ):
            min_cycle_count = 2**63 - 1
            best_mapping = None
            M = self.computational_graph.M
            N = self.computational_graph.N
            K = self.computational_graph.K
            io_bw   = chip_type.IO_BW
            macs_pc = chip_type.CUBE_MACS_PER_CORE
            core_count = chip_type.AI_CORE_COUNT
            clock_freq = chip_type.CLOCK_FREQ

            # 选择候选 tile 的辅助：满足 L1 容量且占用率 >= 60%
            L1_elems_cap = chip_type.L1_CAPACITY // self.data_type.word_size // 2
            def ok_tile(m, n, k):
                elems = m*n + n*k + m*k
                if elems > L1_elems_cap:
                    return False
                occ = elems / max(1, L1_elems_cap)
                return occ >= 0.60

            loop_orders = ["mkn", "nkm", "mnk", "nmk", "knm", "kmn"]

            # ------------------------------------------------------------------
            # 所有模式共用：M==1 或 N==1 时直接走 roofline，不做分块搜索
            # ------------------------------------------------------------------
            if (M == 1 or N == 1):
                total_io_count = (M * K + N * K + M * N) * self.data_type.word_size
                traffic_size_kb = total_io_count / 1024
                OUT2L1_efficiency_roofline = 0.2
                for size, eff in self.OUT2L1_eff_roofline_list:
                    if traffic_size_kb >= size:
                        OUT2L1_efficiency_roofline = eff
                    else:
                        break
                io_latency = total_io_count / (chip_type.IO_BW['MEM1'] * clock_freq * OUT2L1_efficiency_roofline)
                total_flop_count = 2 * M * N * K
                compute_latency = total_flop_count / (chip_type.CUBE_UTILIZATION * macs_pc * 2 * core_count * clock_freq)
                self.latency = max(compute_latency, io_latency)
                return self.latency

            # ==================================================================
            # 模式一：fast
            # 候选来自 pow2 = [32, 64, 128, 256, 512]（来自 new_matmul.py）
            # 全量穷举，速度最快
            # ==================================================================
            if compile_mode == "fast":
                pow2 = [1 << i for i in range(5, 10)]  # [32, 64, 128, 256, 512]
                def sort_candidates_fast(dim):
                    cands = [x for x in pow2 if x <= max(32, dim)]
                    cands.sort(key=lambda x: (dim % x != 0, -x))
                    return cands
                Ms = sort_candidates_fast(M)
                Ns = sort_candidates_fast(N)
                Ks = sort_candidates_fast(K)

                for l1_tile_M in Ms:
                    for l1_tile_N in Ns:
                        for l1_tile_K in Ks:
                            if not ok_tile(l1_tile_M, l1_tile_N, l1_tile_K):
                                continue
                            for l1_loop_order in loop_orders:
                                mapping = self.Mapping(l1_tile_M, l1_tile_N, l1_tile_K, l1_loop_order)
                                cycles = self.simulate(self.computational_graph, mapping, chip_type)
                                if cycles < min_cycle_count:
                                    min_cycle_count = cycles
                                    best_mapping = mapping

                # 兜底：放宽占用率约束，只要不超容量就行
                if best_mapping is None:
                    for l1_tile_M in Ms:
                        for l1_tile_N in Ns:
                            for l1_tile_K in Ks:
                                elems = l1_tile_M*l1_tile_N + l1_tile_N*l1_tile_K + l1_tile_M*l1_tile_K
                                if elems > L1_elems_cap:
                                    continue
                                for l1_loop_order in loop_orders:
                                    mapping = self.Mapping(l1_tile_M, l1_tile_N, l1_tile_K, l1_loop_order)
                                    cycles = self.simulate(self.computational_graph, mapping, chip_type)
                                    if cycles < min_cycle_count:
                                        min_cycle_count = cycles
                                        best_mapping = mapping

            # ==================================================================
            # 模式二：exhaustive
            # 候选为 16 的倍数，上限 = floor(dim/16)*16（即不超过 dim 的最大 16 倍数）
            # 全量穷举，搜索空间最大，结果最精确
            # ==================================================================
            elif compile_mode == "exhaustive":
                def sort_candidates_mul16(dim):
                    max_tile = (dim // 16) * 16  # 不超过 dim 的最大 16 倍数
                    max_tile = max(16, max_tile)
                    cands = list(range(16, max_tile + 1, 16))
                    cands.sort(key=lambda x: (dim % x != 0, -x))
                    return cands
                Ms = sort_candidates_mul16(M)
                Ns = sort_candidates_mul16(N)
                Ks = sort_candidates_mul16(K)

                total_combinations = len(Ms) * len(Ns) * len(Ks) * len(loop_orders)
                print(f"[exhaustive] 候选空间: M×N×K×order = "
                      f"{len(Ms)}×{len(Ns)}×{len(Ks)}×{len(loop_orders)} = {total_combinations}", flush=True)

                for l1_tile_M in Ms:
                    for l1_tile_N in Ns:
                        for l1_tile_K in Ks:
                            if not ok_tile(l1_tile_M, l1_tile_N, l1_tile_K):
                                continue
                            for l1_loop_order in loop_orders:
                                mapping = self.Mapping(l1_tile_M, l1_tile_N, l1_tile_K, l1_loop_order)
                                cycles = self.simulate(self.computational_graph, mapping, chip_type)
                                if cycles < min_cycle_count:
                                    min_cycle_count = cycles
                                    best_mapping = mapping

                # 兜底：放宽占用率约束
                if best_mapping is None:
                    for l1_tile_M in Ms:
                        for l1_tile_N in Ns:
                            for l1_tile_K in Ks:
                                elems = l1_tile_M*l1_tile_N + l1_tile_N*l1_tile_K + l1_tile_M*l1_tile_K
                                if elems > L1_elems_cap:
                                    continue
                                for l1_loop_order in loop_orders:
                                    mapping = self.Mapping(l1_tile_M, l1_tile_N, l1_tile_K, l1_loop_order)
                                    cycles = self.simulate(self.computational_graph, mapping, chip_type)
                                    if cycles < min_cycle_count:
                                        min_cycle_count = cycles
                                        best_mapping = mapping

            # ==================================================================
            # 模式三：bayes
            # 候选为 16 的倍数，上限 = floor(dim/16)*16
            # 预过滤合法 tile，用 scikit-optimize 贝叶斯优化搜索
            # n_calls 由外部传入（默认 80）
            # ==================================================================
            elif compile_mode == "bayes":
                def sort_candidates_mul16(dim):
                    max_tile = (dim // 16) * 16
                    max_tile = max(16, max_tile)
                    cands = list(range(16, max_tile + 1, 16))
                    cands.sort(key=lambda x: (dim % x != 0, -x))
                    return cands
                Ms = sort_candidates_mul16(M)
                Ns = sort_candidates_mul16(N)
                Ks = sort_candidates_mul16(K)

                # 预过滤：只保留满足 ok_tile 的组合
                valid_tiles = [
                    (m, n, k)
                    for m in Ms for n in Ns for k in Ks
                    if ok_tile(m, n, k)
                ]
                print(f"[贝叶斯] 有效 tile 组合数: {len(valid_tiles)} / "
                      f"{len(Ms)*len(Ns)*len(Ks)}（总候选）", flush=True)

                if not valid_tiles:
                    # 没有合法 tile，兜底：放宽占用率
                    valid_tiles = [
                        (m, n, k)
                        for m in Ms for n in Ns for k in Ks
                        if (m*n + n*k + m*k) <= L1_elems_cap
                    ]
                    print(f"[贝叶斯] 放宽约束后有效 tile 组合数: {len(valid_tiles)}", flush=True)

                if not valid_tiles:
                    raise ValueError(f"[贝叶斯] M={M},N={N},K={K} 下找不到任何合法 tile，请检查 L1 容量设置")

                _call_counter = [0]

                def _bayes_objective(params):
                    _call_counter[0] += 1
                    tile_idx, lo_idx = params
                    l1_tile_M, l1_tile_N, l1_tile_K = valid_tiles[int(tile_idx)]
                    l1_loop_order = loop_orders[int(lo_idx)]
                    mapping = self.Mapping(l1_tile_M, l1_tile_N, l1_tile_K, l1_loop_order)
                    cycles = float(self.simulate(self.computational_graph, mapping, chip_type))
                    print(
                        f"[贝叶斯 #{_call_counter[0]:03d}] "
                        f"tile=({l1_tile_M:5d},{l1_tile_N:5d},{l1_tile_K:5d}) "
                        f"order={l1_loop_order}  cycles={cycles:.0f}",
                        flush=True,
                    )
                    return cycles

                try:
                    from skopt import gp_minimize
                    from skopt.space import Integer

                    space = [
                        Integer(0, len(valid_tiles) - 1, name='tile_idx'),
                        Integer(0, len(loop_orders) - 1,  name='lo_idx'),
                    ]
                    n_initial_points = min(15, len(valid_tiles))

                    result = gp_minimize(
                        _bayes_objective,
                        space,
                        n_calls=n_calls,
                        n_initial_points=n_initial_points,
                        random_state=42,
                        noise=1e-10,
                        acq_func="EI",
                    )

                    best_lat_us = int(result.fun) / clock_freq * 1e6
                    best_tile   = valid_tiles[int(result.x[0])]
                    best_order  = loop_orders[int(result.x[1])]

                    print(f"[贝叶斯 完成] 共评估 {_call_counter[0]} 次", flush=True)
                    print(f"[贝叶斯 最优分块策略] tile_M={best_tile[0]}, tile_N={best_tile[1]}, tile_K={best_tile[2]}", flush=True)
                    print(f"[贝叶斯 循环顺序] {best_order}", flush=True)
                    print(f"[贝叶斯 预估执行时间] {best_lat_us:.3f} μs  ({int(result.fun)} cycles)", flush=True)
                    print(f"[贝叶斯 收敛曲线] call -> best_cycles_so_far:", flush=True)
                    best_so_far = float('inf')
                    for i, v in enumerate(result.func_vals):
                        if v < best_so_far:
                            best_so_far = v
                            print(f"  #{i+1:03d}: {best_so_far:.0f}", flush=True)

                    min_cycle_count = int(result.fun)
                    best_mapping = self.Mapping(best_tile[0], best_tile[1], best_tile[2], best_order)

                except ImportError:
                    print("[贝叶斯] scikit-optimize 未安装，请执行: pip install scikit-optimize", flush=True)
                    raise
                except Exception as e:
                    print(f"[贝叶斯] 运行异常: {e}", flush=True)
                    raise

            else:
                raise ValueError(f"compile_mode '{compile_mode}' 不支持，请选择 'fast' / 'exhaustive' / 'bayes'")

            self.best_mapping = best_mapping
            self.best_cycle_count = min_cycle_count
            self.best_latency = min_cycle_count / clock_freq
            self.latency = self.best_latency
            return self.latency

    
    def simulate(
            self,
            computational_graph: ComputationalGraph,
            mapping: Mapping,
            chip_type: 'HardwareSpec' = HW,
        ) -> int:

            # 基本参数
            M = computational_graph.M
            N = computational_graph.N
            K = computational_graph.K
            word = computational_graph.data_type.word_size

            l1M = mapping.l1_tile_M
            l1N = mapping.l1_tile_N
            l1K = mapping.l1_tile_K

            # 分块数与边界大小
            M_t = M // l1M; M_r = M % l1M
            N_t = N // l1N; N_r = N % l1N
            K_t = K // l1K; K_r = K % l1K

            tiles_M = (M_t + (1 if M_r else 0))
            tiles_N = (N_t + (1 if N_r else 0))
            tiles_K = (K_t + (1 if K_r else 0))

            #  预计算每类 tile 的 compute 周期 & reduce 周期
            util = max(1e-6, float(getattr(chip_type, 'CUBE_UTILIZATION', 1.0))) 
            macs_pc = float(getattr(chip_type, 'CUBE_MACS_PER_CORE', 4096))
            vec_pc  = max(1, int(getattr(chip_type, 'total_vector_flops_per_cycle', 128)))

            def cc(m, n, k):
                # ceil( M*N*K / (CUBE_MACS_PER_CORE * UTIL) )
                return int(ceil((m*n*k) / (macs_pc * util)))

            def rc(m, n):
                # 归约周期：ceil(M*N / total_vector_flops_per_cycle)
                return int(ceil((m*n) / vec_pc))
            
            def read_l1_to_l0(m,n,k):
                #L1到L0A/LOB的read周期，属于MTE1
                '''---'''
                # 转换为KB
                traffic_size_kb_L0A = m*k*word / 1024
                # 查找对应的效率值（大于等于当前traffic_size_kb的最小key对应的效率）
                L12L0A_efficiency = 0.49  # 默认值
                for size, eff in self.L12L0A_eff_list:
                    if traffic_size_kb_L0A  >= size:
                        L12L0A_efficiency = eff
                    else:
                        break  # 已排序，后续值更大，直接跳出

                read_l1_to_l0A_cycle=m*k*word/(chip_type.IO_BW['L1→L0A']*L12L0A_efficiency)
                '''---'''
                '''---'''
                # 转换为KB
                traffic_size_kb_L0B = k*n*word / 1024
                # 查找对应的效率值（大于等于当前traffic_size_kb的最小key对应的效率）
                L12L0B_efficiency = 0.79  # 默认值
                for size, eff in self.L12L0B_eff_list:
                    if traffic_size_kb_L0B  >= size:
                        L12L0B_efficiency = eff
                    else:
                        break  # 已排序，后续值更大，直接跳出

                read_l1_to_l0B_cycle=k*n*word/(chip_type.IO_BW['L1→L0B']*L12L0B_efficiency)
                '''---'''
                return max(read_l1_to_l0A_cycle,read_l1_to_l0B_cycle)

            baseM = l1M
            baseN = l1N
            baseK = l1K
            m_last = M_r if M_r else l1M
            n_last = N_r if N_r else l1N
            k_last = K_r if K_r else l1K

            # 8类组合（含三边界）
            CC = {
                (0,0,0): cc(baseM, baseN, baseK),
                (1,0,0): cc(m_last, baseN, baseK),
                (0,1,0): cc(baseM, n_last, baseK),
                (0,0,1): cc(baseM, baseN, k_last),
                (1,1,0): cc(m_last, n_last, baseK),
                (1,0,1): cc(m_last, baseN, k_last),
                (0,1,1): cc(baseM, n_last, k_last),
                (1,1,1): cc(m_last, n_last, k_last),
            }
            RC = {
                (0,0): rc(baseM, baseN),
                (1,0): rc(m_last, baseN),
                (0,1): rc(baseM, n_last),
                (1,1): rc(m_last, n_last),
            }
            #L1到L0A/LOB的read周期
            READ_L1_TO_L0 = {
                (0,0,0): read_l1_to_l0(baseM, baseN, baseK),
                (1,0,0): read_l1_to_l0(m_last, baseN, baseK),
                (0,1,0): read_l1_to_l0(baseM, n_last, baseK),
                (0,0,1): read_l1_to_l0(baseM, baseN, k_last),
                (1,1,0): read_l1_to_l0(m_last, n_last, baseK),
                (1,0,1): read_l1_to_l0(m_last, baseN, k_last),
                (0,1,1): read_l1_to_l0(baseM, n_last, k_last),
                (1,1,1): read_l1_to_l0(m_last, n_last, k_last),
            }

            # ---- 预计算 tile 尺寸（用于 IO 统计） ----
            MK_size, KN_size, MN_size = {}, {}, {}
            for mi in range(tiles_M):
                msz = (M_r if (mi == tiles_M-1 and M_r) else l1M)
                for kk in range(tiles_K):
                    ksz = (K_r if (kk == tiles_K-1 and K_r) else l1K)
                    MK_size[(mi,kk)] = msz * ksz
            for kk in range(tiles_K):
                ksz = (K_r if (kk == tiles_K-1 and K_r) else l1K)
                for nj in range(tiles_N):
                    nsz = (N_r if (nj == tiles_N-1 and N_r) else l1N)
                    KN_size[(kk,nj)] = ksz * nsz
            for mi in range(tiles_M):
                msz = (M_r if (mi == tiles_M-1 and M_r) else l1M)
                for nj in range(tiles_N):
                    nsz = (N_r if (nj == tiles_N-1 and N_r) else l1N)
                    MN_size[(mi,nj)] = msz * nsz

            # ---- DMA 带宽常数（与原逻辑一致：平均到 24 核） ----

            # ---- 批次状态（集合实现） ----
            prev_read_mk: set = set()
            prev_read_kn: set = set()
            prev_read_mn: set = set()
            prev_write_mn: set = set()
            prev_compute_cc  = 0

            total_cycles = 0

            def edge_flags(mi, nj, kk):
                return (1 if (mi==tiles_M-1 and M_r) else 0,
                        1 if (nj==tiles_N-1 and N_r) else 0,
                        1 if (kk==tiles_K-1 and K_r) else 0)
            def edge_flags_mn(mi, nj):
                return (1 if (mi==tiles_M-1 and M_r) else 0,
                        1 if (nj==tiles_N-1 and N_r) else 0)

            # ---- 分块循环（按 loop_order 遍历） ----
            active: list = []
            for m in range(tiles_M) if mapping.l1_loop_order[0] == 'm' else \
                            range(tiles_N) if mapping.l1_loop_order[0] == 'n' else \
                            range(tiles_K):
                for n in range(tiles_N) if mapping.l1_loop_order[1] == 'n' else \
                                range(tiles_K) if mapping.l1_loop_order[1] == 'k' else \
                                range(tiles_M):
                    for k in range(tiles_K) if mapping.l1_loop_order[2] == 'k' else \
                                    range(tiles_M) if mapping.l1_loop_order[2] == 'm' else \
                                    range(tiles_N):
                        # 将 (m,n,k) 映射回标准下标
                        if mapping.l1_loop_order == 'mnk':
                            mi, nj, kk = m, n, k
                        elif mapping.l1_loop_order == 'mkn':
                            mi, kk, nj = m, n, k
                        elif mapping.l1_loop_order == 'nmk':
                            nj, mi, kk = m, n, k
                        elif mapping.l1_loop_order == 'nkm':
                            nj, kk, mi = m, n, k
                        elif mapping.l1_loop_order == 'knm':
                            kk, nj, mi = m, n, k
                        elif mapping.l1_loop_order == 'kmn':
                            kk, mi, nj = m, n, k
                        else:
                            mi, nj, kk = m, n, k

                        active.append((mi, nj, kk))

                        # 批次未满且非最后一个 tile，继续攒批
                        if not (len(active) >= chip_type.AI_CORE_COUNT or
                                (mi == tiles_M-1 and nj == tiles_N-1 and kk == tiles_K-1)):
                            continue

                        # --- 统计当前批次 ---
                        cur_mk: set = set()
                        cur_kn: set = set()
                        cur_mn_read: set = set()
                        cur_mn_write: set = set()
                        cur_max_compute = 0

                        for (mi2, nj2, kk2) in active:
                            cur_mk.add((mi2, kk2))
                            cur_kn.add((kk2, nj2))
                            if kk2 > 0:
                                cur_mn_read.add((mi2, nj2))
                            cur_mn_write.add((mi2, nj2))

                            flg = edge_flags(mi2, nj2, kk2)
                            cc_val = CC[flg]
                            # L1到L0A/LOB的read周期
                            cc_val += READ_L1_TO_L0[flg]
                            #不确定vector这块加不加
                            # if kk2 > 0:
                            #     rc_val = RC[edge_flags_mn(mi2, nj2)]
                            #     cc_val += rc_val
                            if cc_val > cur_max_compute:
                                cur_max_compute = cc_val

                        # 新读取的数据量（排除上批缓存/写回中数据）
                        new_mk = cur_mk - prev_read_mk
                        new_kn = cur_kn - prev_read_kn
                        blocked = prev_read_mn | prev_write_mn  # 避免 RAW
                        new_mn = {pair for pair in cur_mn_read if pair not in blocked}

                        cur_read_elems = 0
                        for key in new_mk: cur_read_elems += MK_size[key]
                        for key in new_kn: cur_read_elems += KN_size[key]
                        for key in new_mn: cur_read_elems += MN_size[key]
                        '''---'''
                        # 转换为KB
                        traffic_size_kb = cur_read_elems * word / 1024
                        # 查找对应的效率值（大于等于当前traffic_size_kb的最小key对应的效率）
                        OUT2L1_efficiency = 0.2  # 默认值
                        for size, eff in self.OUT2L1_eff_list:
                            if traffic_size_kb >= size:
                                OUT2L1_efficiency = eff
                            else:
                                break  # 已排序，后续值更大，直接跳出
                        
                        cur_read_cycles = int(ceil((cur_read_elems * word) / (24*chip_type.IO_BW['Real DRAM→L1']*OUT2L1_efficiency)))
                        '''---'''

                        # 上一批次需要写回但本批不再复用的 C
                        prev_write_only = prev_write_mn - cur_mn_read
                        prev_write_elems = 0
                        for key in prev_write_only: prev_write_elems += MN_size[key]
                        '''---'''
                        # 转换为KB
                        traffic_size_kb = prev_write_elems * word / 1024
                        # 查找对应的效率值（大于等于当前traffic_size_kb的最小key对应的效率）
                        OUT2L1_efficiency = 0.2  # 默认值
                        for size, eff in self.OUT2L1_eff_list:
                            if traffic_size_kb >= size:
                                OUT2L1_efficiency = eff
                            else:
                                break  # 已排序，后续值更大，直接跳出

                        prev_write_cycles = int(ceil((prev_write_elems * word) / (24*chip_type.IO_BW['Real DRAM→L1']*OUT2L1_efficiency)))
                        '''---'''

                        total_cycles += max(cur_read_cycles, prev_compute_cc) + prev_write_cycles

                        # 更新状态
                        prev_compute_cc = cur_max_compute
                        prev_read_mk = cur_mk
                        prev_read_kn = cur_kn
                        prev_read_mn = cur_mn_read
                        prev_write_mn = cur_mn_write
                        active.clear()

            # 尾批：完成计算 + 写回
            tail_write_elems = 0
            for key in prev_write_mn:
                tail_write_elems += MN_size[key]
            
            '''---'''
            # 转换为KB
            traffic_size_kb = tail_write_elems * word / 1024
            # 查找对应的效率值（大于等于当前traffic_size_kb的最小key对应的效率）
            OUT2L1_efficiency = 0.2  # 默认值
            for size, eff in self.OUT2L1_eff_list:
                if traffic_size_kb >= size:
                    OUT2L1_efficiency = eff
                else:
                    break  # 已排序，后续值更大，直接跳出

            tail_write_cycles = int(ceil((tail_write_elems * word) / (24*chip_type.IO_BW['Real DRAM→L1']*OUT2L1_efficiency)))
            '''---'''

            total_cycles += prev_compute_cc + tail_write_cycles
            return total_cycles

    
    class L1TileSimulator:
        '''初始化一个L1缓存级分块，并立即计算其在硬件上的执行周期。'''
        def __init__(
            self,
            M: int,                             
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chip_type: 'HardwareSpec' = HW,
        ):
            # print(f'L1 tile: {M} {N} {K}')
            self.M = M  #其实赋值的是l1_tile_M！！！！！！！！！！！
            self.N = N
            self.K = K
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(M, N, K, data_type, mapping, chip_type)
            

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chip_type: 'HardwareSpec' = HW,
        ):  
            '''1.L1容量校验
            硬件映射：确保分块数据可放入计算核心的片上L1（考虑双缓冲）

                公式：
                L1容量 ≥ (A元素+B元素+C元素) × 数据类型字节数 × 2
                （//2 是因为双缓冲需要两倍空间）'''
            assert (
                M * K + K * N + M * N  # A(M×K)+B(K×N)+C(M×N)的总元素数
                <= chip_type.L1_CAPACITY // data_type.word_size // 2
            )
            # 删去了脉动阵列并行度校验！！！！！
            '''2. 计算周期核心逻辑'''
            flop_count = 2 * self.M * self.K * self.N
            compute_cycle_count = ceil(
                flop_count/(chip_type.CUBE_MACS_PER_CORE*2*chip_type.CUBE_UTILIZATION)
            ) #AI_corecount记得删掉，之前除以24，导致cube时间过小！！！！！

            return compute_cycle_count