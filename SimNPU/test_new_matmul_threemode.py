# -*- coding: utf-8 -*-
"""
Parallel test runner for Matmul.
- Supports Chinese punctuation in MNK strings.
- Runs Roofline and compile&simulate per case.
- Parallel via ProcessPoolExecutor (process-based, safe for CPU-bound work).
Usage (optional):
    python test_new_matmul.py --workers 4 --mode fast --limit 0
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# --- Your project imports (import inside worker too for safety) ---
from utils import data_type_dict, Tensor
from new_matmul_threemode import Matmul
from hardware import HW

# === Test shapes (supports Chinese punctuation) ===
# DIM_STRS = [
#     "1096,4096;4096,4096",
#     "1114,12288;12288,8064",
#     "10,15616;15616,4224",
#     "1051,15488;15488,12160",
#     "1054,8192;8192,128",
#     "1032,9472;9472,13568",
#     "1920,1920;1920,9088",
#     "101,1408;1408,256",
#     "1196,1152;1152,14080",
#     "1617,8704;8704,7296",
#     "1026,2176;2176,1536",
#     "1002,6912;6912,1536",
#     "556,640;640,384",
#     "1915,6016;6016,6400",
#     "268,384;384,3072",
#     "456,4096;4096,7424",
#     "1097,2432;2432,8448",
#     "1431,5888;5888,5632",
# ]

# # #训练
# DIM_STRS = [
#     "2733，7616；7616，3136",
#     "5461,15552;15552,7616",
#     "9118,15936;15936,12608",
#     "8304,5184;5184,7360",
#     "13435,15552;15552,5056",
#     "15282,3840;3840,4416",
#     "12921,14080;14080,7040",
#     "14736,12480;12480,6080",
#     "1887,9536;9536,8064",
#     "13017,10368;10368,12096",
#     "7367,13952;13952,2880",
#     "14405,13504;13504,13248",
#     "15370,7232;7232,9408",
#     "4250,10048;10048,4864",
#     "1117,9344;9344,8704",
#     "5568,7232;7232,3776",
#     "4728,7616;7616,15104",
#     "6824,4608;4608,10816",
#     "5200,1920;1920,14656",
#     "11749,6080;6080,2368",
#     "688,14208;14208,7168",
#     "3547,13568;13568,9536",
#     "404,14848;14848,10432",
#     "5743,1984;1984,6720",
#     "7237,16000;16000,2432",
#     "8652,3008;3008,7744",
#     "10953,704;704,4224",
#     "1120,13312;13312,9728",
#     "15777,11776;11776,11648",
#     "8265,10368;10368,2688",
#     "15381,2944;2944,10880",
#     "6835,6656;6656,10496",
#     "10528,2368;2368,15744",
#     "10348,9728;9728,9856",
#     "10563,7168;7168,16000",
#     "12393,1344;1344,2368",
#     "12414,4224;4224,6080",
#     "1829,13440;13440,9984",
#     "1769,10752;10752,6656",
#     "510,4352;4352,8384",
#     "2932,12352;12352,7296",
#     "3825,15488;15488,1536",
#     "4894,4928;4928,5696",
#     "12792,2560;2560,15296",
#     "14849,7680;7680,15040",
#     "3415,2944;2944,8448",
#     "11493,7360;7360,14720",
#     "14686,1600;1600,11456",
#     "13935,8512;8512,11776",
#     "237,14912;14912,14912",
#     "15082,14400;14400,10496",
#     "16204,13952;13952,15168",
#     "11751,9088;9088,5248",
#     "4071,4864;4864,4800",
#     "684,6464;6464,5568",
#     "5418,12352;12352,3264",
#     "10355,12352;12352,1408",
#     "7504,7296;7296,1856",
#     "726,5760;5760,9152",
#     "15276,12736;12736,12800",
#     "3981,2112;2112,6080",
#     "2306,16192;16192,4928",
#     "9000,12032;12032,12736",
#     "962,6080;6080,14400",
#     "3979,6912;6912,9984",
#     "1092,13312;13312,3648",
#     "12857,6656;6656,6464",
#     "5115,14144;14144,6016",
#     "15356,1344;1344,16320",
#     "11853,7232;7232,8448",
#     "2397,14912;14912,5888",
#     "6445,7424;7424,4544",
#     "16343,7232;7232,4352",
#     "16185,13632;13632,3968",
#     "4507,5760;5760,12224",
#     "7610,4992;4992,15744",
#     "12485,7104;7104,11584",
#     "971,6720;6720,10176",
#     "13598,2368;2368,2176",
#     "15393,14720;14720,15168",
#     "6853,2048;2048,13568",
#     "6260,5312;5312,14336",
#     "13121,4416;4416,15680",
#     "14497,10752;10752,13760",
#     "1452,9664;9664,14912",
#     "12430,320;320,10048",
#     "11739,9856;9856,2752",
#     "6902,9152;9152,8000",
#     "10680,9792;9792,15680",
#     "13032,10496;10496,13952",
#     "1599,13312;13312,5696",
#     "12915,192;192,13952",
#     "563,7424;7424,10048",
#     "11746,9536;9536,13632",
#     "11505,15488;15488,3520",
# ]

# #维度为1
# DIM_STRS = [
#     "1,15168;15168,2496",
#     "1,8832;8832,3008",
#     "1,12800;12800,12096",
#     "1,5568;5568,11968",
#     "1,11136;11136,11904",
#     "1,13952;13952,6016",
# ]

def csv_column_to_list(file_path, column_name):
    """
    加载 CSV 文件，并将指定列转换为字符串列表

    参数:
    file_path (str): CSV 文件路径
    column_name (str): 要转换的列名

    返回:
    list: 包含指定列所有元素的字符串列表
    """
    try:
        # 加载 CSV 文件
        df = pd.read_csv(file_path)
        
        # 检查指定列是否存在
        if column_name not in df.columns:
            raise ValueError(f"列 '{column_name}' 不在文件中")
        
        # 将指定列转换为字符串类型，并转换为列表
        result_list = df[column_name].astype(str).tolist()

        return result_list

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return []
    except Exception as e:
        print(f"发生错误: {e}")
        return []

def parse_dim(s: str):
    s = s.strip().replace('，', ',').replace('；', ';').replace(' ', '')
    lhs, rhs = s.split(';')
    m, k1 = lhs.split(',')
    k2, n = rhs.split(',')
    m, n, k1, k2 = int(m), int(n), int(k1), int(k2)
    if k1 != k2:
        raise ValueError(f"K mismatch in '{s}': {k1} vs {k2}")
    return m, n, k1

def run_one(index: int, s: str, compile_mode: str = "fast",n_calls=80):
    """
    Worker run for a single case. Returns a tuple:
    (index, M, N, K, compute_us, io_us, roofline_us, sim_us, err)
    """
    try:
        # Import inside worker context to be robust to multiprocessing spawn
        from utils import data_type_dict, Tensor
        from new_matmul_threemode import Matmul
        from hardware import HW
        M, N, K = parse_dim(s)
        model = Matmul(data_type=data_type_dict["fp16"])
        model(Tensor([M, K]), Tensor([K, N]))
        # roofline
        a, b, compute, io, latency = model.roofline_model()
        compute_us = compute * 1e6
        io_us = io * 1e6
        roofline_us = latency * 1e6
        # compile & simulate
        sim_latency = model.compile_and_simulate(HW, compile_mode=compile_mode,n_calls=n_calls)
        sim_us = sim_latency * 1e6
        return (index, M, N, K, compute_us, io_us, roofline_us, sim_us, "")
    except Exception as e:
        return (index, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, repr(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Number of parallel workers (processes).")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "exhaustive","bayes"],
                        help="Compile mode for search.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of cases (0 for all).")
    parser.add_argument("--no-live", action="store_true",
                        help="Suppress live (out-of-order) progress lines.")
    parser.add_argument("--csv-only", action="store_true",
                        help="Print only a final CSV table in input order (best for Excel).")
    parser.add_argument("--n_calls", type=int, default=80,
                        help="Number of sampling points for Bayesian optimization.")
    args = parser.parse_args()

    '''单矩阵测试'''
    model = Matmul(data_type=data_type_dict["fp16"])
    model(Tensor([1096, 1600]), Tensor([1600, 1096]))
    sim_latency =model.compile_and_simulate(HW, compile_mode=args.mode,n_calls=args.n_calls)
    print(sim_latency* 1e6)

    '''文件多矩阵测试'''
#     #加载csv文件，并把矩阵那列转换为str的list 可注释！！！！！
#     file_path = '101 个矩阵_Input_Shapes.csv'#
#     column_name = 'Input Shapes'
#     DIM_STRS = csv_column_to_list(file_path, column_name)

#    #可注释！！！！！
#     tasks = list(enumerate(DIM_STRS))
#     if args.limit and args.limit > 0:
#         tasks = tasks[:args.limit]

#     if not args.csv_only:
#         print(f"Running {len(tasks)} cases with {args.workers} workers, mode={args.mode}")

#     # Prepare ordered containers
#     results = [None] * len(tasks)
#     compute_list = [0.0] * len(tasks)
#     total_list   = [0.0] * len(tasks)

#     # Parallel execution
#     with ProcessPoolExecutor(max_workers=args.workers) as ex:
#         fut2idx = {ex.submit(run_one, idx, s, args.mode,args.n_calls): idx for idx, s in tasks}
#         for fut in as_completed(fut2idx):
#             idx = fut2idx[fut]
#             try:
#                 res = fut.result(timeout=None)
#             except Exception as e:
#                 res = (idx, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, "future-exc:"+repr(e))
#             results[idx] = res
#             if (not args.no_live) and (not args.csv_only):
#                 # live progress line in completion order
#                 _, M, N, K, compute_us, io_us, roofline_us, sim_us, err = res
#                 if err:
#                     print(f"[live {idx:03d}] ERROR {DIM_STRS[idx]} -> {err}")
#                 else:
#                     print(f"[live {idx:03d}] done M={M},N={N},K={K}  sim={sim_us:.3f}μs")

#     # ---- Ordered printing (strictly follows the input order) ----
#     if not args.csv_only:
#         for idx, _ in tasks:
#             _, M, N, K, compute_us, io_us, roofline_us, sim_us, err = results[idx]
#             if err:
#                 print(f"[{idx:03d}] ERROR {DIM_STRS[idx]} -> {err}")
#             else:
#                 print(f"[{idx:03d}] M={M},N={N},K={K}  "
#                       f"compute={compute_us:.3f}μs  io={io_us:.3f}μs  "
#                       f"roofline={roofline_us:.3f}μs  sim={sim_us:.3f}μs")
#                 compute_list[idx] = compute_us
#                 total_list[idx]   = sim_us

#         # Ordered numeric summaries (exact input order) — 保留你原有的两段纯数字
#         for t in compute_list:
#             print(t)
        
#         print('-------------------------------------------------------------')
        
#         for t in total_list:
#             print(t)

#     # ---- Final CSV (for Excel) in the exact input order ----
#     # Columns: idx,M,N,K,compute_us,io_us,roofline_us,sim_us
#     print("idx,M,N,K,compute_us,io_us,roofline_us,sim_us")
#     for idx, _ in tasks:
#         if results[idx] is None:
#             print(f"{idx},,,," ",,,")
#             continue
#         _, M, N, K, compute_us, io_us, roofline_us, sim_us, err = results[idx]
#         if err:
#             print(f"{idx},{M},{N},{K},ERROR,ERROR,ERROR,ERROR")
#         else:
#             print(f"{idx},{M},{N},{K},{compute_us:.6f},{io_us:.6f},{roofline_us:.6f},{sim_us:.6f}")


    

if __name__ == "__main__":
    main()
