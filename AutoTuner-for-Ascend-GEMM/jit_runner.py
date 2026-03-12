#!/usr/bin/env python3
"""
CATLASS JIT Runner
动态编译和执行CATLASS kernel，支持任意tile配置
"""

import os
import re
import time
import subprocess
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional


class CATLASSJITRunner:
    """JIT编译和执行CATLASS kernel"""
    
    def __init__(self, catlass_dir: Optional[Path] = None):
        """初始化JIT Runner"""
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.catlass_dir = catlass_dir or self.base_dir / "catlass"
        self.template_file = self.base_dir / "custom_kernel" / "tunable_gemm_template.cpp"
        self.examples_dir = self.catlass_dir / "examples" / "jit_gemm"
        self.output_dir = self.catlass_dir / "output" / "bin"
        
        # 缓存已编译的kernel
        self.kernel_cache: Dict[str, Path] = {}
        
        # 检查模板文件
        if not self.template_file.exists():
            raise FileNotFoundError(f"模板文件不存在: {self.template_file}")
        
        # 创建JIT examples目录
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"JIT Runner初始化完成")
        print(f"  CATLASS目录: {self.catlass_dir}")
        print(f"  模板文件: {self.template_file}")
    
    def _config_hash(self, config: dict) -> str:
        """生成配置的唯一hash"""
        config_str = "_".join(f"{k}={v}" for k, v in sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _generate_cpp(self, config: dict) -> str:
        """根据配置生成C++代码"""
        with open(self.template_file, 'r') as f:
            template = f.read()
        
        # 替换占位符
        replacements = {
            '{{TILE_M}}': str(config.get('tile_m', 128)),
            '{{TILE_N}}': str(config.get('tile_n', 256)),
            '{{L1_K}}': str(config.get('l1_k', 256)),
            '{{L0_K}}': str(config.get('l0_k', 64)),
            '{{ENABLE_UNIT}}': 'true' if config.get('enable_unit', 0) else 'false',
            '{{ENABLE_SHUFFLE_K}}': 'true' if config.get('enable_shuffle_k', 0) else 'false',
            '{{ENABLE_ABBA}}': 'true' if config.get('enable_abba', 0) else 'false',
            '{{SWIZZLE_OFFSET}}': str(config.get('swizzle_offset', 3)),
            '{{SWIZZLE_DIRECTION}}': str(config.get('swizzle_direction', 0)),
        }
        
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        
        return template
    
    def _compile_kernel(self, config: dict) -> Tuple[bool, Optional[Path], str]:
        """编译kernel，返回(成功, kernel路径, 错误信息)"""
        config_hash = self._config_hash(config)
        kernel_name = f"jit_gemm_{config_hash}"
        
        # 检查缓存
        if config_hash in self.kernel_cache:
            kernel_path = self.kernel_cache[config_hash]
            if kernel_path.exists():
                return True, kernel_path, ""
        
        # 检查输出是否已存在
        kernel_path = self.output_dir / kernel_name
        if kernel_path.exists():
            self.kernel_cache[config_hash] = kernel_path
            return True, kernel_path, ""
        
        # 生成C++代码
        cpp_code = self._generate_cpp(config)
        
        # 写入文件
        cpp_file = self.examples_dir / f"{kernel_name}.cpp"
        with open(cpp_file, 'w') as f:
            f.write(cpp_code)
        
        # 创建/更新 CMakeLists.txt（追加模式，避免覆盖之前的 kernel）
        cmake_entry = f"""set_source_files_properties({kernel_name}.cpp PROPERTIES LANGUAGE ASCEND)
catlass_example_add_executable({kernel_name} dav-c220 {kernel_name}.cpp)
"""
        cmake_file = self.examples_dir / "CMakeLists.txt"
        
        # 检查是否已存在该 kernel 的条目
        existing_content = ""
        if cmake_file.exists():
            with open(cmake_file, 'r') as f:
                existing_content = f.read()
        
        # 如果不存在则追加
        if kernel_name not in existing_content:
            with open(cmake_file, 'a') as f:
                f.write(cmake_entry)

        
        # 确保jit_gemm在examples列表中
        self._ensure_example_registered()
        
        # 编译
        print(f"  编译中: {kernel_name}...", end=" ", flush=True)
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ["bash", "scripts/build.sh", kernel_name],
                cwd=str(self.catlass_dir),
                capture_output=True,
                text=True,
                timeout=120
            )
            
            compile_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"失败 ({compile_time:.1f}s)")
                error_msg = self._extract_error_message(result.stderr)
                print(f"  错误: {result.stderr[:200]}...")
                return False, None, error_msg
            
            print(f"成功 ({compile_time:.1f}s)")
            
            # 检查输出文件
            kernel_path = self.output_dir / kernel_name
            if kernel_path.exists():
                self.kernel_cache[config_hash] = kernel_path
                return True, kernel_path, ""
            else:
                print(f"  警告: 编译成功但找不到输出文件")
                return False, None, "输出文件不存在"
                
        except subprocess.TimeoutExpired:
            print(f"超时")
            return False, None, "编译超时"
        except Exception as e:
            print(f"异常: {e}")
            return False, None, str(e)
    
    def _extract_error_message(self, stderr: str) -> str:
        """从编译错误输出中提取关键信息"""
        # 常见错误模式
        patterns = [
            (r"error:.*?(?=\n|$)", "编译错误"),
            (r"L0C.*?exceeded", "L0C空间超限"),
            (r"L1.*?exceeded", "L1空间超限"),
            (r"tile.*?too large", "Tile过大"),
            (r"invalid.*?configuration", "无效配置"),
            (r"static_assert.*?failed", "静态断言失败"),
        ]
        
        import re
        for pattern, label in patterns:
            match = re.search(pattern, stderr, re.IGNORECASE)
            if match:
                return f"{label}: {match.group(0)[:100]}"
        
        # 如果没有匹配到，返回前100个字符
        return stderr[:100] if stderr else "未知错误"
    
    def _ensure_example_registered(self):
        """确保jit_gemm在examples CMakeLists.txt中注册"""
        examples_cmake = self.catlass_dir / "examples" / "CMakeLists.txt"
        
        with open(examples_cmake, 'r') as f:
            content = f.read()
        
        if 'jit_gemm' not in content:
            # 找到foreach循环，添加jit_gemm
            # 这是简化处理，实际可能需要更复杂的解析
            if 'add_subdirectory(jit_gemm)' not in content:
                # 在文件末尾添加
                with open(examples_cmake, 'a') as f:
                    f.write('\nadd_subdirectory(jit_gemm)\n')
    
    def run(self, m: int, n: int, k: int, config: dict, 
            warmup: int = 3, repeat: int = 10) -> Tuple[float, float, str]:
        """
        JIT编译并运行kernel
        
        返回: (执行时间ms, GFLOPS, 错误信息)
        """
        # 检查约束
        if not self._check_constraints(config):
            print(f"  配置违反约束，跳过")
            return 1e10, 0.0, "违反约束"
        
        # 编译kernel
        success, kernel_path, error_msg = self._compile_kernel(config)
        if not success:
            return 1e10, 0.0, error_msg
        
        # 执行kernel
        cmd = [
            str(kernel_path),
            str(m), str(n), str(k),
            str(warmup), str(repeat)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.output_dir),
                env=os.environ.copy()  # 传递环境变量（含ASCEND_DEVICE_ID）
            )
            
            if result.returncode != 0:
                print(f"  执行失败: {result.stderr}")
                return 1e10, 0.0, f"执行失败: {result.stderr[:100]}"
            
            # 解析输出（支持科学计数法如 1.79e+06）
            output = result.stdout
            time_match = re.search(r'TIME_MS:\s*([\d.eE+\-]+)', output)
            gflops_match = re.search(r'GFLOPS:\s*([\d.eE+\-]+)', output)
            
            if time_match and gflops_match:
                exec_time = float(time_match.group(1))
                gflops = float(gflops_match.group(1))
                return exec_time, gflops, ""
            else:
                print(f"  无法解析输出: {output}")
                return 1e10, 0.0, "无法解析输出"
                
        except subprocess.TimeoutExpired:
            print(f"  执行超时")
            return 1e10, 0.0, "执行超时"
        except Exception as e:
            print(f"  执行异常: {e}")
            return 1e10, 0.0, str(e)
    
    def _check_constraints(self, config: dict) -> bool:
        """检查配置是否满足硬件约束"""
        tile_m = config.get('tile_m', 128)
        tile_n = config.get('tile_n', 256)
        l1_k = config.get('l1_k', 256)
        l0_k = config.get('l0_k', 64)
        
        # 约束1: L1_K必须能被L0_K整除
        if l1_k % l0_k != 0:
            return False
        
        # 约束2: 256x256超过L0C空间
        if tile_m == 256 and tile_n == 256:
            return False
        
        # 约束3: tile_m和tile_n至少64
        if tile_m < 64 or tile_n < 64:
            return False
        
        return True
    
    def get_valid_search_space(self) -> list:
        """返回满足约束的完整搜索空间"""
        valid_configs = []
        
        tile_options = [64, 128, 256]
        k_options = {
            64: [16, 32, 64],
            128: [16, 32, 64, 128],
            256: [16, 32, 64, 128, 256],
            512: [32, 64, 128, 256, 512],
        }
        
        for tile_m in tile_options:
            for tile_n in tile_options:
                # 跳过256x256
                if tile_m == 256 and tile_n == 256:
                    continue
                
                for l1_k in [64, 128, 256, 512]:
                    for l0_k in k_options.get(l1_k, []):
                        for enable_unit in [0, 1]:
                            for enable_shuffle_k in [0, 1]:
                                for enable_abba in [0, 1]:
                                    for swizzle_offset in [1, 2, 3, 4]:
                                        for swizzle_direction in [0, 1]:
                                            config = {
                                                'tile_m': tile_m,
                                                'tile_n': tile_n,
                                                'l1_k': l1_k,
                                                'l0_k': l0_k,
                                                'enable_unit': enable_unit,
                                                'enable_shuffle_k': enable_shuffle_k,
                                                'enable_abba': enable_abba,
                                                'swizzle_offset': swizzle_offset,
                                                'swizzle_direction': swizzle_direction,
                                            }
                                            valid_configs.append(config)
        
        return valid_configs


def main():
    """测试JIT Runner"""
    runner = CATLASSJITRunner()
    
    print("\n" + "=" * 60)
    print("JIT编译测试")
    print("=" * 60)
    
    # 测试配置
    config = {
        'tile_m': 128,
        'tile_n': 256,
        'l1_k': 256,
        'l0_k': 64,
        'enable_unit': 1,
        'enable_shuffle_k': 0,
        'enable_abba': 1,
        'swizzle_offset': 3,
        'swizzle_direction': 0,
    }
    
    print(f"\n测试配置: {config}")
    
    exec_time, gflops = runner.run(m=4096, n=4096, k=4096, config=config)
    tflops = gflops / 1000
    efficiency = tflops / 352 * 100
    
    print(f"\n结果:")
    print(f"  时间: {exec_time:.3f} ms")
    print(f"  性能: {tflops:.2f} TFLOPS ({efficiency:.1f}%)")
    
    # 显示搜索空间大小
    valid_space = runner.get_valid_search_space()
    print(f"\n有效搜索空间大小: {len(valid_space)} 个配置")


if __name__ == "__main__":
    main()
