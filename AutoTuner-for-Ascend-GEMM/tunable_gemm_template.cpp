/**
 * CATLASS Tunable GEMM - JIT Template
 * 
 * 这个文件包含占位符，由Python脚本动态替换后编译
 * 占位符格式: {{PARAMETER_NAME}}
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <acl/acl.h>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"

using namespace Catlass;

// ========== 由JIT脚本替换的参数 ==========
// Tile维度
constexpr int TILE_M = {{TILE_M}};
constexpr int TILE_N = {{TILE_N}};
constexpr int L1_K = {{L1_K}};
constexpr int L0_K = {{L0_K}};

// GemmAtlasA2 flags
constexpr bool ENABLE_UNIT = {{ENABLE_UNIT}};
constexpr bool ENABLE_SHUFFLE_K = {{ENABLE_SHUFFLE_K}};
constexpr bool ENABLE_ABBA = {{ENABLE_ABBA}};

// Swizzle参数
constexpr int SWIZZLE_OFFSET = {{SWIZZLE_OFFSET}};
constexpr int SWIZZLE_DIRECTION = {{SWIZZLE_DIRECTION}};
// ========================================

#define ACL_CHECK(status)                                                   \
    do {                                                                   \
        aclError error = status;                                          \
        if (error != ACL_ERROR_NONE) {                                   \
            std::cerr << __FILE__ << ":" << __LINE__                      \
                     << " aclError:" << error << std::endl;              \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

template<typename T>
void FillRandomData(std::vector<T>& data, float min_val, float max_val) {
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned>(time(nullptr)));
        seeded = true;
    }
    for (auto& val : data) {
        float rand_val = min_val + (max_val - min_val) * (rand() / static_cast<float>(RAND_MAX));
        val = static_cast<T>(rand_val);
    }
}

class PerfTimer {
    std::chrono::high_resolution_clock::time_point t0;
public:
    PerfTimer() : t0(std::chrono::high_resolution_clock::now()) {}
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

double RunGemm(int m, int n, int k, int warmup, int repeat) {
    // 初始化ACL
    ACL_CHECK(aclInit(nullptr));
    
    // 从环境变量读取设备ID，支持多卡并行
    int deviceId = 0;
    const char* deviceIdEnv = getenv("ASCEND_DEVICE_ID");
    if (deviceIdEnv != nullptr) {
        deviceId = atoi(deviceIdEnv);
    }
    ACL_CHECK(aclrtSetDevice(deviceId));
    
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    // 使用编译时确定的tile参数
    using L1TileShape = GemmShape<TILE_M, TILE_N, L1_K>;
    using L0TileShape = GemmShape<TILE_M, TILE_N, L0_K>;  // M/N与L1相同

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(half);
    size_t sizeB = lenB * sizeof(half);
    size_t sizeC = lenC * sizeof(half);

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    LayoutA layoutA{static_cast<uint32_t>(m), static_cast<uint32_t>(k)};
    LayoutB layoutB{static_cast<uint32_t>(k), static_cast<uint32_t>(n)};
    LayoutC layoutC{static_cast<uint32_t>(m), static_cast<uint32_t>(n)};

    std::vector<half> hostA(lenA);
    std::vector<half> hostB(lenB);
    FillRandomData<half>(hostA, -5.0f, 5.0f);
    FillRandomData<half>(hostB, -5.0f, 5.0f);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint32_t aicCoreNum = 4;

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;
    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<SWIZZLE_OFFSET, SWIZZLE_DIRECTION>;

    using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    GemmCoord problemShape{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};
    typename MatmulKernel::Arguments arguments{problemShape, deviceA, deviceB, deviceC};
    MatmulAdapter matmul_op;
    matmul_op.CanImplement(arguments);
    
    // 分配workspace
    size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    
    matmul_op.Initialize(arguments, deviceWorkspace);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        matmul_op(stream, aicCoreNum);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // Timing
    PerfTimer timer;
    timer.start();
    for (int i = 0; i < repeat; ++i) {
        matmul_op(stream, aicCoreNum);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    double total_ms = timer.elapsed_ms();
    double avg_ms = total_ms / repeat;

    // 计算GFLOPS
    double flops = 2.0 * m * n * k;
    double gflops = flops / (avg_ms / 1000.0) / 1e9;

    // Cleanup
    if (deviceWorkspace) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtDestroyStream(stream));
    
    // 重置设备（使用与初始化相同的设备ID）
    int resetDeviceId = 0;
    const char* resetEnv = getenv("ASCEND_DEVICE_ID");
    if (resetEnv != nullptr) {
        resetDeviceId = atoi(resetEnv);
    }
    ACL_CHECK(aclrtResetDevice(resetDeviceId));
    ACL_CHECK(aclFinalize());

    std::cout << "TIME_MS: " << avg_ms << " GFLOPS: " << gflops << std::endl;
    return avg_ms;
}

int main(int argc, const char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " M N K WARMUP REPEAT" << std::endl;
        return 1;
    }

    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);
    int warmup = std::atoi(argv[4]);
    int repeat = std::atoi(argv[5]);

    std::cout << "Config: Tile[" << TILE_M << "," << TILE_N << "] ";
    std::cout << "K[L1=" << L1_K << ",L0=" << L0_K << "] ";
    std::cout << "Unit=" << ENABLE_UNIT << " Shuffle=" << ENABLE_SHUFFLE_K;
    std::cout << " ABBA=" << ENABLE_ABBA << " Swizzle[" << SWIZZLE_OFFSET << "," << SWIZZLE_DIRECTION << "]" << std::endl;

    RunGemm(m, n, k, warmup, repeat);
    return 0;
}
