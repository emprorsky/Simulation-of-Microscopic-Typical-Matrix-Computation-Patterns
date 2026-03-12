class HardwareSpec:
    def __init__(self):
        self.CUBE_UTILIZATION=0.97       #0.97在95组数据上拟合得最好！！！！！
        self.L2_CACHE_HIT_RATE=1
        self.AI_CORE_COUNT = 24
        self.CUBE_MACS_PER_CORE = 4096
        # For backward compatibility: use per-core number here
        self.CUBE_MACS_PER_CYCLE = self.CUBE_MACS_PER_CORE
        self.CHIP_MACS_PER_CYCLE = self.CUBE_MACS_PER_CORE * self.AI_CORE_COUNT
        self.CLOCK_FREQ = 1.85e9  # Hz
        # 每个物理周期包含的 AIC 内部 tick 数（与 profiling 对齐）
        self.TICKS_PER_CYCLE = 6
        # AIC tick 的等效频率
        self.AIC_TICK_FREQ = self.CLOCK_FREQ * self.TICKS_PER_CYCLE
        self.ALIGN_COMPUTE_16 = False
        self.total_vector_flops_per_cycle = 256 # 占位值，不一定准 ！！！！！ 

        self.MIN_ACCESS = {
            'Chip': 2,
            'L2': 512,
            'L1': 32,
            'L0A': 512,
            'L0B': 512,
            'L0C': 512,
            'UB': 32,
            'SB': 2,
            'ABDFF': 512,
            'AccumDFF': 512
        }
        self.MIN_ACCESS['DRAM'] = self.MIN_ACCESS['L2']
        self.MIN_ACCESS['EXT']  = self.MIN_ACCESS['Chip']

        # Capacities
        self.MEM_CAPACITY      = 64 * 1024**3
        self.L2_CAPACITY       = 192 * 1024**2
        self.L1_CAPACITY       = 1   * 1024**2
        self.L0A_CAPACITY      = 64  * 1024
        self.L0B_CAPACITY      = 64  * 1024
        self.L0C_CAPACITY      = 256 * 1024
        self.UB_CAPACITY       = 256 * 1024
        self.SB_CAPACITY       = 16  * 1024
        self.ABDFF_CAPACITY    = 512
        self.AccumDFF_CAPACITY = 512

        # I/O bandwidth (bytes/cycle)
        def tbps_to_bpc(tbps: float) -> float:
            return (tbps * 1e12) / self.CLOCK_FREQ
        def gbps_to_bpc(gbps: float) -> float:
            return (gbps * 1e9) / self.CLOCK_FREQ
        #这4个是每周期带宽，如果算原来的每秒还要用每周期带宽再乘回CLOCK_FREQ
        #改动
        # l2_to_l1_bpc   = tbps_to_bpc(4.07)
        # dram_to_l2_bpc = tbps_to_bpc(1.35)
        l2_to_l1_bpc   = tbps_to_bpc(0.290)
        dram_to_l1_bpc = tbps_to_bpc(0.120) #从130改成了
        bigger_dram_to_l1_bpc = tbps_to_bpc(0.100)#无奈下先把带宽调大试试！！！！！
        l1_to_l0a_bpc  = gbps_to_bpc(440.0)
        l1_to_l0b_bpc  = gbps_to_bpc(220.0)
        dram_to_l2_bpc = tbps_to_bpc(0.130) #占位值！！！
        mem1 = gbps_to_bpc(1600) #根据calculon实测

        self.IO_BW = {
            'MEM1': float(mem1),
            'Bigger DRAM→L1':float(bigger_dram_to_l1_bpc),
            'Real DRAM→L1':self.L2_CACHE_HIT_RATE*float(l2_to_l1_bpc)+(1-self.L2_CACHE_HIT_RATE)*float(dram_to_l1_bpc), #！！！！！
            'DRAM→L1': float(dram_to_l1_bpc),
            'DRAM→L2': float(dram_to_l2_bpc),
            'L2→L1'  : float(l2_to_l1_bpc),
            'L1→L0A' : float(l1_to_l0a_bpc),
            'L1→L0B' : float(l1_to_l0b_bpc),
            # 这些带宽不用先除以时钟频率，转换成每周期带宽吗？？？
            'AccumDFF→L0C': 210.0,
            'L0C→AccumDFF': 210.0,

            'L0C→L2' : 86.0, #占位值！！！
            'L2→DRAM': float(dram_to_l2_bpc),
            'DRAM→EXT': 32.0, # 占位值！！！
            'EXT→DRAM': 32.0, # 占位值！！！

            'L2→L0C' : 110.0,
            'L0A→L1' : float(self.MIN_ACCESS['L0A']),
            'L0B→L1' : float(self.MIN_ACCESS['L0B']),
            'L1→L0C' : float(self.MIN_ACCESS['L0C']),
            'L0C→L1' : 20.0,
            'L0C→UB' : float(self.MIN_ACCESS['L0C']),
            'UB→L0C' : float(self.MIN_ACCESS['L0C']),
            'UB→L1'  : float(self.MIN_ACCESS['L1']),
        }

        # L2 cache policy
        self.L2_ASSOCIATIVITY = 8
        self.L2_INPUT_RATIO   = 0.8
        self.L2_FIXED_HIT_RATE = 0.95
        self.DEFER_OUT1_TO_END = True    # ← 尾部一次性 OUT1
        self.DEFER_OUT2_TO_END = True    # ← 尾部一次性 OUT2
        self.GFLOPS_EFF_CURVE = [(128,0.98),(16,0.95),(1,0.80),(0,0.40)]
        self.MEM_MB_EFF_CURVE = [(128,0.98),(32,0.95),(8,0.90),(1,0.75),(0,0.50)]
        self.COMPUTE_EFF_BIAS = 1.10
        self.MEM_EFF_BIAS     = 1.20
HW = HardwareSpec()
