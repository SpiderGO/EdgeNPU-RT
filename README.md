# EdgeNPU-RT

EdgeNPU-RT 是一个面向资源受限边缘设备的轻量级 AI Runtime 与 FPGA 加速平台。

# 项目目标是打通：

```text
C++ Runtime
    ↓
Linux Driver
    ↓
MMIO / DMA / Interrupt
    ↓
FPGA Matrix Accelerator
```
# 项目阶段
## Phase 1：软件模拟后端
### 目标平台：
- x86_64 Linux
- i.MX6U ARM Linux
### 功能：
- Tensor 和 Buffer 管理
- GEMM 命令提交
- Linux 字符设备
- ioctl / mmap / poll
- CPU 软件模拟设备
## Phase 2：Zynq FPGA 后端
### 计划平台：
- Zynq-7020
### 计划功能：
- AXI-Lite 控制寄存器
- AXI DMA 数据传输
- FPGA GEMM 加速器
- 中断完成通知
### 应用验证
暂定以多通道压力传感器数据为输入，构建轻量级抓握状态识别模型，验证边缘 AI 推理链路。
### 当前状态
- [√] Runtime 基础接口
- [√] 软件模拟设备
- [ ] Linux 字符设备驱动
- [ ] i.MX6U 交叉编译
- [ ] Zynq FPGA GEMM
- [ ] AXI DMA
- [ ] 中断通知
- [ ] 压力传感器 MLP Demo