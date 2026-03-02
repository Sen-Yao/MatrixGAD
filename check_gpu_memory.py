import torch
import gc

def print_gpu_memory_usage(prefix=""):
    """打印当前GPU内存使用情况"""
    if torch.cuda.is_available():
        # 当前设备
        device = torch.cuda.current_device()

        # 获取已分配的显存
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        # 获取缓存的显存
        cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
        # 获取最大显存使用
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB

        print(f"{prefix}GPU显存使用情况:")
        print(f"  当前已分配: {allocated:.2f} GB")
        print(f"  当前缓存: {cached:.2f} GB")
        print(f"  最大已分配: {max_allocated:.2f} GB")
        print(f"  设备总显存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        print(f"  设备名: {torch.cuda.get_device_name(device)}")

        # 重置最大内存跟踪
        torch.cuda.reset_peak_memory_stats(device)
    else:
        print("CUDA不可用")

def print_tensor_memory():
    """打印所有张量的内存占用"""
    if torch.cuda.is_available():
        print("\n张量内存占用详情:")
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    print(f"  张量形状: {obj.shape}, 类型: {obj.dtype}, 显存: {obj.numel() * obj.element_size() / 1024**2:.2f} MB")

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("已清理GPU缓存和执行垃圾回收")

