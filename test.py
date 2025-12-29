import torch
import torch.nn as nn


def verify_backward_location():
    """验证 backward() 方法的位置"""

    import torch

    # 创建计算图
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2
    z = y.mean()

    print("验证 backward() 方法的位置:")
    print("=" * 60)

    # # 方法1：直接调用（最常用）
    # z.backward()
    # print(f"1. 直接调用: z.backward()")
    # print(f"   x.grad: {x.grad}")
    #
    # # 清空梯度
    # x.grad.zero_()

    # 方法2：通过 torch.autograd.backward
    # torch.autograd.backward(z)
    # print(f"\n2. 通过 torch.autograd.backward: torch.autograd.backward(z)")
    # print(f"   x.grad: {x.grad}")

    # # 清空梯度
    # x.grad.zero_()
    #
    # # 方法3：验证确实是 Tensor 的方法
    # print(f"\n3. 验证方法归属:")
    # print(f"   hasattr(z, 'backward'): {hasattr(z, 'backward')}")
    # print(f"   hasattr(torch, 'backward'): {hasattr(torch, 'backward')}")
    # print(f"   hasattr(torch.autograd, 'backward'): {hasattr(torch.autograd, 'backward')}")
    #
    # 方法4：查看方法的实际位置
    print(f"\n4. 方法实际位置:")
    print(f"   z.backward 的类型: {type(z.backward)}")
    print(f"   z.backward 所在模块: {z.backward.__module__}")
    #
    # # 方法5：等价调用
    # print(f"\n5. 等价调用关系:")
    # print(f"   z.backward() 等价于 torch.autograd.backward(z)")
    # print(f"   但参数传递方式不同")
    #
    # # 演示参数传递差异
    # x = torch.tensor([1.0, 2.0], requires_grad=True)
    # y = torch.tensor([3.0, 4.0], requires_grad=True)
    # z = x * y
    #
    # print(f"\n6. 非标量情况下的差异:")
    #
    # # 对于非标量，需要指定 gradient 参数
    # # 方法 A: Tensor.backward()
    # grad_output = torch.tensor([1.0, 1.0])
    # z.backward(gradient=grad_output)
    # print(f"   A. z.backward(gradient=torch.tensor([1.0, 1.0]))")
    # print(f"      x.grad: {x.grad}")
    #
    # x.grad = None
    # y.grad = None
    #
    # # 方法 B: torch.autograd.backward()
    # torch.autograd.backward(z, grad_tensors=grad_output)
    # print(f"   B. torch.autograd.backward(z, grad_tensors=torch.tensor([1.0, 1.0]))")
    # print(f"      x.grad: {x.grad}")


verify_backward_location()