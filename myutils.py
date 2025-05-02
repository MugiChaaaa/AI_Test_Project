import torch

def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available using {torch.cuda.get_device_name(0)}")
        torch_mem = torch.cuda.mem_get_info(0)
        print('Memory Usage:', round((torch_mem [1] - torch_mem[0]) / 1024 ** 3, 1), '/',
              round(torch_mem[1] / 1024 ** 3, 1), f'GB, {round((torch_mem[1] - torch_mem[0]) / torch_mem[1] * 100, 2)}%')
    else:
        print("CUDA is not available")


def test_cuda():
    x = torch.rand(5, 3)
    print(x)
    return x


def pre_settings():
    print("### Pre Settings ###")
    print("####### DONE #######")