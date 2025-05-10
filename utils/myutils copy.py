import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import subprocess
import os

class logger:
    def __init__(self, name):
        self.name = name
        self.info_stream = ''

    def run(self, cmd):
        result = subprocess.run([cmd], shell= True, capture_output=True, text=True)
        if result.stdout != '':
            str = '[INFO]:\n' + result.stdout + "\n"
            self.info_stream += str
            print(str)
        if result.stderr != '':
            str = '[ERROR]:' + result.stderr + "\n"
            self.info_stream += str
            print(str)

    def write2file(self, dir):
        with open(dir+f"/{self.name}.log", "w") as log_file:
            log_file.write(self.info_stream)
        print(f"save {self.name}.logger to {dir}")
        
def tensor_to_2d_map(tensor):
    """
    将一个形状为 (n*n, 1) 的 tensor 转换为二维数组 map[n][n]。
    遍历 tensor 向量，按顺序填充 map，从 map[0][0] 到 map[n-1][n-1]。

    参数:
    - tensor: 输入 tensor，形状为 (n*n, 1)

    返回:
    - map: 形状为 (n, n) 的二维数组 (torch.Tensor)
    """
    # 确保输入 tensor 是 (n*n, 1)
    assert len(tensor.shape) == 2 and tensor.shape[1] == 1, "输入 tensor 的形状必须为 (n*n, 1)"

    # 根据 tensor 的大小推断 n
    total_elements = tensor.shape[0]
    n = int(total_elements ** 0.5)
    assert n * n == total_elements, "输入 tensor 的长度不是完全平方数，无法转换为二维数组"

    # 重塑 tensor 为 (n, n)
    map_2d = tensor.view(n, n)
    return map_2d

def plot_Map(data, title, display_range=None):
    """
    绘制二维数据的热图，并将原点坐标设置在左下角，交换X和Y坐标
    :param data: 二维数组数据
    :param title: 图表标题
    :param display_range: 显示的范围，格式为 (min, max)，如果为 None，则自动确定范围
    :return: 实际显示的范围
    """
    data = np.array(data)

    # 设置绘图风格
    plt.figure(figsize=(12, 10))

    # 使用Seaborn绘制热图
    if display_range is not None:
        vmin, vmax = display_range
    else:
        vmin, vmax = data.min(), data.max()
    
    ax = sns.heatmap(data, cmap='YlGnBu', annot=False, fmt='.2f', cbar=True, vmin=vmin, vmax=vmax)

    # 设置坐标轴的方向：将原点移动到左下角
    plt.gca().invert_yaxis()  # 反转Y轴，使得原点在左下角
    
    
    # 添加标题和标签（X和Y对调后）
    plt.title(title, fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)  # X轴变为Y
    plt.ylabel('Y Coordinate', fontsize=12)  # Y轴变为X

    

    # 显示图形
    plt.show()

    return (vmin, vmax)

def vector2txt(vector, file_path):
    """
    将浮点数向量保存到一个文本文件中，每个元素占一行。

    参数:
    - vector (list or np.ndarray): 浮点数向量
    - file_path (str): 保存的文件路径

    返回:
    - None
    """
    try:
        with open(file_path, 'w') as f:
            for value in vector:
                f.write(f"{value}\n")  # 每个元素占一行
        print(f"Vector successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_line_graph(vector, title, x_label, y_label, save_path=None):
    """
    绘制折线图并显示或保存到文件。

    参数:
    - vector (list or np.ndarray): 要绘制的向量数据
    - title (str): 图的标题
    - x_label (str): X轴的标签
    - y_label (str): Y轴的标签
    - save_path (str): 可选，保存图像的文件路径（如 'output.png'）。若为 None，则直接显示图像。

    返回:
    - None
    """
    try:
        plt.figure(figsize=(8, 6))  # 设置图形大小
        plt.plot(vector, marker='o', linestyle='-', color='b', label='Data')  # 折线图
        plt.title(title)  # 设置标题
        plt.xlabel(x_label)  # 设置X轴标签
        plt.ylabel(y_label)  # 设置Y轴标签
        plt.grid(True)  # 添加网格线
        plt.legend()  # 添加图例

        if save_path:  # 如果提供了保存路径
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:  # 显示图形
            plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # PyTorch 随机性
    torch.cuda.manual_seed(seed)  # PyTorch CUDA 随机性
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    random.seed(seed)

def plot_Map_B(data, title, save_dir=None):
    """
    绘制二值数据的图，0用白色表示，1用黑色表示
    Args:
        data: 二维数组数据，只包含0和1
        title: 图表标题
        save_dir: 保存图片的目录路径，如果为None则只显示不保存
    """
    data = np.array(data)
    
    plt.figure(figsize=(12, 10))
    
    # 使用binary colormap (黑白)，白色表示0，黑色表示1
    ax = plt.imshow(data, cmap='binary', aspect='equal')
    
    # 设置坐标轴的方向：将原点移动到左下角
    plt.gca().invert_yaxis()
    
    # 添加标题和标签
    # plt.title(title, fontsize=16)
    # plt.xlabel('X Coordinate', fontsize=12)
    # plt.ylabel('Y Coordinate', fontsize=12)
    
    # 保存图片
    if save_dir:
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        # 将标题中的空格替换为下划线，作为文件名
        filename = title.replace(' ', '_') + '.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # 显示图形
    plt.show()
    
    # 关闭图形，释放内存
    plt.close()

def plot_edge_map(tensor_input, title, save_dir=None):
    length = int(tensor_input.shape[0])
    
    # Find the closest n where 4*n*(n+1) is less than or equal to length
    n = 1
    while 4*n*(n+1) <= length:
        n += 1
    n = n - 1  # Go back to last valid n
    
    required_length = 4*n*(n+1)
    if required_length < length:
        print(f"Warning: Using first {required_length*2} elements of tensor (n={n})")

    gcell_length = n + 1
    # Create n x (n+1) map
    map_2d_horizontal = tensor_input[:gcell_length*(gcell_length-1)*2].cpu()
    # Create (n+1) x n map
    map_2d_vertical = tensor_input[gcell_length*(gcell_length-1)*2:len(tensor_input)+1].cpu()
    assert map_2d_horizontal.shape == map_2d_vertical.shape
    
    avg_ef_horizontal = torch.zeros(gcell_length, gcell_length-1)
    avg_ef_vertical = torch.zeros(gcell_length-1, gcell_length)

    # horizental    
    for y in range(gcell_length):
        for x in range(gcell_length - 1):
            avg_ef_horizontal[y][x] = (map_2d_horizontal[y*(gcell_length-1)*2+x*2] + map_2d_horizontal[y*(gcell_length-1)*2+x*2+1])/2
            #test
            # assert map_2d_horizontal[y*(gcell_length-1)*2+x*2] == map_2d_horizontal[y*(gcell_length-1)*2+x*2+1]
            
    # vertical
    for x in range(gcell_length):
        for y in range(gcell_length - 1):
            avg_ef_vertical[y][x] = (map_2d_vertical[x*(gcell_length-1)*2+y*2] + map_2d_vertical[x*(gcell_length-1)*2+y*2+1])/2
            #test
            # assert map_2d_vertical[x*(gcell_length-1)*2+y*2] == map_2d_vertical[x*(gcell_length-1)*2+y*2+1]

    # Plot the maps
    plt.figure(figsize=(12, 5))
    
    # Plot horizontal map
    plt.subplot(1, 2, 1)
    ax1 = sns.heatmap(avg_ef_horizontal, cmap='YlGnBu', annot=False, fmt='.2f', cbar=True)
    plt.gca().invert_yaxis()
    plt.title('Horizontal '+title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Plot vertical map
    plt.subplot(1, 2, 2)
    ax2 = sns.heatmap(avg_ef_vertical, cmap='YlGnBu', annot=False, fmt='.2f', cbar=True)
    plt.gca().invert_yaxis()
    plt.title('Vertical '+title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    plt.tight_layout()
    
    # Save figure if save_dir is provided
    if save_dir:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        # Replace spaces with underscores in title for filename
        filename = title.replace(' ', '_') + '_edge_map.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    plt.close()
    
    return avg_ef_horizontal, avg_ef_vertical

def avg_OF_result(tensor_input, is_plot=False):
    """优化版本的avg_OF_result函数，减少CPU-GPU数据传输和提高计算效率"""
    device = tensor_input.device
    length = tensor_input.shape[0]
    
    # 计算n值
    n = int(np.sqrt(length/4 - 0.25) - 0.5)  # 解方程 4*n*(n+1) = length
    
    # 快速检查n值是否合理
    required_length = 4*n*(n+1)
    if required_length != length:
        print(f"Warning: Input length {length} doesn't match 4*n*(n+1). Using n={n}")
    
    gcell_length = n + 1
    h_length = gcell_length * (gcell_length-1) * 2
    
    # 直接在GPU上分割数据，避免不必要的CPU传输
    map_2d_horizontal = tensor_input[:h_length]
    map_2d_vertical = tensor_input[h_length:2*h_length]
    
    # 使用向量化操作代替循环
    # 创建索引张量
    h_indices = torch.arange(0, h_length, 2, device=device)
    v_indices = torch.arange(0, h_length, 2, device=device)
    
    # 使用高效的向量化操作计算平均值
    h_values = (map_2d_horizontal[h_indices] + map_2d_horizontal[h_indices+1]) / 2
    v_values = (map_2d_vertical[v_indices] + map_2d_vertical[v_indices+1]) / 2
    
    # 重塑为所需的形状
    avg_ef_horizontal = h_values.reshape(gcell_length, gcell_length-1)
    avg_ef_vertical = v_values.reshape(gcell_length-1, gcell_length)
    
    # 如果需要二值化（用于分类任务）
    if not is_plot:  # 只在非绘图模式下二值化
        avg_ef_horizontal = (avg_ef_horizontal > 0).float()
        avg_ef_vertical = (avg_ef_vertical > 0).float()
    
    return avg_ef_horizontal, avg_ef_vertical

