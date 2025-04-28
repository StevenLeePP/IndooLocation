import numpy as np

def process_srs_file(file_path):
    """
    处理二进制文件，提取数据并转换为复数形式的数组。

    参数:
        file_path (str): 二进制文件的路径。

    返回:
        processed_data (numpy.ndarray): 处理后的数组。
    """
    # 打开二进制文件
    with open(file_path, "rb") as fid1:
        # 读取所有数据，假设数据类型为 uint16
        fdata = np.fromfile(fid1, dtype=np.uint16)

    # 计算周期长度
    cycle_length = 2048 * 2 * 4 + 1

    # 计算周期数量
    num_cycles = len(fdata) // cycle_length
    # 初始化一个空列表来存储处理后的数据
    processed_data = []

    # 对每个周期进行处理
    for i in range(num_cycles):
        # 提取一个周期的数据
        cycle_data = fdata[i * cycle_length : (i + 1) * cycle_length]
        cycle_data = cycle_data[:-1]  # 去掉最后一个元素

        # 初始化复数形式的数组
        antenna_data_complex = np.empty((2048, 8), dtype=np.uint16)

        # 填充复数形式的数组
        for j in range(4):
            antenna_data_complex[:, j * 2] = cycle_data[j * 2048 * 2 : (j + 1) * 2048 * 2 : 2]  # 实部
            antenna_data_complex[:, j * 2 + 1] = cycle_data[j * 2048 * 2 + 1 : (j + 1) * 2048 * 2 : 2]  # 虚部

        # 保留奇数行
        antenna_data_complex = antenna_data_complex[::2, :]

        # 将处理后的数据添加到列表中
        processed_data.append(antenna_data_complex)

    # 将列表中的所有数据合并成一个大数组
    processed_data = np.concatenate(processed_data, axis=0)

    return processed_data
if __name__=="__main__":
    data = process_srs_file("/home/luhan/lap/IndooLocation/Data/srs-38points-1000time-1cycles/01/srs_get_01.bin")
    print(data[:20])