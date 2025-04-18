import pandas as pd
import os
import re
from tqdm import tqdm


for cycle in [10]:
    # 设置CSV文件所在的文件夹路径
    formatted_string = str(cycle).zfill(2)
    
    folder_path = r'E:\北邮2024\本科毕设\Model\Data\csv-5-10000-10\\'+formatted_string  # 替换为你的文件夹路径
    
    print("本轮处理文件夹：",folder_path[-2:])
    # 获取文件夹中所有的CSV文件
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # 计算每次处理的文件数量
    batch_size = len(csv_files) // 10

    # 读取第一个CSV文件，并获取其标题行
    first_csv = pd.read_csv(os.path.join(folder_path, csv_files[0]))
    headers = first_csv.columns.tolist()

    # 检查标题行是否符合预期
    if headers != ['序号', '天线1', '天线2', '天线3', '天线4']:
        raise ValueError("CSV文件的列名不符合预期，请检查文件格式")

    # 新的标题行
    new_headers = ['序号', '天线1实部', '天线1虚部', '天线2实部', '天线2虚部', '天线3实部', '天线3虚部', '天线4实部', '天线4虚部']

    # 使用for循环分批次处理文件
    for i in range(0, round(len(csv_files)), batch_size):
        # 获取当前批次的文件列表
        formatted_string_add = str((cycle-1)*10+(i//5000)+1)
        output_path = r'E:\北邮2024\本科毕设\Model\Data\csv-5-10000-10\test\merged_batch_'+formatted_string_add+r'.csv'
        batch_files = csv_files[i:i + batch_size]

        # 使用tqdm包装当前批次的文件列表，以显示进度条
        for file in tqdm(batch_files, desc=f"处理第 {(i // batch_size) + 1} 批文件", unit="文件"):
            position_number=file[2:4]
            df = pd.read_csv(os.path.join(folder_path, file))
            # 创建一个新的DataFrame来存储拆分后的数据
            new_df = pd.DataFrame(columns=new_headers)
            new_df['序号'] = [position_number] * len(df)
            # 拆分每一列的数据
            for col in df.columns[1:]:  # 跳过“序号”列
                df[col] = df[col].apply(lambda x: complex(x.replace('i', 'j')))  # 将字符串转换为复数
                new_df[f'{col}实部'] = df[col].apply(lambda x: x.real)  # 提取实部
                new_df[f'{col}虚部'] = df[col].apply(lambda x: x.imag)  # 提取虚部
            # 将当前文件的数据追加到CSV文件
            if i == 0:
                # 第一批次的第一个文件写入时包含标题行
                new_df.to_csv(output_path, mode='w', index=False, encoding='utf-8-sig')
            else:
                # 后续批次追加数据，不包含标题行
                new_df.to_csv(output_path, mode='a', index=False, header=False, encoding='utf-8-sig')

    print("所有批次处理完成")