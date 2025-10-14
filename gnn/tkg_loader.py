import pandas as pd
import os
import re
import numpy as np

def parse_year_from_messy_string(s):
    if not isinstance(s, str):
        return np.nan
    
    # 查找字符串中的第一个四位数序列
    match = re.search(r'\d{4}', s)
    if match:
        year = int(match.group(0))
        # 进行一个简单的合理性检查，避免像 '0000' 这样的年份
        if 1000 < year < 3000:
            return float(year)
    return np.nan

def load_all_tkg_data(data_dir: str = '/root/tkgc_data/share/ind-YAGO11k/'):
    filepaths = {
        'train': os.path.join(data_dir, 'train.txt'),
        'valid': os.path.join(data_dir, 'valid.txt'),
        'test': os.path.join(data_dir, 'test.txt')
    }
    all_dfs = {}
    
    for split, path in filepaths.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep='\t', header=None, names=['s', 'r', 'o', 'start_str', 'end_str'])
                if not df.empty:
                    df['split'] = split
                    all_dfs[split] = df
            except pd.errors.EmptyDataError:
                print(f"警告: 数据文件 '{path}' 为空，已跳过。")
        else:
            print(f"警告: 未找到数据集文件 '{path}'。")

    if not all_dfs:
        raise FileNotFoundError(f"错误: 在目录 '{os.path.abspath(data_dir)}' 中未找到任何有效的、非空的数据文件。")

    full_df = pd.concat(all_dfs.values(), ignore_index=True)
    

    full_df['start_time'] = full_df['start_str'].apply(parse_year_from_messy_string)
    full_df['end_time'] = full_df['end_str'].apply(parse_year_from_messy_string)
    
    full_df['end_time'] = full_df['end_time'].fillna(full_df['start_time'])
    
    full_df.dropna(subset=['start_time'], inplace=True)
    full_df = full_df.astype({"start_time": int, "end_time": int})

    entities = sorted(list(set(full_df['s']) | set(full_df['o'])))
    relations = sorted(list(set(full_df['r'])))
    entity_map = {name: i for i, name in enumerate(entities)}
    relation_map = {name: i for i, name in enumerate(relations)}

    full_df['s_id'] = full_df['s'].map(entity_map)
    full_df['o_id'] = full_df['o'].map(entity_map)
    full_df['r_id'] = full_df['r'].map(relation_map)
    
    t_min = full_df['start_time'].min()
    t_max = full_df['end_time'].max()

    if pd.isna(t_min) or pd.isna(t_max):
        raise ValueError("错误: 即使在清理后，仍然无法从数据中解析出有效的时间范围。")
        
    # 3. 分割回训练/验证/测试 DataFrame
    processed_dfs = {
        split: full_df[full_df['split'] == split].drop(columns=['split']).reset_index(drop=True)
        for split in filepaths.keys() if split in full_df['split'].unique()
    }

    return (
        processed_dfs.get('train'), 
        processed_dfs.get('valid'), 
        processed_dfs.get('test'), 
        entity_map, 
        relation_map, 
        (int(t_min), int(t_max))
    )