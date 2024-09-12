import os
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def split_ssl_multilabel_data(args, data, targets, num_classes,
                              lb_num_labels, ulb_num_labels=None,
                              lb_index=None, ulb_index=None, include_lb_to_ulb=True, load_exist=True):
    """
    Split the data into labeled and unlabeled data for multi-label classification.

    Args:
        args: 其他参数配置
        data: 图像路径的列表
        targets: 图像标签的列表，每一个元素是一个list，是多标签，因此如果属于某个类别，那么对应的位置是1，否则是0
        num_classes: 类别数
        lb_num_labels: labeled data的数量
        ulb_num_labels: unlabeled data的数量
        lb_index: labeled data的索引
        ulb_index: unlabeled data的索引
        include_lb_to_ulb: 是否将labeled data包含到unlabeled data中
        load_exist: 是否加载已经存在的数据

    Returns: lb_data, lb_targets, ulb_data, ulb_targets
    """
    data, targets = np.array(data), np.array(targets)

    # 如果 load_exist 为 True，则尝试加载已经存在的索引文件
    dump_dir = os.path.join(base_dir, 'data', args.dataset, 'multilabel_idx')
    os.makedirs(dump_dir, exist_ok=True)
    print("=============label number: =====================", lb_num_labels)
    print("=============unlabel number: =====================", ulb_num_labels)
    lb_dump_path = os.path.join(dump_dir, f'lb_multilabel_{lb_num_labels}_seed{args.seed}_idx.npy')
    ulb_dump_path = os.path.join(dump_dir, f'ulb_multilabel_{lb_num_labels}_seed{args.seed}_idx.npy')

    if os.path.exists(lb_dump_path) and os.path.exists(ulb_dump_path) and load_exist:
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)
    else:
        lb_idx = set()
        remaining_indices = set(np.arange(len(data)))

        # 分层抽样以确保每个类别都有至少一个标记样本
        for class_idx in range(num_classes):
            class_indices = np.where(targets[:, class_idx] == 1)[0]
            np.random.shuffle(class_indices)
            lb_count = min(len(class_indices), 1)  # 确保每个类别至少有一个标记样本
            lb_idx.update(class_indices[:lb_count])
            remaining_indices.difference_update(class_indices[:lb_count])

        lb_idx = np.array(list(lb_idx))

        # 如果标记数据数量不足，随机补充
        if len(lb_idx) < lb_num_labels:
            additional_lb_count = lb_num_labels - len(lb_idx)
            additional_lb_indices = np.random.choice(list(remaining_indices), additional_lb_count, replace=False)
            lb_idx = np.concatenate([lb_idx, additional_lb_indices])
            remaining_indices.difference_update(additional_lb_indices)

        ulb_idx = np.array(list(remaining_indices))
        if ulb_num_labels:
            ulb_idx = np.random.choice(ulb_idx, ulb_num_labels, replace=False)
        else:
            ulb_idx = np.array(list(remaining_indices))

        if include_lb_to_ulb:
            ulb_idx = np.concatenate([lb_idx, ulb_idx])

        np.save(lb_dump_path, lb_idx)
        np.save(ulb_dump_path, ulb_idx)

    # 如果手动提供了索引，则使用提供的索引
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]

