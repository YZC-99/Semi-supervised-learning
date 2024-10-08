from semilearn.nets import densenet121
from semilearn.datasets.cv_datasets.isic2018 import ISIC2018Dataset
import torchvision.transforms as transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
import math
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np
matplotlib.use('Agg')


def load_model(ckpt_path, num_classes):
    """加载模型并移除backbone前缀"""
    model = densenet121(num_classes)
    ckpt_dict = torch.load(ckpt_path, map_location='cpu')['model']
    # 去除backbone前缀
    ckpt_dict = {k.replace('backbone.', ''): v for k, v in ckpt_dict.items()}
    model.load_state_dict(ckpt_dict, strict=False)
    model = model.cuda()
    model.eval()
    return model


def prepare_data(csv_dir, data_dir, num_classes, img_size=224):
    """准备数据集和DataLoader"""
    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    test_all_info = pd.read_csv(f"{csv_dir}val_dataset.csv")
    # test_all_info = pd.read_csv(f"{csv_dir}train_dataset.csv")
    test_all_info = test_all_info.fillna(0)
    test_data = test_all_info.iloc[:, 0].values
    test_data = [data_dir + i + '.jpg' for i in test_data]
    test_targets = test_all_info.iloc[:, 1:num_classes + 1].values
    test_dset = ISIC2018Dataset('hyperplusfixmatchv3', test_data, test_targets, num_classes, transform_val, False, strong_transform=None,
                                onehot=False, is_test=True)
    dataloader = DataLoader(test_dset, batch_size=16, shuffle=False, num_workers=4)

    return dataloader


def prepare_data_wStrong(csv_dir, data_dir, num_classes, img_size=224):
    """准备数据集和DataLoader"""
    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])
    crop_ratio = 0.875
    transform_strong = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    test_all_info = pd.read_csv(f"{csv_dir}val_dataset.csv")
    # test_all_info = pd.read_csv(f"{csv_dir}train_dataset.csv")
    test_all_info = test_all_info.fillna(0)
    test_data = test_all_info.iloc[:, 0].values
    test_data = [data_dir + i + '.jpg' for i in test_data]
    test_targets = test_all_info.iloc[:, 1:num_classes + 1].values
    test_dset = ISIC2018Dataset('hyperplusfixmatchv3', test_data, test_targets, num_classes, transform_val, False, strong_transform=None,
                                onehot=False, is_test=True,test_transform=transform_strong)
    dataloader = DataLoader(test_dset, batch_size=16, shuffle=False, num_workers=4)

    return dataloader


def plot_continue_prob(dataloader, model, num_classes, save_path='ours_isic2018_val_probability_distribution.png'):
    """绘制不同概率区间内的正确和错误样本数量"""
    bins = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
            0.90, 0.95, 1.00]
    correct_counts = [0] * (len(bins) - 1)
    incorrect_counts = [0] * (len(bins) - 1)
    results = []

    # 遍历dataloader
    for batch in dataloader:
        image_paths, x_lb, y_lb = batch['image_path'], batch['x_lb'], batch['y_lb']
        x_lb = x_lb.cuda()
        y_lb = y_lb.cuda()

        with torch.no_grad():
            logits = model(x_lb)['logits']
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

        # 统计并记录结果
        for i, path in enumerate(image_paths):
            pred_prob = probs[i][preds[i]].item()  # 模型预测类别的概率
            pred = preds[i].cpu().item()  # 预测值
            true_label = y_lb[i].cpu().numpy().tolist()  # 真实标签（one-hot编码）
            results.append([path] + probs[i].cpu().tolist() + [pred] + true_label)

            # 统计预测正确与错误的概率分布，只统计预测类别的概率
            bin_idx = min(len(bins) - 2, int(pred_prob // 0.05))  # 使用预测类别的概率
            if pred == true_label.index(1):  # 如果预测值与真实值相同
                correct_counts[bin_idx] += 1
            else:
                incorrect_counts[bin_idx] += 1

    # 将结果存储到 DataFrame 中
    columns = ['Image Path'] + [f'Class_{i}_Prob' for i in range(num_classes)] + ['Prediction'] + [f'True_Class_{i}' for
                                                                                                   i in
                                                                                                   range(num_classes)]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv('model_predictions.csv', index=False)

    # 只保留有数据的区间
    filtered_x_ticks = []
    filtered_correct_counts = []
    filtered_incorrect_counts = []
    for i in range(len(bins) - 1):
        if correct_counts[i] > 0 or incorrect_counts[i] > 0:  # 过滤掉没有数据的区间
            filtered_x_ticks.append(f'{bins[i]:.2f}-{bins[i + 1]:.2f}')
            filtered_correct_counts.append(correct_counts[i])
            filtered_incorrect_counts.append(incorrect_counts[i])

    # 绘图
    plt.figure(figsize=(10, 6.18))
    plt.bar(filtered_x_ticks, filtered_correct_counts, label='Correct', alpha=0.7)
    plt.bar(filtered_x_ticks, filtered_incorrect_counts, bottom=filtered_correct_counts, label='Incorrect', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Probability Range')
    plt.ylabel('Number of Samples')
    plt.ylim(0, 700)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_multi_continue_prob(dataloader, models, model_names, num_classes,
                             save_path='multi_model_prob_distribution.svg', csv_path='probability_distribution.csv', show_counts=False):
    """绘制不同模型的正确和错误样本分布在同一幅图上，控制是否显示数量，且保存为 CSV 文件"""

    bins = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
            0.90, 0.95, 1.00]

    # 准备为每个模型记录数据的结构
    model_correct_counts = {model_name: [0] * (len(bins) - 1) for model_name in model_names}
    model_incorrect_counts = {model_name: [0] * (len(bins) - 1) for model_name in model_names}

    # 颜色列表，使用同一色系，深色代表正确，浅色代表错误
    colors = ['blue', 'green', 'orange', 'purple', 'brown']  # 为每个模型指定一个主色调

    # 遍历dataloader
    for batch in dataloader:
        image_paths, x_lb, y_lb = batch['image_path'], batch['x_lb'], batch['y_lb']
        x_lb = x_lb.cuda()
        y_lb = y_lb.cuda()

        # 针对每个模型计算预测
        for model, model_name in zip(models, model_names):
            with torch.no_grad():
                logits = model(x_lb)['logits']
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

            # 统计并记录结果
            for i, path in enumerate(image_paths):
                pred_prob = probs[i][preds[i]].item()  # 模型预测类别的概率
                pred = preds[i].cpu().item()  # 预测值
                true_label = y_lb[i].cpu().numpy().tolist()  # 真实标签（one-hot编码）

                # 统计预测正确与错误的概率分布，只统计预测类别的概率
                bin_idx = min(len(bins) - 2, int(pred_prob // 0.05))  # 使用预测类别的概率
                if pred == true_label.index(1):  # 如果预测值与真实值相同
                    model_correct_counts[model_name][bin_idx] += 1
                else:
                    model_incorrect_counts[model_name][bin_idx] += 1

    # 过滤掉所有模型中没有样本的区间
    filtered_bins = []
    filtered_correct_counts = {model_name: [] for model_name in model_names}
    filtered_incorrect_counts = {model_name: [] for model_name in model_names}

    for i in range(len(bins) - 1):
        # 检查当前区间在所有模型中是否有样本
        if any(model_correct_counts[model_name][i] > 0 or model_incorrect_counts[model_name][i] > 0 for model_name in model_names):
            filtered_bins.append(f'{bins[i]:.2f}-{bins[i + 1]:.2f}')
            for model_name in model_names:
                filtered_correct_counts[model_name].append(model_correct_counts[model_name][i])
                filtered_incorrect_counts[model_name].append(model_incorrect_counts[model_name][i])

    # 保存数据为 CSV 文件
    csv_data = {}
    for model_name in model_names:
        csv_data[f'{model_name}_Correct'] = filtered_correct_counts[model_name]
        csv_data[f'{model_name}_Incorrect'] = filtered_incorrect_counts[model_name]

    csv_df = pd.DataFrame(csv_data, index=filtered_bins)
    csv_df.to_csv(csv_path)
    print(f"Probability distribution data saved to {csv_path}")

    # 绘图
    plt.figure(figsize=(12, 8))

    # 为每个模型绘制柱状图
    bar_width = 0.3  # 每个模型的柱状图宽度
    for idx, model_name in enumerate(model_names):
        # 计算柱状图的位置偏移量
        x_positions = [i + idx * bar_width for i in range(len(filtered_bins))]

        # 使用主色调绘制正确和错误分布，正确为深色，错误为浅色
        bars_correct = plt.bar(x_positions, filtered_correct_counts[model_name], width=bar_width,
                               label=f'{model_name} Correct',
                               color=colors[idx], alpha=0.8)
        bars_incorrect = plt.bar(x_positions, filtered_incorrect_counts[model_name],
                                 bottom=filtered_correct_counts[model_name],
                                 width=bar_width, label=f'{model_name} Incorrect', color=colors[idx], alpha=0.4)

        # 根据 show_counts 参数控制是否显示数量
        if show_counts:
            for bar_correct, bar_incorrect, correct_count, incorrect_count in zip(bars_correct, bars_incorrect,
                                                                                  filtered_correct_counts[model_name],
                                                                                  filtered_incorrect_counts[model_name]):
                height_correct = bar_correct.get_height()
                height_incorrect = bar_incorrect.get_height() + height_correct  # 堆叠后的总高度
                plt.text(bar_correct.get_x() + bar_correct.get_width() / 2., height_correct, f'{correct_count}',
                         ha='center', va='bottom', fontsize=10)
                plt.text(bar_incorrect.get_x() + bar_incorrect.get_width() / 2., height_incorrect, f'{incorrect_count}',
                         ha='center', va='bottom', fontsize=10)

    plt.xticks([i + (len(model_names) - 1) * bar_width / 2 for i in range(len(filtered_bins))], filtered_bins,
               rotation=45, ha='right')
    plt.xlabel('Probability Range')
    plt.ylabel('Number of Samples')
    plt.ylim(0, 700)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()




def plot_multi_incorrect_per_class(dataloader, models, model_names, num_classes, save_path='multi_model_incorrect_per_class.png', csv_path='incorrect_rates.csv'):
    """绘制多个模型在阈值大于等于0.95时，每个类别的错误率，横坐标为类别索引，纵坐标为错误率，并保存为 CSV 文件"""

    # 初始化每个模型的统计数据
    model_incorrect_counts = {model_name: [0] * num_classes for model_name in model_names}
    model_total_counts = {model_name: [0] * num_classes for model_name in model_names}

    # 遍历dataloader
    for batch in dataloader:
        image_paths, x_lb, y_lb = batch['image_path'], batch['x_lb'], batch['y_lb']
        x_lb = x_lb.cuda()
        y_lb = y_lb.cuda()

        # 针对每个模型计算预测
        for model, model_name in zip(models, model_names):
            with torch.no_grad():
                logits = model(x_lb)['logits']
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

            # 统计每个类别的错误率
            for i in range(len(image_paths)):
                pred_prob = probs[i][preds[i]].item()  # 模型预测类别的概率
                pred = preds[i].cpu().item()  # 预测值
                true_label = y_lb[i].cpu().tolist().index(1)  # 真实类别的索引

                # 记录大于等于0.95阈值的样本
                if pred_prob >= 0.95:
                    model_total_counts[model_name][true_label] += 1  # 记录该类别的总样本数
                    if pred != true_label:  # 预测错误
                        model_incorrect_counts[model_name][true_label] += 1

    # 计算每个类别的错误率
    model_incorrect_rates = {}
    for model_name in model_names:
        incorrect_rates = []
        for i in range(num_classes):
            if model_total_counts[model_name][i] > 0:
                incorrect_rate = model_incorrect_counts[model_name][i] / model_total_counts[model_name][i]
            else:
                incorrect_rate = 0.0
            incorrect_rates.append(incorrect_rate)
        model_incorrect_rates[model_name] = incorrect_rates

    # Debug: 输出每个模型的错误率和样本统计以供验证
    for model_name in model_names:
        print(f"Model: {model_name}")
        print(f"Total samples per class: {model_total_counts[model_name]}")
        print(f"Incorrect samples per class: {model_incorrect_counts[model_name]}")
        print(f"Incorrect rates per class: {model_incorrect_rates[model_name]}")
        print("\n")

    # 保存数据为 CSV 文件
    df = pd.DataFrame(model_incorrect_rates)
    df.index = [f'Class {i}' for i in range(num_classes)]  # 设置行标签为类别索引
    df.to_csv(csv_path, index=True)
    print(f"Incorrect rates data saved to {csv_path}")

    # 绘图
    plt.figure(figsize=(12, 8))

    # 颜色列表
    colors = ['blue', 'green', 'orange', 'purple', 'brown']  # 为每个模型指定颜色

    # 为每个模型绘制折线图
    for idx, model_name in enumerate(model_names):
        plt.plot(range(num_classes), model_incorrect_rates[model_name], marker='o', label=model_name, color=colors[idx])

    plt.xticks(range(num_classes), [f'Class {i}' for i in range(num_classes)])
    plt.xlabel('Class Index')
    plt.ylabel('Error Rate')
    plt.ylim(0, 1)  # 错误率的范围是0到1
    plt.title('Error Rates Per Class for Models with Threshold ≥ 0.95')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_multi_continue_prob_loss(dataloader, models, model_names, num_classes, criterion,
                                  save_path='multi_model_prob_distribution_loss.svg',
                                  csv_path='probability_distribution_loss.csv', show_counts=False):
    """绘制不同模型的正确和错误样本分布，并计算每个概率区间的平均交叉熵损失，显示为折线图"""

    bins = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
            0.90, 0.95, 1.00]

    # 准备为每个模型记录数据的结构
    model_correct_counts = {model_name: [0] * (len(bins) - 1) for model_name in model_names}
    model_incorrect_counts = {model_name: [0] * (len(bins) - 1) for model_name in model_names}
    model_correct_losses = {model_name: [[] for _ in range(len(bins) - 1)] for model_name in model_names}
    model_incorrect_losses = {model_name: [[] for _ in range(len(bins) - 1)] for model_name in model_names}

    # 颜色列表，使用同一色系，深色代表正确，浅色代表错误
    colors = ['blue', 'green', 'orange', 'purple', 'brown']  # 为每个模型指定一个主色调

    # 遍历dataloader
    for batch in dataloader:
        image_paths, x_lb, y_lb = batch['image_path'], batch['x_lb'], batch['y_lb']
        x_lb = x_lb.cuda()
        y_lb = y_lb.cuda()

        # 针对每个模型计算预测
        for model, model_name in zip(models, model_names):
            with torch.no_grad():
                logits = model(x_lb)['logits']
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                # 计算损失
                losses = criterion(logits, y_lb.to(torch.float32))

            # 统计并记录结果
            for i, path in enumerate(image_paths):
                pred_prob = probs[i][preds[i]].item()  # 模型预测类别的概率
                pred = preds[i].cpu().item()  # 预测值
                true_label = y_lb[i].cpu().numpy().tolist()  # 真实标签（one-hot编码）

                # 统计预测正确与错误的概率分布，只统计预测类别的概率
                bin_idx = min(len(bins) - 2, int(pred_prob // 0.05))  # 使用预测类别的概率
                if pred == true_label.index(1):  # 如果预测值与真实值相同
                    model_correct_counts[model_name][bin_idx] += 1
                    model_correct_losses[model_name][bin_idx].append(losses[i].item())
                else:
                    model_incorrect_counts[model_name][bin_idx] += 1
                    model_incorrect_losses[model_name][bin_idx].append(losses[i].item())

    # 计算每个区间的平均损失
    model_correct_avg_loss = {model_name: [np.mean(losses) if losses else 0 for losses in model_correct_losses[model_name]] for model_name in model_names}
    model_incorrect_avg_loss = {model_name: [np.mean(losses) if losses else 0 for losses in model_incorrect_losses[model_name]] for model_name in model_names}

    # 过滤掉所有模型中没有样本的区间
    filtered_bins = []
    filtered_correct_counts = {model_name: [] for model_name in model_names}
    filtered_incorrect_counts = {model_name: [] for model_name in model_names}
    filtered_correct_avg_loss = {model_name: [] for model_name in model_names}
    filtered_incorrect_avg_loss = {model_name: [] for model_name in model_names}

    for i in range(len(bins) - 1):
        # 检查当前区间在所有模型中是否有样本
        if any(model_correct_counts[model_name][i] > 0 or model_incorrect_counts[model_name][i] > 0 for model_name in model_names):
            filtered_bins.append(f'{bins[i]:.2f}-{bins[i + 1]:.2f}')
            for model_name in model_names:
                filtered_correct_counts[model_name].append(model_correct_counts[model_name][i])
                filtered_incorrect_counts[model_name].append(model_incorrect_counts[model_name][i])
                filtered_correct_avg_loss[model_name].append(model_correct_avg_loss[model_name][i])
                filtered_incorrect_avg_loss[model_name].append(model_incorrect_avg_loss[model_name][i])

    # 保存数据为 CSV 文件
    csv_data = {}
    for model_name in model_names:
        csv_data[f'{model_name}_Correct'] = filtered_correct_counts[model_name]
        csv_data[f'{model_name}_Incorrect'] = filtered_incorrect_counts[model_name]
        csv_data[f'{model_name}_Correct_Avg_Loss'] = filtered_correct_avg_loss[model_name]
        csv_data[f'{model_name}_Incorrect_Avg_Loss'] = filtered_incorrect_avg_loss[model_name]

    csv_df = pd.DataFrame(csv_data, index=filtered_bins)
    csv_df.to_csv(csv_path)
    print(f"Probability distribution and loss data saved to {csv_path}")

    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 为每个模型绘制柱状图
    bar_width = 0.3  # 每个模型的柱状图宽度
    for idx, model_name in enumerate(model_names):
        # 计算柱状图的位置偏移量
        x_positions = [i + idx * bar_width for i in range(len(filtered_bins))]

        # 使用主色调绘制正确和错误分布，正确为深色，错误为浅色
        bars_correct = ax1.bar(x_positions, filtered_correct_counts[model_name], width=bar_width,
                               label=f'{model_name} Correct',
                               color=colors[idx], alpha=0.8)
        bars_incorrect = ax1.bar(x_positions, filtered_incorrect_counts[model_name],
                                 bottom=filtered_correct_counts[model_name],
                                 width=bar_width, label=f'{model_name} Incorrect', color=colors[idx], alpha=0.4)

        # 根据 show_counts 参数控制是否显示数量
        if show_counts:
            for bar_correct, bar_incorrect, correct_count, incorrect_count in zip(bars_correct, bars_incorrect,
                                                                                  filtered_correct_counts[model_name],
                                                                                  filtered_incorrect_counts[model_name]):
                height_correct = bar_correct.get_height()
                height_incorrect = bar_incorrect.get_height() + height_correct  # 堆叠后的总高度
                ax1.text(bar_correct.get_x() + bar_correct.get_width() / 2., height_correct, f'{correct_count}',
                         ha='center', va='bottom', fontsize=10)
                ax1.text(bar_incorrect.get_x() + bar_incorrect.get_width() / 2., height_incorrect, f'{incorrect_count}',
                         ha='center', va='bottom', fontsize=10)

    ax1.set_xticks([i + (len(model_names) - 1) * bar_width / 2 for i in range(len(filtered_bins))])
    ax1.set_xticklabels(filtered_bins, rotation=45, ha='right')
    ax1.set_xlabel('Probability Range')
    ax1.set_ylabel('Number of Samples')
    ax1.set_ylim(0, 5000)
    ax1.legend(loc='upper left')

    # 绘制平均损失的折线图，使用第二个y轴
    ax2 = ax1.twinx()
    for idx, model_name in enumerate(model_names):
        x_positions = [i + (len(model_names) - 1) * bar_width / 2 for i in range(len(filtered_bins))]
        ax2.plot(x_positions, filtered_correct_avg_loss[model_name], color=colors[idx], linestyle='-', marker='o', label=f'{model_name} Correct Loss')
        ax2.plot(x_positions, filtered_incorrect_avg_loss[model_name], color=colors[idx], linestyle='--', marker='x', label=f'{model_name} Incorrect Loss')

    ax2.set_ylabel('Average Loss')
    ax2.set_ylim(0, max(max(max(filtered_correct_avg_loss[model_name]), max(filtered_incorrect_avg_loss[model_name])) for model_name in model_names) * 1.2)

    # 统一刻度
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

import pandas as pd

import pandas as pd

def plot_multi_continue_prob_loss_range(dataloader, models, model_names, num_classes, criterion,
                                        save_path='multi_model_prob_distribution_loss_scatter.svg',
                                        csv_path='probability_distribution_loss_scatter.csv', show_counts=False):
    """绘制不同模型的正确和错误样本的损失散点图，横坐标为概率值，纵坐标为损失值"""

    # 准备数据结构
    model_data = {model_name: {'correct_probs': [], 'correct_losses': [], 'incorrect_probs': [], 'incorrect_losses': []}
                  for model_name in model_names}

    # 遍历dataloader
    for batch in dataloader:
        image_paths, x_lb, y_lb = batch['image_path'], batch['x_lb'], batch['y_lb']
        x_lb = x_lb.cuda()
        y_lb = y_lb.cuda()

        # 针对每个模型计算预测
        for model, model_name in zip(models, model_names):
            with torch.no_grad():
                logits = model(x_lb)['logits']
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                # 计算损失
                losses = criterion(logits, y_lb.to(torch.float32))

            # 统计并记录结果
            for i, path in enumerate(image_paths):
                pred_prob = probs[i][preds[i]].item()  # 模型预测类别的概率
                pred = preds[i].cpu().item()  # 预测值
                true_label = y_lb[i].cpu().numpy().tolist()  # 真实标签（one-hot编码）
                loss_value = losses[i].item()

                # 区分正确和错误的样本
                if pred == true_label.index(1):  # 如果预测值与真实值相同
                    model_data[model_name]['correct_probs'].append(pred_prob)
                    model_data[model_name]['correct_losses'].append(loss_value)
                else:
                    model_data[model_name]['incorrect_probs'].append(pred_prob)
                    model_data[model_name]['incorrect_losses'].append(loss_value)

    # 确保所有列表的长度相同
    max_len = max(max(len(model_data[model_name]['correct_probs']),
                      len(model_data[model_name]['correct_losses']),
                      len(model_data[model_name]['incorrect_probs']),
                      len(model_data[model_name]['incorrect_losses'])) for model_name in model_names)

    # 填充数据以确保所有列表长度相同
    for model_name in model_names:
        for key in ['correct_probs', 'correct_losses', 'incorrect_probs', 'incorrect_losses']:
            model_data[model_name][key].extend([None] * (max_len - len(model_data[model_name][key])))

    # 保存数据为 CSV 文件
    csv_data = {}
    for model_name in model_names:
        csv_data[f'{model_name}_Correct_Probs'] = model_data[model_name]['correct_probs']
        csv_data[f'{model_name}_Correct_Losses'] = model_data[model_name]['correct_losses']
        csv_data[f'{model_name}_Incorrect_Probs'] = model_data[model_name]['incorrect_probs']
        csv_data[f'{model_name}_Incorrect_Losses'] = model_data[model_name]['incorrect_losses']

    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_path)
    print(f"Probability and loss scatter data saved to {csv_path}")

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制每个模型的正确和错误样本的散点图
    colors = ['blue', 'green', 'orange', 'purple', 'brown']  # 为每个模型指定一个主色调
    for idx, model_name in enumerate(model_names):
        # 正确样本
        ax.scatter(model_data[model_name]['correct_probs'], model_data[model_name]['correct_losses'],
                   color=colors[idx], marker='o', alpha=0.7, label=f'{model_name} Correct')
        # 错误样本
        ax.scatter(model_data[model_name]['incorrect_probs'], model_data[model_name]['incorrect_losses'],
                   color=colors[idx], marker='x', alpha=0.7, label=f'{model_name} Incorrect')

    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Loss')
    ax.set_xticks([i * 0.1 for i in range(11)])  # 从0到1，以0.1为步长
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

def plot_ce_kl(dataloader, model, num_classes, save_path='ce_kl_scatter_plot.svg',
               csv_path='ce_kl_loss_data.csv', show_counts=False):
    kl_criterion = nn.KLDivLoss(reduction='none')
    ce_criterion = nn.CrossEntropyLoss(reduction='none')

    ce_correct_probs, ce_correct_losses = [], []
    ce_incorrect_probs, ce_incorrect_losses = [], []
    kl_correct_probs, kl_correct_losses = [], []
    kl_incorrect_probs, kl_incorrect_losses = [], []

    # 遍历dataloader
    for batch in dataloader:
        image_paths, x_lb, y_lb, x_lb_s = batch['image_path'], batch['x_lb'], batch['y_lb'], batch['x_lb_s']
        x_lb = x_lb.cuda()
        y_lb = y_lb.cuda()
        x_lb_s = x_lb_s.cuda()

        # 针对每个模型计算预测
        with torch.no_grad():
            logits = model(x_lb)['logits']
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            # 将标签转换为类别索引形式
            target_labels = y_lb.argmax(dim=1)

            # 计算交叉熵损失
            ce_loss = ce_criterion(logits, target_labels)
            # 计算KL散度损失（self-KL）
            probs_s = torch.nn.functional.softmax(logits, dim=1)
            pseudo_label = torch.argmax(probs_s, dim=1) & preds
            kl_loss = ce_criterion(logits, pseudo_label)

        # 将损失和概率按正确和错误样本分别记录
        for i in range(len(target_labels)):
            pred_prob = probs[i][preds[i]].item()  # 预测类别的概率
            if preds[i] == target_labels[i]:  # 预测正确的样本
                ce_correct_probs.append(pred_prob)
                ce_correct_losses.append(ce_loss[i].item())
                kl_correct_probs.append(pred_prob)
                kl_correct_losses.append(kl_loss[i].item())
            else:  # 预测错误的样本
                ce_incorrect_probs.append(pred_prob)
                ce_incorrect_losses.append(ce_loss[i].item())
                kl_incorrect_probs.append(pred_prob)
                kl_incorrect_losses.append(kl_loss[i].item())

    # 找到最长列表的长度
    max_len = max(len(ce_correct_probs), len(ce_incorrect_probs),
                  len(kl_correct_probs), len(kl_incorrect_probs))

    # 将所有列表填充至相同长度
    for lst in [ce_correct_probs, ce_correct_losses, ce_incorrect_probs, ce_incorrect_losses,
                kl_correct_probs, kl_correct_losses, kl_incorrect_probs, kl_incorrect_losses]:
        lst.extend([None] * (max_len - len(lst)))

    # 保存数据为 CSV 文件
    csv_data = {
        'CE_Correct_Probs': ce_correct_probs,
        'CE_Correct_Losses': ce_correct_losses,
        'CE_Incorrect_Probs': ce_incorrect_probs,
        'CE_Incorrect_Losses': ce_incorrect_losses,
        'KL_Correct_Probs': kl_correct_probs,
        'KL_Correct_Losses': kl_correct_losses,
        'KL_Incorrect_Probs': kl_incorrect_probs,
        'KL_Incorrect_Losses': kl_incorrect_losses
    }
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_path, index=False)
    print(f"Cross Entropy and KL Divergence loss data saved to {csv_path}")

    # 绘制散点图
    plt.figure(figsize=(12, 8))

    # 绘制交叉熵损失的散点图
    plt.scatter(ce_correct_probs, ce_correct_losses, color='blue', marker='o', alpha=0.6, label='CE Correct')
    plt.scatter(ce_incorrect_probs, ce_incorrect_losses, color='blue', marker='x', alpha=0.6, label='CE Incorrect')

    # 绘制KL散度损失的散点图
    plt.scatter(kl_correct_probs, kl_correct_losses, color='red', marker='o', alpha=0.6, label='KL Correct')
    plt.scatter(kl_incorrect_probs, kl_incorrect_losses, color='red', marker='x', alpha=0.6, label='KL Incorrect')

    plt.xlabel('Prediction Probability')
    plt.ylabel('Loss')
    plt.title(f'Cross Entropy and KL Divergence Losses')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks([i * 0.1 for i in range(11)])
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



# 使用方法：
# plot_ce_kl(dataloader, model, "ModelName", num_classes)


def plot_fixthod_probs(dataloader, model, num_classes, save_path='fixthod_probs_distribution.png'):
    """绘制大于等于0.95阈值和小于0.95阈值的正确和错误样本分布，采用堆叠柱状图"""
    ge_95_correct = 0  # 预测正确，概率大于等于0.95
    ge_95_incorrect = 0  # 预测错误，概率大于等于0.95
    lt_95_correct = 0  # 预测正确，概率小于0.95
    lt_95_incorrect = 0  # 预测错误，概率小于0.95

    # 遍历dataloader
    for batch in dataloader:
        image_paths, x_lb, y_lb = batch['image_path'], batch['x_lb'], batch['y_lb']
        x_lb = x_lb.cuda()
        y_lb = y_lb.cuda()

        with torch.no_grad():
            logits = model(x_lb)['logits']
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

        # 统计每个样本的正确和错误分布
        for i in range(len(image_paths)):
            pred_prob = probs[i][preds[i]].item()  # 模型预测类别的概率
            pred = preds[i].cpu().item()  # 预测值
            true_label = y_lb[i].cpu().numpy().tolist()  # 真实标签（one-hot编码）

            if pred_prob >= 0.95:
                if pred == true_label.index(1):  # 如果预测正确
                    ge_95_correct += 1
                else:
                    ge_95_incorrect += 1
            else:
                if pred == true_label.index(1):  # 如果预测正确
                    lt_95_correct += 1
                else:
                    lt_95_incorrect += 1

    # 堆叠柱状图
    categories = ['≥ 0.95', '< 0.95']
    correct_values = [ge_95_correct, lt_95_correct]
    incorrect_values = [ge_95_incorrect, lt_95_incorrect]

    plt.figure(figsize=(10, 6.18))

    # 绘制堆叠柱状图
    plt.bar(categories, correct_values, label='Correct', color='green', alpha=0.7)
    plt.bar(categories, incorrect_values, bottom=correct_values, label='Incorrect', color='red', alpha=0.7)

    plt.ylabel('Number of Samples')
    plt.title('Correct and Incorrect Predictions by Probability Threshold 0.95')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # 参数
    ckpt_path = "/dk1/isic2018-exp/densenet121_/fullysupervised_nlratio0.05__Adam_lr0.0001_num_train_iter3400_bs64_seed42/model_best.pth"
    ckpt_path1 = "/dk1/isic2018-exp/densenet121_None/fixmatch___ratio_Adam_extr0.0_uratio3_nlratio0.05_lr0.0001_num_train_iter5500_bs64_seed3407/model_best.pth"
    ckpt_path2 = "/dk1/isic2018-exp/densenet121_None/flexmatch___ratio_Adam_extr0.0_uratio3_nlratio0.05_lr0.0001_num_train_iter5500_bs64_seed2300/model_best.pth"
    save_path = 'sup-fixmatch_isic2018_val_probability_distribution.png'
    csv_dir = f'/home/gu721/yzc/Semi-supervised-learning/data/ISIC2018/'
    data_dir = f"/home/gu721/yzc/data/ISIC2018/images/"
    num_classes = 7

    # 加载模型
    model = load_model(ckpt_path, num_classes)

    # 准备数据
    dataloader = prepare_data(csv_dir, data_dir, num_classes)

    # 多个模型的柱状图绘制
    models = [load_model(ckpt_path, num_classes),load_model(ckpt_path1, num_classes), load_model(ckpt_path2, num_classes)]
    # model_names = ['Supervised','Fixmatch', 'Flexmatch']
    model_names = ['Supervised']
    # save_path = 'fixmatch-flexmatch_isic2018_val_probability_distribution.svg'
    # plot_multi_continue_prob(dataloader, models, model_names, num_classes, save_path)

    save_path = 'ce-kl_isic2018_val_probability_loss_range_distribution.svg'
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # plot_multi_continue_prob_loss(dataloader, models, model_names, num_classes,criterion, save_path)
    # plot_multi_continue_prob_loss_range(dataloader, models, model_names, num_classes,criterion, save_path)
    dataloader = prepare_data_wStrong(csv_dir, data_dir, num_classes)
    plot_ce_kl(dataloader, model, num_classes,save_path)

    # save_path = 'fixmatch-flexmatch_isic2018_error_per_class.svg'
    # plot_multi_incorrect_per_class(dataloader, models, model_names, num_classes, save_path)



    # 绘制概率分布图
    # plot_continue_prob(dataloader, model, num_classes, save_path)
    # plot_fixthod_probs(dataloader, model, num_classes, save_path)