from semilearn.nets import densenet121
from semilearn.datasets.cv_datasets.isic2018 import ISIC2018Dataset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

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
    test_all_info = test_all_info.fillna(0)
    test_data = test_all_info.iloc[:, 0].values
    test_data = [data_dir + i + '.jpg' for i in test_data]
    test_targets = test_all_info.iloc[:, 1:num_classes + 1].values
    test_dset = ISIC2018Dataset('hyperplusfixmatchv3', test_data, test_targets, num_classes, transform_val, False, None,
                                False, is_test=True)
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

# def plot_multi_incorrect_per_class(dataloader, models, model_names, num_classes, save_path='multi_model_incorrect_per_class.png'):
#     """绘制多个模型在阈值大于等于0.95时，每个类别的错误率，横坐标为类别索引，纵坐标为错误率"""
#
#     # 初始化每个模型的统计数据
#     model_incorrect_counts = {model_name: [0] * num_classes for model_name in model_names}
#     model_total_counts = {model_name: [0] * num_classes for model_name in model_names}
#
#     # 遍历dataloader
#     for batch in dataloader:
#         image_paths, x_lb, y_lb = batch['image_path'], batch['x_lb'], batch['y_lb']
#         x_lb = x_lb.cuda()
#         y_lb = y_lb.cuda()
#
#         # 针对每个模型计算预测
#         for model, model_name in zip(models, model_names):
#             with torch.no_grad():
#                 logits = model(x_lb)['logits']
#                 probs = torch.nn.functional.softmax(logits, dim=1)
#                 preds = probs.argmax(dim=1)
#
#             # 统计每个类别的错误率
#             for i in range(len(image_paths)):
#                 pred_prob = probs[i][preds[i]].item()  # 模型预测类别的概率
#                 pred = preds[i].cpu().item()  # 预测值
#                 true_label = y_lb[i].cpu().tolist().index(1)  # 真实类别的索引
#
#                 # 记录大于等于0.95阈值的样本
#                 if pred_prob >= 0.95:
#                     model_total_counts[model_name][true_label] += 1  # 记录该类别的总样本数
#                     if pred != true_label:  # 预测错误
#                         model_incorrect_counts[model_name][true_label] += 1
#
#     # 计算每个类别的错误率
#     model_incorrect_rates = {}
#     for model_name in model_names:
#         incorrect_rates = []
#         for i in range(num_classes):
#             if model_total_counts[model_name][i] > 0:
#                 incorrect_rate = model_incorrect_counts[model_name][i] / model_total_counts[model_name][i]
#             else:
#                 incorrect_rate = 0.0
#             incorrect_rates.append(incorrect_rate)
#         model_incorrect_rates[model_name] = incorrect_rates
#
#     # Debug: 输出每个模型的错误率和样本统计以供验证
#     for model_name in model_names:
#         print(f"Model: {model_name}")
#         print(f"Total samples per class: {model_total_counts[model_name]}")
#         print(f"Incorrect samples per class: {model_incorrect_counts[model_name]}")
#         print(f"Incorrect rates per class: {model_incorrect_rates[model_name]}")
#         print("\n")
#
#     # 绘图
#     plt.figure(figsize=(12, 8))
#
#     # 颜色列表
#     colors = ['blue', 'green', 'orange', 'purple', 'brown']  # 为每个模型指定颜色
#
#     # 为每个模型绘制折线图
#     for idx, model_name in enumerate(model_names):
#         plt.plot(range(num_classes), model_incorrect_rates[model_name], marker='o', label=model_name, color=colors[idx])
#
#     plt.xticks(range(num_classes), [f'Class {i}' for i in range(num_classes)])
#     plt.xlabel('Class Index')
#     plt.ylabel('Error Rate')
#     plt.ylim(0, 1)  # 错误率的范围是0到1
#     plt.title('Error Rates Per Class for Models with Threshold ≥ 0.95')
#     plt.legend(loc='upper right')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()



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
    models = [load_model(ckpt_path1, num_classes), load_model(ckpt_path2, num_classes)]
    model_names = ['Fixmatch', 'Flexmatch']
    # save_path = 'fixmatch-flexmatch_isic2018_val_probability_distribution.svg'
    # plot_multi_continue_prob(dataloader, models, model_names, num_classes, save_path)


    save_path = 'fixmatch-flexmatch_isic2018_error_per_class.svg'
    plot_multi_incorrect_per_class(dataloader, models, model_names, num_classes, save_path)



    # 绘制概率分布图
    # plot_continue_prob(dataloader, model, num_classes, save_path)
    # plot_fixthod_probs(dataloader, model, num_classes, save_path)