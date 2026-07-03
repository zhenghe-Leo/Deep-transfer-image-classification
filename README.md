# Deep-transfer-image-classification
# 基于深度迁移学习的图像分类系统

本仓库用于 NTU 硕士课程 EE6483 Project 2 的图像分类实验，主要包含猫狗二分类实验与 CIFAR-10 扩展实验的代码、结果文件和最终提交文件。

## 项目目标

本项目的目标是完成以下几类实验：

1. 猫狗二分类基础实验
   - 从零训练 `SimpleCNN`
   - 从零训练 `ResNet-50`
   - 使用 ImageNet 预训练权重的 `ResNet-50` 微调
2. 消融实验
   - 学习率对比
   - 数据增强对比
   - 模型效果汇总对比
3. 扩展实验
   - CIFAR-10 多分类实验
   - 类别不平衡处理实验
4. 最终提交
   - 生成 `submission.csv` 作为测试集预测结果

## 项目结构

- `src_code/`：训练、预测、模型、数据集、工具函数等代码
- `outputs/`：整理好的图表和表格，供报告使用
- `checkpoints/`：训练保存的模型权重
- `experiments/`：每个实验的日志与结果记录
- `submission.csv`：最终提交给助教的测试集预测结果
- `requirements.txt`：项目依赖

## 数据集约定

请按照实验计划准备数据目录，推荐结构如下：

```text
project/
├── data/
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   ├── val/
│   │   ├── cat/
│   │   └── dog/
│   └── test/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
```

其中：

- `train/`：训练集，cat 与 dog 各 10000 张
- `val/`：验证集，cat 与 dog 各 2500 张
- `test/`：测试集，共 500 张，文件名为 `1.jpg` 到 `500.jpg`

## 环境安装

建议使用 Python 3.10 或以上版本。

```bash
pip install -r requirements.txt
```

如果你使用虚拟环境，建议先创建并激活环境，再安装依赖。

## 主要脚本说明

### 1. 训练脚本

用于训练猫狗分类模型，并自动保存最佳权重、训练曲线、混淆矩阵和分类报告。

```bash
python src_code/train.py --model resnet50_pretrained --save_path checkpoints/resnet50_pretrained_best.pth --log_dir experiments/exp3_resnet50_pretrained
```

可选模型：

- `simplecnn`
- `resnet50_scratch`
- `resnet50_pretrained`

### 2. 预测脚本

用于加载训练好的模型，对测试集进行预测并生成 `submission.csv`。

```bash
python src_code/predict.py --model resnet50_pretrained --model_path checkpoints/resnet50_pretrained_best.pth --save_path submission.csv
```

### 3. CIFAR-10 训练脚本

用于 CIFAR-10 多分类实验以及类别不平衡实验。

```bash
python src_code/train_cifar10.py --save_path checkpoints/cifar10_best.pth --log_dir experiments/exp6_cifar10
```

## 实验输出文件说明

训练与实验结果会整理到 `outputs/` 目录，主要包括：

- `fig_simplecnn_training_curve.png`：SimpleCNN 训练曲线
- `fig_resnet50scratch_training_curve.png`：ResNet-50 从零训练曲线
- `fig_main_training_curve.png`：主模型训练曲线
- `fig_confusion_matrix.png`：验证集混淆矩阵
- `fig_correct_samples.png`：正确分类样本示例
- `fig_wrong_samples.png`：错误分类样本示例
- `fig_cifar10_training_curve.png`：CIFAR-10 训练曲线
- `fig_cifar10_confusion_matrix.png`：CIFAR-10 混淆矩阵
- `table_classification_report.txt`：分类报告
- `table_simplecnn_result.txt`：SimpleCNN 最佳验证准确率
- `table_resnet50scratch_result.txt`：ResNet-50 从零训练最佳验证准确率
- `table_model_comparison.csv`：模型对比汇总表
- `table_lr_ablation.csv`：学习率消融结果
- `table_augmentation_ablation.csv`：数据增强消融结果
- `table_cifar10_per_class_accuracy.csv`：CIFAR-10 各类别准确率
- `table_cifar10_overall_accuracy.txt`：CIFAR-10 总体准确率
- `table_imbalance_comparison.csv`：类别不平衡实验结果

## 推荐运行顺序

建议按以下顺序完成实验：

1. 先训练主模型 `resnet50_pretrained`
2. 使用最佳模型生成 `submission.csv`
3. 再训练 `simplecnn` 和 `resnet50_scratch` 作为对比实验
4. 进行学习率与数据增强消融实验
5. 再运行 CIFAR-10 实验与类别不平衡实验

## 结果提交说明

最终需要提交的内容通常包括：

- `submission.csv`
- `outputs/` 中的图表与表格
- 相关实验日志与模型权重（如课程要求保留）

## 说明

当前代码结构已尽量按照实验执行计划整理，后续可根据你的数据集路径、训练参数和课程要求继续调整。如果你希望，我也可以进一步帮你把 README 改成“可直接提交给老师/助教”的正式版说明文档。
