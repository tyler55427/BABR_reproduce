import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import timm
from pathlib import Path
import argparse
import random
import warnings
warnings.filterwarnings('ignore')

class OptimizedBoneDataset(Dataset):
    """优化的数据集类，支持数据增强和平衡采样"""
    
    def __init__(self, dataframe, image_dir, transform=None, is_training=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_training = is_training
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # 获取图像文件名和标签
        img_name = self.dataframe.loc[idx, '新文件名']
        label = int(self.dataframe.loc[idx, '等级']) - 1  # 转为0-based
        
        # 构建图像路径
        img_path = self.image_dir / img_name
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_optimized_transforms(input_size=224):
    """优化的数据增强策略"""
    
    # 训练时的强数据增强
    train_transform = transforms.Compose([
        transforms.Resize((int(input_size * 1.2), int(input_size * 1.2))),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])
    
    # 验证时的轻度增强
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_class_weights(df):
    """计算类别权重来处理类别不平衡"""
    class_counts = df['等级'].value_counts().sort_index()
    total_samples = len(df)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights.values)

def get_balanced_sampler(df):
    """创建平衡采样器"""
    labels = df['等级'] - 1  # 转为0-based
    class_counts = pd.Series(labels).value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels].values
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def create_robust_model(model_name='efficientnet_b0', num_classes=3, pretrained=True, dropout_rate=0.5):
    """创建具有更强正则化的模型"""
    
    # 使用较小的模型避免过拟合
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    # 添加更强的正则化
    if hasattr(model, 'classifier'):
        # 对于EfficientNet等模型
        if isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(model.classifier.in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(512, num_classes)
            )
    elif hasattr(model, 'fc'):
        # 对于ResNet等模型
        if isinstance(model.fc, nn.Linear):
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(512, num_classes)
            )
    
    return model

def train_with_regularization(model, train_loader, val_loader, class_weights, 
                            num_epochs=50, initial_lr=0.001, patience=10):
    """使用正则化技术的训练函数"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = model.to(device)
    
    # 使用加权损失函数处理类别不平衡
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    
    # 使用AdamW优化器，更好的权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)
    
    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # 早停机制
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # 训练历史
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print("开始训练（带正则化）...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 添加L2正则化损失
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 1e-4 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
        
        # 计算平均指标
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct_val / total_val
        
        # 记录历史
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✓ 新最佳验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 60)
        
        # # 早停
        # if patience_counter >= patience:
        #     print(f"早停触发，最佳验证准确率: {best_val_acc:.2f}%")
        #     break
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc, (train_losses, val_losses, train_accuracies, val_accuracies)

def evaluate_model_detailed(model, test_loader, class_names=['等级1', '等级2', '等级3'], output_dir=None):
    """详细的模型评估"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 详细的分类报告
    print("=== 详细分类报告 ===")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(str(output_dir / 'optimized_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算各类别准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print("\n各类别准确率:")
    for i, acc in enumerate(class_accuracies):
        print(f"  {class_names[i]}: {acc:.3f}")
    
    # 总体准确率
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"\n总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

def main_optimized(label_csv=None, image_dir=None, output_dir=None):
    """优化的主训练函数

    Parameters
    ----------
    label_csv : str or Path
        Path to labels CSV/Excel file (columns: 新文件名, 等级).
        Default: ./datas/labels.csv
    image_dir : str or Path
        Path to directory containing all images (flat structure).
        Default: ./datas/images
    output_dir : str or Path
        Path to save model and results.
        Default: ./results
    """

    # 设置路径 — 使用相对路径，可被命令行参数覆盖
    project_root = Path(__file__).parent

    if label_csv is None:
        label_csv = project_root / "datas" / "labels.csv"
    else:
        label_csv = Path(label_csv)

    if image_dir is None:
        image_dir = project_root / "datas" / "images"
    else:
        image_dir = Path(image_dir)

    if output_dir is None:
        output_dir = project_root / "results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 优化的骨吸收等级分类实验 ===")
    print(f"标签文件: {label_csv}")
    print(f"图像目录: {image_dir}")
    print(f"输出目录: {output_dir}")

    # 读取数据 — 支持 CSV 和 Excel 两种格式
    label_csv = Path(label_csv)
    if label_csv.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(label_csv)
    else:
        df = pd.read_csv(label_csv)
    df = df[df['等级'].isin([1, 2, 3])].copy()
    
    print(f"数据集大小: {len(df)}")
    print("等级分布:")
    level_counts = df['等级'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"  等级 {level}: {count} 张 ({count/len(df)*100:.1f}%)")
    
    # 计算类别权重
    class_weights = get_class_weights(df)
    print(f"\n类别权重: {class_weights}")
    
    # 数据划分
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['等级'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['等级'], random_state=42
    )
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_df)}")
    print(f"验证集: {len(val_df)}")
    print(f"测试集: {len(test_df)}")
    
    # 使用较小的模型避免过拟合
    model_name = 'efficientnet_b0'  # 改用较小的模型
    input_size = 224
    batch_size = 32
    
    print(f"\n使用模型: {model_name}")
    print(f"输入尺寸: {input_size}, 批次大小: {batch_size}")
    
    # 获取变换
    train_transform, val_transform = get_optimized_transforms(input_size)
    
    # 创建数据集
    train_dataset = OptimizedBoneDataset(train_df, image_dir, train_transform, is_training=True)
    val_dataset = OptimizedBoneDataset(val_df, image_dir, val_transform, is_training=False)
    test_dataset = OptimizedBoneDataset(test_df, image_dir, val_transform, is_training=False)
    
    # 创建平衡采样器
    train_sampler = get_balanced_sampler(train_df)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,  # 使用平衡采样
        num_workers=0, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    # 创建优化的模型
    model = create_robust_model(
        model_name=model_name, 
        num_classes=3, 
        pretrained=True, 
        dropout_rate=0.5
    )
    
    # 训练模型
    print("开始优化训练...")
    trained_model, best_val_acc, history = train_with_regularization(
        model, train_loader, val_loader, class_weights,
        num_epochs=150, initial_lr=0.001, patience=15
    )
    
    # 评估模型
    print("评估优化后的模型...")
    test_accuracy = evaluate_model_detailed(trained_model, test_loader, output_dir=output_dir)
    
    # 保存模型
    model_path = output_dir / 'optimized_bone_classifier.pth'
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_name': model_name,
        'input_size': input_size,
        'class_weights': class_weights,
        'best_val_acc': best_val_acc,
        'test_acc': test_accuracy
    }, str(model_path))

    print(f"\n=== 优化实验结果 ===")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"模型已保存为: {model_path}")


def main():
    """命令行入口 — 解析参数并调用训练主函数"""
    parser = argparse.ArgumentParser(
        description="Train a 3-class BRAR severity classifier on panoramic radiographs"
    )
    parser.add_argument(
        "--label_csv",
        type=str,
        default=None,
        help="Path to labels CSV/Excel file (columns: 新文件名, 等级). "
             "Default: ./datas/labels.csv",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to directory containing all images (flat structure). "
             "Default: ./datas/images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model and results. Default: ./results",
    )
    args = parser.parse_args()
    main_optimized(
        label_csv=args.label_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
    )


# 运行优化版本
if __name__ == "__main__":
    main()