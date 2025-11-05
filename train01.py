import pandas as pd
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import timm  # 更新的模型库
import warnings
warnings.filterwarnings('ignore')

class BoneResorptionDataset(Dataset):
    """骨吸收等级分类数据集 - 使用脱敏后的数据"""
    
    def __init__(self, dataframe,脱敏图像目录, transform=None):
        """
        Args:
            dataframe: 包含图像信息和标签的DataFrame
            脱敏图像目录: 脱敏后图像文件的目录
            transform: 图像变换
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.脱敏图像目录 = Path(脱敏图像目录)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # 获取脱敏后的图像文件名和标签
        img_name = self.dataframe.loc[idx, '新文件名']  # 使用脱敏后的文件名
        label = int(self.dataframe.loc[idx, '等级']) - 1  # 转换为0-based索引 (0,1,2)
        
        # 构建脱敏图像路径
        img_path = self.脱敏图像目录 / img_name
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}, 使用默认图像")
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_advanced_transforms():
    """获取先进的数据增强变换"""
    
    # 训练时的先进数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])
    
    # 验证时的标准变换
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_modern_model(model_name='efficientnet_b0', num_classes=3, pretrained=True):
    """
    创建现代的深度学习模型
    
    Args:
        model_name: 模型名称 ('efficientnet_b0', 'efficientnet_b2', 'vit_base_patch16_224', etc.)
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
    """
    
    if 'efficientnet' in model_name:
        # 使用EfficientNet系列
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
    elif 'vit' in model_name:
        # 使用Vision Transformer
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
    elif 'resnet' in model_name:
        # 使用更新的ResNet变体
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
    else:
        # 默认使用efficientnet_b2
        model = timm.create_model('efficientnet_b2', pretrained=pretrained, num_classes=num_classes)
    
    return model

def get_model_parameters(model_name):
    """获取模型推荐的输入尺寸和参数"""
    
    model_configs = {
        'efficientnet_b0': {'input_size': 224, 'batch_size': 32, 'lr': 0.001},
        'efficientnet_b1': {'input_size': 240, 'batch_size': 24, 'lr': 0.001},
        'efficientnet_b2': {'input_size': 260, 'batch_size': 24, 'lr': 0.001},
        'efficientnet_b3': {'input_size': 300, 'batch_size': 16, 'lr': 0.0005},
        'vit_base_patch16_224': {'input_size': 224, 'batch_size': 16, 'lr': 0.0001},
        'vit_small_patch16_224': {'input_size': 224, 'batch_size': 32, 'lr': 0.0001},
        'swin_tiny_patch4_window7_224': {'input_size': 224, 'batch_size': 16, 'lr': 0.0001}
    }
    
    return model_configs.get(model_name, model_configs['efficientnet_b2'])

def create_adaptive_transforms(input_size):
    """根据模型输入尺寸创建自适应变换"""
    
    train_transform = transforms.Compose([
        transforms.Resize((int(input_size * 1.15), int(input_size * 1.15))),  # 稍大一些用于裁剪
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_modern_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001):
    """训练现代深度学习模型"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 梯度裁剪
    grad_clip = 1.0
    
    # 训练历史记录
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print("开始训练现代深度学习模型...")
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
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
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
        
        # 计算平均损失和准确率
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct_val / total_val
        
        # 记录历史
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"✓ 新的最佳验证准确率: {val_acc:.2f}%")
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 60)
        
        scheduler.step()
    
    # 加载最佳模型权重
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型，验证准确率: {best_val_acc:.2f}%")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    return model, best_val_acc

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Training Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('modern_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_loader, class_names=['等级1', '等级2', '等级3']):
    """评估模型性能"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
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
    
    # 计算分类报告
    print("=== 分类性能报告 ===")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 16})
    plt.title('混淆矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.savefig('modern_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算总体准确率
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"\n总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, all_predictions, all_labels, all_probabilities

def main():
    """主函数 - 使用现代模型和脱敏数据"""
    
    # 设置路径
    excel_path = r"C:\work\2025-08-10\files\SD文章\完整图像标签数据集.xlsx"
    脱敏图像目录 = r"C:\work\2025-08-10\files\SD文章\数据\脱敏数据汇总\图像文件"
    
    print("=== 现代深度学习骨吸收等级分类实验 ===")
    print("读取数据集...")
    df = pd.read_excel(excel_path)
    print(f"原始数据集大小: {len(df)}")
    
    # 数据预处理 - 只保留等级1,2,3
    df = df[df['等级'].isin([1, 2, 3])].copy()
    print(f"过滤后数据集大小: {len(df)}")
    
    # 显示等级分布
    print("\n等级分布:")
    level_counts = df['等级'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"  等级 {level}: {count} 张图像")
    
    # 划分数据集
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        stratify=df['等级'],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['等级'],
        random_state=42
    )
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_df)} 张图像")
    print(f"验证集: {len(val_df)} 张图像")
    print(f"测试集: {len(test_df)} 张图像")
    
    # 选择现代模型
    model_name = 'efficientnet_b2'  # 可选: 'vit_base_patch16_224', 'efficientnet_b3', 'swin_tiny_patch4_window7_224'
    print(f"\n使用模型: {model_name}")
    
    # 获取模型参数
    model_params = get_model_parameters(model_name)
    input_size = model_params['input_size']
    batch_size = model_params['batch_size']
    learning_rate = model_params['lr']
    
    print(f"模型参数: 输入尺寸={input_size}, 批次大小={batch_size}, 学习率={learning_rate}")
    
    # 创建自适应变换
    train_transform, val_transform = create_adaptive_transforms(input_size)
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = BoneResorptionDataset(train_df, 脱敏图像目录, train_transform)
    val_dataset = BoneResorptionDataset(val_df, 脱敏图像目录, val_transform)
    test_dataset = BoneResorptionDataset(test_df, 脱敏图像目录, val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 创建现代模型
    print("创建现代深度学习模型...")
    model = create_modern_model(model_name, num_classes=3, pretrained=True)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 训练模型
    print("开始训练...")
    trained_model, best_val_acc = train_modern_model(
        model, train_loader, val_loader, 
        num_epochs=150, learning_rate=learning_rate
    )
    
    # 评估模型
    print("评估模型性能...")
    test_accuracy, predictions, labels, probabilities = evaluate_model(trained_model, test_loader)
    
    # 保存模型
    model_save_path = f'modern_bone_resorption_{model_name}.pth'
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_name': model_name,
        'input_size': input_size,
        'num_classes': 3,
        'best_val_acc': best_val_acc,
        'test_acc': test_accuracy
    }, model_save_path)
    print(f"模型已保存为: {model_save_path}")
    
    # 保存预测结果
    results_df = test_df.copy()
    results_df['预测等级'] = np.array(predictions) + 1  # 转换回1-based
    results_df['预测概率_等级1'] = np.array(probabilities)[:, 0]
    results_df['预测概率_等级2'] = np.array(probabilities)[:, 1]
    results_df['预测概率_等级3'] = np.array(probabilities)[:, 2]
    results_df['预测正确'] = (np.array(predictions) == np.array(labels))
    
    results_df.to_excel('预测结果详情.xlsx', index=False)
    print("预测结果已保存为: 预测结果详情.xlsx")
    
    print(f"\n=== 实验总结 ===")
    print(f"使用模型: {model_name}")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"模型已保存: {model_save_path}")

# 快速开始函数
def quick_start_modern():
    """快速开始现代深度学习训练"""
    
    try:
        main()
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("\n请确保已安装必要的依赖包:")
        print("pip install torch torchvision pandas scikit-learn matplotlib seaborn pillow timm")

if __name__ == "__main__":
    quick_start_modern()