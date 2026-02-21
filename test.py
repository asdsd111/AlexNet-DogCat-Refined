import torch
from Net import AlexNet
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT_TRAIN = r'D:/Desktop/AlexNet/archive/train'
ROOT_TEST = r'D:/Desktop/AlexNet/archive/test'

# 将图像RGB三个通道的像素值分别减去0.5,再除以0.5.从而将所有的像素值固定在[-1,1]范围内
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])

val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 如果显卡可用，则用显卡进行推理
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 调用net里面定义的网络模型，并加载训练好的权重
model = AlexNet().to(device)
model.load_state_dict(torch.load('D:/Desktop/AlexNet/save_model/best_model.pth', map_location=device))
model.eval()

# 类别名称（与训练集的文件夹顺序保持一致），例如 ['cats', 'dogs']
classes = val_dataset.classes
# 英文标签到中文标签的映射
label_map = {
    "cats": "猫",
    "cat": "猫",
    "dogs": "狗",
    "dog": "狗",
}

# 把tensor转成Image,方便可视化（单张）
to_pil = ToPILImage()

# 结果保存目录
OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def visualize_batch(images, preds, labels, classes, batch_idx, max_images=16):
    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()

    # 截取前 max_images 张
    images = images[:max_images]
    preds = preds[:max_images]
    labels = labels[:max_images]

    # 保存每一张图片到指定文件夹，并在控制台打印预测/真实标签（显示为中文）
    print("本批图像预测结果:")
    for i in range(len(images)):
        # 反归一化：从[-1,1]恢复到[0,1]，因为ToPILImage需要[0,1]范围的tensor
        img_tensor = images[i] * 0.5 + 0.5
        img_tensor = img_tensor.clamp(0, 1)  # 确保在[0,1]范围内
        img = to_pil(img_tensor)
        en_pred = classes[preds[i]]
        en_gt = classes[labels[i]]
        pred_label = label_map.get(en_pred, en_pred)
        gt_label = label_map.get(en_gt, en_gt)
        filename = f"batch{batch_idx}_idx{i}_pred-{pred_label}_gt-{gt_label}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        img.save(save_path)
        print(f"Idx {i}: 预测 = {pred_label}, 真实 = {gt_label}, Saved: {save_path}")

    # 生成带文字标签的网格图
    n_images = len(images)
    cols = 4
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n_images:
            img = images[i]
            # 反归一化到 [0,1] 方便显示
            img = img * 0.5 + 0.5
            ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
            en_pred = classes[preds[i]]
            en_gt = classes[labels[i]]
            pred_label = label_map.get(en_pred, en_pred)
            gt_label = label_map.get(en_gt, en_gt)
            ax.set_title(f"预测: {pred_label}\n真实: {gt_label}", fontsize=10)

    fig.suptitle(f"Batch {batch_idx} predictions", fontsize=14)
    grid_path = os.path.join(OUTPUT_DIR, f"batch{batch_idx}_grid.jpg")
    plt.tight_layout()
    plt.savefig(grid_path)
    plt.show()
    print(f"Grid saved to: {grid_path}")


def evaluate_and_visualize(model, dataloader, classes, device, max_batches_to_show=2):
    total, correct = 0, 0
    batch_idx = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 可视化前几个batch，并保存到文件夹
            if batch_idx < max_batches_to_show:
                visualize_batch(images, preds, labels, classes, batch_idx=batch_idx)
                batch_idx += 1

    acc = correct / total if total > 0 else 0
    print(f"\n验证集整体准确率: {acc:.4f}  ({correct}/{total})")


if __name__ == "__main__":
    evaluate_and_visualize(model, val_dataloader, classes, device)