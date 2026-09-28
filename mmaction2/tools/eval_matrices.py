import pickle
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 你的类别定义
CLASSES = [
    'backhand_drive',   
    'dink',
    'drop_shot',
    'forehand_drive',   
    'serve',      
    'volley'    
]

def parse_args():
    parser = argparse.ArgumentParser(description="📈 MMAction2 单模型详细指标计算器 (Acc, Precision, Recall, F1)")
    parser.add_argument('-r', '--results', type=str, required=True, help='模型预测结果 (.pkl)')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='验证集 Ground Truth (.pkl)')
    return parser.parse_args()

def extract_scores(result_list):
    """兼容提取 MMAction2 的预测分数"""
    scores = []
    for item in result_list:
        val = item['pred_scores']['item'] if isinstance(item, dict) and 'pred_scores' in item else \
              item['pred_score'] if isinstance(item, dict) else list(item.values())[0] if isinstance(item, dict) else item
        if hasattr(val, 'cpu'): val = val.cpu().detach().numpy()
        scores.append(np.squeeze(val))
    return np.array(scores)

def main():
    args = parse_args()
    
    # 1. 提取真值
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)
    name_to_label = {ann['frame_dir']: int(ann['label']) for ann in dataset['annotations']}
    val_videos = dataset['split']['val']
    gt_labels = np.array([name_to_label[vid] for vid in val_videos])
    
    # 2. 读取模型预测分数
    with open(args.results, 'rb') as f:
        preds = pickle.load(f)
    scores = extract_scores(preds)
    
    # 获取最高得分的索引作为预测结果
    pred_labels = np.argmax(scores, axis=1)
    overall_acc = np.mean(pred_labels == gt_labels) * 100
    
    print("=" * 60)
    print(f"🏆 全局整体准确率 (Overall Accuracy): {overall_acc:.2f}%")
    print("=" * 60)
    
    # 3. 打印核心指标报表
    print("📊 多分类综合指标报表:")
    print(classification_report(gt_labels, pred_labels, target_names=CLASSES, digits=4))
    
    # 4. 单独打印类别详细指标
    print("🎯 各动作类别独立详细指标 (Precision / Recall / F1-Score):")
    cm = confusion_matrix(gt_labels, pred_labels)
    
    # 计算召回率 Recall = 对角线(答对的) / 行总和(实际总数)
    recall = cm.diagonal() / cm.sum(axis=1)
    
    # 计算查准率 Precision = 对角线(答对的) / 列总和(预测总数)
    # 加上 1e-9 防止除以 0
    precision = cm.diagonal() / (cm.sum(axis=0) + 1e-9)
    
    # 计算 F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    for i, cls_name in enumerate(CLASSES):
        print(f"   - {cls_name.ljust(15)} : Precision: {precision[i]*100:5.2f}% | Recall: {recall[i]*100:5.2f}% | F1: {f1_score[i]*100:5.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    main()