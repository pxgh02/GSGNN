import torch
from torch import nn, optim
from model.GSGNNDataLoader import *
from utils.myutils import *
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, r2_score
import glob
import traceback

def evaluate(model, data_set, is_plot=False):
    model.eval()
    # 分类指标
    acc_scores_avg = []
    f1_scores_avg = []
    fpr_scores_avg = []
    # 回归指标
    r2_scores_avg = []
    rt_avg = []
    
    print("\nBenchmark Details:")
    print(f"{'Benchmark':<15} {'Time(ms)':<10} {'Acc':<8} {'F1':<8} {'FPR':<8} {'Resource_R2':<8}")
    print("-" * 60)
    
    with torch.no_grad():
        for bm in data_set.keys():
            g = data_set[bm]
            OF_target = g.edges['e_cc'].data['isOF']
            resource_target = g.edges['e_cc'].data['resource']  # 假设demand的目标值存储在ef中

            # Start timing - 只记录模型推理时间
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            
            # Model inference
            pred_resource, pred_OF = model(g)
            
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)  # in milliseconds
            
            # 将张量移动到CPU并转换为NumPy数组
            # 分类任务
            target_np = OF_target.cpu().numpy()
            pred_np = (torch.sigmoid(pred_OF) > 0.5).float().cpu().numpy()
            
            # 回归任务
            resource_target_np = resource_target.cpu().numpy()
            pred_resource_np = pred_resource.cpu().numpy()
            
            # Calculate classification metrics
            acc = accuracy_score(target_np, pred_np)
            f1 = f1_score(target_np, pred_np)
            
            # Calculate FPR: FP / (FP + TN)
            tn = np.sum((pred_np == 0) & (target_np == 0))
            fp = np.sum((pred_np == 1) & (target_np == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Calculate regression metrics
            resource_r2 = r2_score(resource_target_np, pred_resource_np)

            # Print individual benchmark results
            print(f"{bm:<15} {elapsed_time:>8.2f} {acc:>7.4f} {f1:>7.4f} {fpr:>7.4f} {resource_r2:>7.4f}")

            acc_scores_avg.append(acc)
            f1_scores_avg.append(f1)
            fpr_scores_avg.append(fpr)
            r2_scores_avg.append(resource_r2)
            rt_avg.append(elapsed_time)
            
            if is_plot:
                # todo: plot the result
                ...
    
    print("-" * 60)
    print(f"{'Average':<15} {np.mean(rt_avg):>8.2f} {np.mean(acc_scores_avg):>7.4f} "
          f"{np.mean(f1_scores_avg):>7.4f} {np.mean(fpr_scores_avg):>7.4f} {np.mean(r2_scores_avg):>7.4f}")
    
    # 返回所有指标的平均值
    return (np.mean(acc_scores_avg), np.mean(f1_scores_avg), 
            np.mean(fpr_scores_avg), np.mean(r2_scores_avg))

def train(model, data_loader, optimizer, criterion_OF, criterion_resource, epochs):
    best_metrics = {'f1': 0}
    best_loss = float('inf')
    train_data = data_loader.train_set
    test_set = data_loader.test_set
    batch_size = data_loader.batch_size
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        sampled_bms = random.sample(list(train_data.keys()), batch_size)
        
        for bm in sampled_bms:
            optimizer.zero_grad()
            
            # Get graph and target
            g = train_data[bm]
            OF_target = g.edges['e_cc'].data['isOF']
            resource_target = g.edges['e_cc'].data['resource']
            
            # Forward pass
            resource_out, OF_out = model(g)
            
            # Compute loss
            loss_OF = criterion_OF(OF_out, OF_target)
            loss_resource = criterion_resource(resource_out, resource_target)
            loss = loss_OF + 2*loss_resource
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Calculate average and log
        avg_loss = total_loss / batch_size
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if is_eval and (epoch % eval_step == 0):
            test_acc, test_f1, test_fpr, test_r2 = evaluate(model, test_set, is_plot=False)

            if test_f1 > best_metrics['f1']:
                best_metrics['f1'] = test_f1
                # 只保存模型参数
                torch.save(model.state_dict(), 
                         f'{data_loader.checkpoint_dir}/t{tag}_f1{test_f1:.4f}.pth')
                print(f"Saved model with F1 score: {test_f1:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存最佳loss的模型参数
            torch.save(model.state_dict(), 
                     f'{data_loader.checkpoint_dir}/{tag}_best_loss.pth')
    
    # 保存最终模型参数
    torch.save(model.state_dict(), f'{data_loader.checkpoint_dir}/{tag}.pth')

def test(model, data_loader):
    model.eval()
    
    # Evaluate training set with plotting
    print("\nEvaluating Training Set:")
    train_acc, train_f1, train_fpr, train_r2 = evaluate(model, data_loader.train_set, is_plot=False)
    
    # Evaluate test set with plotting
    print("\nEvaluating Test Set:")
    test_acc, test_f1, test_fpr, test_r2 = evaluate(model, data_loader.test_set, is_plot=is_plot)

def regression_golden(file_path):
    """
    遍历目录下所有模型文件并评估性能
    Args:
        file_path: 模型文件所在目录
    """
    # 存储所有结果
    benchmark_results = {
        'train': {},
        'test': {}
    }
    
    # 获取所有模型文件
    model_files = glob.glob(os.path.join(file_path, '*.pth'))
    print(f"\nFound {len(model_files)} model files")
    
    # 对每个benchmark存储评估指标
    for bm in data_loader.train_set.keys():
        benchmark_results['train'][bm] = {
            'acc': [], 'f1': [], 'fpr': [], 'r2': []
        }
    for bm in data_loader.test_set.keys():
        benchmark_results['test'][bm] = {
            'acc': [], 'f1': [], 'fpr': [], 'r2': []
        }
    
    # 评估每个模型
    for model_file in model_files:
        print(f"\nEvaluating model: {os.path.basename(model_file)}")
        try:
            # 加载模型
            model.load_state_dict(torch.load(model_file, weights_only=True))
            model.eval()
            
            # 评估训练集
            with torch.no_grad():
                for bm, g in data_loader.train_set.items():
                    # 获取目标值
                    OF_target = g.edges['e_cc'].data['isOF']
                    resource_target = g.edges['e_cc'].data['resource']  # 使用正确的目标变量
                    
                    # 模型推理
                    pred_resource, pred_OF = model(g)
                    
                    # 分类任务
                    target_np = OF_target.cpu().numpy()
                    pred_np = (torch.sigmoid(pred_OF) > 0.5).float().cpu().numpy()
                    
                    # 回归任务
                    resource_target_np = resource_target.cpu().numpy()
                    pred_resource_np = pred_resource.cpu().numpy()
                    
                    # 计算分类指标
                    acc = accuracy_score(target_np, pred_np)
                    f1 = f1_score(target_np, pred_np)
                    
                    # Calculate FPR: FP / (FP + TN)
                    tn = np.sum((pred_np == 0) & (target_np == 0))
                    fp = np.sum((pred_np == 1) & (target_np == 0))
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    # 计算回归指标
                    r2 = r2_score(resource_target_np, pred_resource_np)
                    
                    # 存储结果
                    benchmark_results['train'][bm]['acc'].append(acc)
                    benchmark_results['train'][bm]['f1'].append(f1)
                    benchmark_results['train'][bm]['fpr'].append(fpr)
                    benchmark_results['train'][bm]['r2'].append(r2)
            
            # 评估测试集
            with torch.no_grad():
                for bm, g in data_loader.test_set.items():
                    # 获取目标值
                    OF_target = g.edges['e_cc'].data['isOF']
                    resource_target = g.edges['e_cc'].data['resource']  # 使用正确的目标变量
                    
                    # 模型推理
                    pred_resource, pred_OF = model(g)
                    
                    # 分类任务
                    target_np = OF_target.cpu().numpy()
                    pred_np = (torch.sigmoid(pred_OF) > 0.5).float().cpu().numpy()
                    
                    # 回归任务
                    resource_target_np = resource_target.cpu().numpy()
                    pred_resource_np = pred_resource.cpu().numpy()
                    
                    # 计算分类指标
                    acc = accuracy_score(target_np, pred_np)
                    f1 = f1_score(target_np, pred_np)
                    
                    # Calculate FPR: FP / (FP + TN)
                    tn = np.sum((pred_np == 0) & (target_np == 0))
                    fp = np.sum((pred_np == 1) & (target_np == 0))
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    # 计算回归指标
                    r2 = r2_score(resource_target_np, pred_resource_np)
                    
                    # 存储结果
                    benchmark_results['test'][bm]['acc'].append(acc)
                    benchmark_results['test'][bm]['f1'].append(f1)
                    benchmark_results['test'][bm]['fpr'].append(fpr)
                    benchmark_results['test'][bm]['r2'].append(r2)
                    
        except Exception as e:
            print(f"Error evaluating {model_file}: {str(e)}")
            traceback.print_exc()
            continue
    
    # 打印结果
    print("\nTraining Set Results:")
    print(f"{'Benchmark':<15} {'ACC(μ±σ)':<15} {'F1(μ±σ)':<15} {'FPR(μ±σ)':<15} {'R2(μ±σ)':<15}")
    print("-" * 75)
    
    train_metrics = {'acc': [], 'f1': [], 'fpr': [], 'r2': []}
    train_stds = {'acc': [], 'f1': [], 'fpr': [], 'r2': []}
    
    for bm in benchmark_results['train'].keys():
        metrics = benchmark_results['train'][bm]
        
        # 检查是否有有效结果
        if len(metrics['acc']) > 0:
            acc_mean, acc_std = np.mean(metrics['acc'])*100, np.std(metrics['acc'])*100
            f1_mean, f1_std = np.mean(metrics['f1'])*100, np.std(metrics['f1'])*100
            fpr_mean, fpr_std = np.mean(metrics['fpr'])*100, np.std(metrics['fpr'])*100
            r2_mean, r2_std = np.mean(metrics['r2'])*100, np.std(metrics['r2'])*100
            
            print(f"{bm:<15} {acc_mean:>6.2f}±{acc_std:>4.2f} {f1_mean:>6.2f}±{f1_std:>4.2f} "
                  f"{fpr_mean:>6.2f}±{fpr_std:>4.2f} {r2_mean:>6.2f}±{r2_std:>4.2f}")
            
            train_metrics['acc'].append(acc_mean)
            train_metrics['f1'].append(f1_mean)
            train_metrics['fpr'].append(fpr_mean)
            train_metrics['r2'].append(r2_mean)
            
            train_stds['acc'].append(acc_std)
            train_stds['f1'].append(f1_std)
            train_stds['fpr'].append(fpr_std)
            train_stds['r2'].append(r2_std)
        else:
            print(f"{bm:<15} {'No valid results':<60}")
    
    print("\nTest Set Results:")
    print(f"{'Benchmark':<15} {'ACC(μ±σ)':<15} {'F1(μ±σ)':<15} {'FPR(μ±σ)':<15} {'R2(μ±σ)':<15}")
    print("-" * 75)
    
    test_metrics = {'acc': [], 'f1': [], 'fpr': [], 'r2': []}
    test_stds = {'acc': [], 'f1': [], 'fpr': [], 'r2': []}
    
    for bm in benchmark_results['test'].keys():
        metrics = benchmark_results['test'][bm]
        
        # 检查是否有有效结果
        if len(metrics['acc']) > 0:
            acc_mean, acc_std = np.mean(metrics['acc'])*100, np.std(metrics['acc'])*100
            f1_mean, f1_std = np.mean(metrics['f1'])*100, np.std(metrics['f1'])*100
            fpr_mean, fpr_std = np.mean(metrics['fpr'])*100, np.std(metrics['fpr'])*100
            r2_mean, r2_std = np.mean(metrics['r2'])*100, np.std(metrics['r2'])*100
            
            print(f"{bm:<15} {acc_mean:>6.2f}±{acc_std:>4.2f} {f1_mean:>6.2f}±{f1_std:>4.2f} "
                  f"{fpr_mean:>6.2f}±{fpr_std:>4.2f} {r2_mean:>6.2f}±{r2_std:>4.2f}")
            
            test_metrics['acc'].append(acc_mean)
            test_metrics['f1'].append(f1_mean)
            test_metrics['fpr'].append(fpr_mean)
            test_metrics['r2'].append(r2_mean)
            
            test_stds['acc'].append(acc_std)
            test_stds['f1'].append(f1_std)
            test_stds['fpr'].append(fpr_std)
            test_stds['r2'].append(r2_std)
        else:
            print(f"{bm:<15} {'No valid results':<60}")
    
    # 打印总体平均结果
    print("\nOverall Results:")
    print(f"{'Dataset':<15} {'ACC(μ±σ)':<15} {'F1(μ±σ)':<15} {'FPR(μ±σ)':<15} {'R2(μ±σ)':<15}")
    print("-" * 75)
    
    # 训练集平均
    if train_metrics['acc']:
        train_avg = {k: np.mean(v) for k, v in train_metrics.items()}
        train_avg_std = {k: np.mean(v) for k, v in train_stds.items()}
        print(f"{'Train':<15} {train_avg['acc']:>6.2f}±{train_avg_std['acc']:>4.2f} "
              f"{train_avg['f1']:>6.2f}±{train_avg_std['f1']:>4.2f} "
              f"{train_avg['fpr']:>6.2f}±{train_avg_std['fpr']:>4.2f} "
              f"{train_avg['r2']:>6.2f}±{train_avg_std['r2']:>4.2f}")
    else:
        print(f"{'Train':<15} {'No valid results':<60}")
    
    # 测试集平均
    if test_metrics['acc']:
        test_avg = {k: np.mean(v) for k, v in test_metrics.items()}
        test_avg_std = {k: np.mean(v) for k, v in test_stds.items()}
        print(f"{'Test':<15} {test_avg['acc']:>6.2f}±{test_avg_std['acc']:>4.2f} "
              f"{test_avg['f1']:>6.2f}±{test_avg_std['f1']:>4.2f} "
              f"{test_avg['fpr']:>6.2f}±{test_avg_std['fpr']:>4.2f} "
              f"{test_avg['r2']:>6.2f}±{test_avg_std['r2']:>4.2f}")
    else:
        print(f"{'Test':<15} {'No valid results':<60}")

random_seed = 27
tag = 30615270
epochs = 500
lr = 0.0002
weight_decay = 0.0001
eval_step = 5
model_name = "GSGNN"
is_plot = False
is_eval = True

if __name__ == '__main__':
    data_loader = GSGNN_DataLoader(model_name, random_seed=random_seed, n_train=10, batch_size=7)
    # model = data_loader.load_model(model_name, tag=tag, load_path="/home/pengxuan/Project/CSteinerPred/data/checkpoints/GSGNN_Gloden/t30615040_f10.8310.pth")
    model = data_loader.load_model(model_name, tag=tag, load_path=None)

    # /home/pengxuan/Project/CSteinerPred/data/checkpoints/GSGNN_Gloden/t30615220_f10.8274.pth， r2很低
    # /home/pengxuan/Project/CSteinerPred/data/checkpoints/GSGNN_Gloden/t30615250_f10.8269.pth， r2很低
    criterion_OF = nn.BCEWithLogitsLoss()
    criterion_resource = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("Starting training...")
    train(model, data_loader, optimizer, criterion_OF, criterion_resource, epochs)
    print("Training completed.")

    print("Starting testing...")
    test(model, data_loader)
    print("Testing completed.")

    # regression_golden('/home/pengxuan/Project/CSteinerPred/data/checkpoints/GSGNN_Gloden')
    

