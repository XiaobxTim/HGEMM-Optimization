import os
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_data_from_file(file_path):
    """从单个result.txt文件中提取score值和gFLOPS数据"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # 提取score值
            score_match = re.search(r'Related Score: (\d+(\.\d+)?) / 100', content)
            score = float(score_match.group(1)) if score_match else None
            
            # 提取cuBLAS FP16 GEMM的gFLOPS
            cublas_match = re.search(r'cuBLAS FP16 GEMM.*gFLOPS: (\d+(\.\d+)?)', content)
            cublas_gflops = float(cublas_match.group(1)) if cublas_match else None
            
            # 提取Custom FP16 Kernel的gFLOPS
            custom_match = re.search(r'Custom FP16 Kernel.*gFLOPS: (\d+(\.\d+)?)', content)
            custom_gflops = float(custom_match.group(1)) if custom_match else None
            
            if not all([score, cublas_gflops, custom_gflops]):
                missing = []
                if score is None:
                    missing.append("score")
                if cublas_gflops is None:
                    missing.append("cuBLAS gFLOPS")
                if custom_gflops is None:
                    missing.append("Custom Kernel gFLOPS")
                print(f"警告: 文件 {file_path} 中未找到以下信息: {', '.join(missing)}")
                return None
            
            return {
                'score': score,
                'cublas_gflops': cublas_gflops,
                'custom_gflops': custom_gflops
            }
            
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return None

def process_result_files(directory):
    """处理目录下的所有result.txt文件"""
    data = []
    
    # 查找目录下所有包含"result"的txt文件
    for filename in os.listdir(directory):
        if 'result' in filename.lower() and filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            file_data = extract_data_from_file(file_path)
            if file_data is not None:
                # 从文件名提取case名称
                case_name = filename.split('_')[1].replace('.txt', '')
                data.append({
                    'case_name': case_name,
                    **file_data
                })
                print(f"{case_name} -- Score: {file_data['score']}, "
                      f"cuBLAS: {file_data['cublas_gflops']:.4f} gFLOPS, "
                      f"Custom: {file_data['custom_gflops']:.4f} gFLOPS")
    
    return data

def plot_gflops_comparison(data):
    """绘制cuBLAS和Custom Kernel的gFLOPS对比柱状图"""
    case_names = [item['case_name'] for item in data]
    cublas_gflops = [item['cublas_gflops'] for item in data]
    custom_gflops = [item['custom_gflops'] for item in data]
    
    x = np.arange(len(case_names))  # 标签位置
    width = 0.35  # 柱状图宽度
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, cublas_gflops, width, label='cuBLAS FP16 GEMM')
    rects2 = ax.bar(x + width/2, custom_gflops, width, label='Custom FP16 Kernel')
    
    # 添加一些文本用于标签、标题和自定义x轴刻度标签， etc.
    ax.set_ylabel('gFLOPS')
    ax.set_title('cuBLAS vs Custom Kernel FP16 GEMM Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(case_names, rotation=45, ha='right')
    ax.legend()
    
    # 在柱状图上添加数值标签
    def autolabel(rects):
        """在柱状图上附加文本标签"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0, fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()  # 调整布局，防止标签被截断
    plt.savefig('gflops_comparison.png', dpi=300)
    plt.show()

def main():
    # 指定result文件所在目录
    result_directory = '/work/sustcsc_12/xbx/HGEMM/data/output/compare_results'
    
    # 处理所有result文件
    data = process_result_files(result_directory)
    
    if not data:
        print("未找到任何有效的result文件！")
        return
    
    # 提取分数用于统计
    scores = [item['score'] for item in data]
    
    # 计算统计信息
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    std_dev = np.std(scores)
    
    # 找到最高分和最低分对应的case名称
    max_idx = np.argmax(scores)
    min_idx = np.argmin(scores)
    
    # 输出结果
    print("\n===== 统计结果 =====")
    print(f"共处理 {len(data)} 个case")
    print(f"平均分数: {avg_score:.4f}")
    print(f"最高分数: {max_score:.4f} ({data[max_idx]['case_name']})")
    print(f"最低分数: {min_score:.4f} ({data[min_idx]['case_name']})")
    print(f"分数标准差: {std_dev:.4f}")
    
    # 绘制gFLOPS对比图
    plot_gflops_comparison(data)

if __name__ == "__main__":
    main()
