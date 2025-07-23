import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 设置中文字体（按优先级排列）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# 读取已有的复杂度评估结果
df = pd.read_csv('../results/RQ3/complexity_evaluation_results.csv')

# 将results列转换为数值形式
df["result_value"] = df["results"].apply(lambda x: eval(x)[0] if isinstance(x, str) else x)

# 计算CWE数量
df["cwe_count"] = df["found_cwes"].apply(lambda x: len(eval(x)) if isinstance(x, str) and x != "[]" else 0)

# 对overall score与任务结果(result_value)之间相关性的分析
print("\n===== Overall Complexity Score and Task Result Correlation Analysis =====")


# 分析各复杂度维度与结果之间的相关性
print("\n===== 各复杂度维度与结果之间的相关性分析 =====")
complexity_dimensions = ["complexity_conceptual", "complexity_implementation", 
                        "complexity_edge_case", "complexity_security", 
                        "complexity_performance", "complexity_overall"]

# ====== 成功与失败案例的模式比较分析 ======
print("\n\n===== 成功与失败案例的模式比较分析 =====")

# 将数据分为成功和失败两组
success_df = df[df["result_value"] == 1]
failure_df = df[df["result_value"] == 0]

print(f"\n成功组样本数: {len(success_df)}")
print(f"失败组样本数: {len(failure_df)}")

# 1. 比较两组的基本统计指标
print("\n1. 各组的基本统计指标比较:")
print("\n成功组的统计指标:")
success_stats = success_df[complexity_dimensions].describe()
print(success_stats)

print("\n失败组的统计指标:")
failure_stats = failure_df[complexity_dimensions].describe()
print(failure_stats)


def plot_cumulative_difference_analysis_individual(success_df, failure_df, complexity_dimensions):
    """
    为每个复杂度维度单独绘制累积分布差异分析图
    """
    plt.rcParams.update({
        'font.size': 22,        
        'axes.titlesize': 22,   
        'axes.labelsize': 22,   
        'xtick.labelsize': 22,  
        'ytick.labelsize': 22,  
        'legend.fontsize': 22,  
    })
    
    dimension_names = {
        "complexity_conceptual": "Conceptual Complexity",
        "complexity_implementation": "Implementation Complexity",
        "complexity_edge_case": "Edge Case Complexity",
        "complexity_security": "Security Complexity",
        "complexity_performance": "Performance Complexity", 
        "complexity_overall": "Overall Complexity"
    }
    
    for dim in complexity_dimensions:

        plt.figure(figsize=(10, 8))
        
        if len(success_df) > 0 and len(failure_df) > 0:
            success_data = success_df[dim].dropna()
            failure_data = failure_df[dim].dropna()
            
            if len(success_data) > 0 and len(failure_data) > 0:
                min_val = min(success_data.min(), failure_data.min())
                max_val = max(success_data.max(), failure_data.max())
                x_range = np.linspace(min_val, max_val, 200)
                
                success_cdf = np.array([np.mean(success_data <= x) for x in x_range])
                failure_cdf = np.array([np.mean(failure_data <= x) for x in x_range])
                
                plt.plot(x_range, success_cdf, color='green', linewidth=3, 
                        label=f'Success CDF')
                plt.plot(x_range, failure_cdf, color='red', linewidth=3, 
                        label=f'Failure CDF')
                
                difference = success_cdf - failure_cdf
                plt.plot(x_range, difference, color='blue', linewidth=3, linestyle=':', 
                        label='Difference (Success - Failure)')
                
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # 标记最大差异点
                max_diff_idx = np.argmax(np.abs(difference))
                max_diff_val = difference[max_diff_idx]
                max_diff_x = x_range[max_diff_idx]
                plt.plot(max_diff_x, max_diff_val, 'bo', markersize=12, 
                        label=f'Max |Diff|: {abs(max_diff_val):.3f}')
                
        readable_name = dimension_names.get(dim, dim)
        # plt.title(f'{readable_name} CDF Difference Analysis', fontsize=22, fontweight='bold')
        plt.xlabel('Complexity Score', fontsize=22)
        plt.ylabel('Cumulative Probability / Difference', fontsize=22)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=22, loc='best')
        
        plt.tight_layout()
        filename = f'cdf_difference_{dim}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存 {readable_name} 的累积分布差异分析图为 '{filename}'")
        plt.close() 

plot_cumulative_difference_analysis_individual(success_df, failure_df, complexity_dimensions)


