1.2 Paper Organization The remainder of this paper is organized as follows: Section 2 reviews the related work and existing limitations. Section 3 details the proposed ARA-Fed framework and its mathematical formulation. Section 4 outlines the operational pipeline. Section 5 describes the comprehensive experimental setup and the custom simulation environment. Section 6 presents the multi-modal simulation results and comparative analysis. Finally, Section 7 concludes the paper, and Section 8 outlines future research directions.import numpy as np
import matplotlib.pyplot as plt

# 1. DATA: Framework Capabilities Ratings (0 to 5)

categories = [
    'Multi-Modal Support\n(Vision, Text, Audio)', 
    'Network Profiling\n(Bandwidth/Latency)', 
    'Non-IID Granularity\n(Strict Partitioning)', 
    'Lightweight/Modular\n(Ease of Modification)', 
    'Algorithm Diversity\n(Supported Baselines)'
]
N = len(categories)

# تقييم المحاكيات (بناءً على الجدول اللي كتبناه)
# ARA-Fed واخد 5 في التخصص بتاعه
values_arafed = [5, 5, 5, 5, 5] 
# FedML (تقيل ومعقد شوية)
values_fedml = [5, 3, 4, 2, 5]   
# Flower (هندسي أكتر منه بحثي دقيق في الشبكات)
values_flower = [3, 2, 3, 4, 3]  
# LEAF (قديم شوية ومحدود في الداتا)
values_leaf = [2, 1, 5, 3, 2]    

# 2. PLOTTING LOGIC (Radar Chart)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1] # Close the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Helper to plot each framework
def add_to_radar(values, label, color, style):
    values += values[:1] # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle=style, label=label, color=color)
    ax.fill(angles, values, color=color, alpha=0.1)

# Plotting the Competitors
add_to_radar(values_leaf, 'LEAF (Benchmark)', 'gray', '--')
add_to_radar(values_flower, 'Flower (Deployment)', 'blue', '-.')
add_to_radar(values_fedml, 'FedML (Library)', 'green', ':')

# Plotting ARA-Fed (Ours) - The Best
add_to_radar(values_arafed, 'ARA-Fed (Ours)', 'red', '-')

# Styling
plt.xticks(angles[:-1], categories, color='black', size=10)
ax.set_rlabel_position(0)
plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=7)
plt.ylim(0, 5.5)

plt.title('Comparison of Simulation Framework Capabilities', size=15, weight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.tight_layout()
plt.savefig('Framework_Comparison_Radar.png', dpi=300)
plt.show()
