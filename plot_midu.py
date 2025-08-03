import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path_1 = r'D:\PYALL\MT4_move_bizhi\histogram_MT4_20_snr_2_model_RML201610A.npy'
path_2 = r'D:\PYALL\SP_CNN_bizhi\histogram_bizhi_50_snr_0_model_RML201610A.npy'
histogram_MT4_sum_move_2016B = np.load(path_1)
histogram_SP_sum_move_2016B = np.load(path_2)
# value_plot = np.vstack([histogram_MT4_sum_move_2016B, histogram_SP_sum_move_2016B])
# 绘制密度分布柱状图
# colors = ["black", "red"]
d = plt.figure(figsize=(8, 6))
# sns.kdeplot(histogram,shade=True,color='red')
sns.set(style="whitegrid", palette=["#cd4f27", "#745ea6", "#184991"])
# sns.set(style="whitegrid", palette=["#cd4f27", "#501DBA", "#184991"])
sns.kdeplot(histogram_MT4_sum_move_2016B, common_norm=False, fill=True)
# plt.text(0.7, 1.5, 'TAC', fontsize = 12, color ='#184991')
sns.kdeplot(histogram_SP_sum_move_2016B, common_norm=False, fill=True)
# plt.text(1, 1, 'MAC', fontsize = 12, color ='#cd4f27')
# plt.xlabel("cosine_similarity")
plt.ylabel("Density")
name_mx = 'MD_view_bizhi_snr_0_2016a.png'
# plt.xticks(x, [f"{min_similarity + i * interval:.2f}-{min_similarity + (i + 1) * interval:.2f}" for i in range(m)])
# plt.show()
plt.savefig(name_mx, transparent=True, dpi=800)
plt.close(d)