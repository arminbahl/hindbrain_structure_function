from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import truncnorm
import seaborn as sns


def truncated_normal(mean, std, low, upp, size):
    seed = 42
    rng = np.random.default_rng(seed)
    return truncnorm(
        (low - mean) / std, (upp - mean) / std, loc=mean, scale=std
    ).rvs(size, random_state=rng)



c1_length = truncated_normal(25, 10, 5, 100, 20)
c2_length = truncated_normal(14, 10, 5, 100, 20)
c3_length = truncated_normal(12, 10, 5, 100, 20)
c4_length = truncated_normal(15, 10, 5, 100, 20)
unclear_length = truncated_normal(29, 50, 5, 100, 100)

plt.figure(figsize=(10, 6))
sns.kdeplot(c1_length, label='C1', bw_adjust=1.2)
sns.kdeplot(c2_length, label='C2', bw_adjust=1.2)
sns.kdeplot(c3_length, label='C3', bw_adjust=1.2)
sns.kdeplot(c4_length, label='C4', bw_adjust=1.2)
sns.kdeplot(unclear_length, label='Unclear', bw_adjust=1.2)

plt.title('PDF of Lengths')
plt.xlabel('Length')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

scaler_only_train = StandardScaler()
scaler_only_test = StandardScaler()

scaler_only_train.fit(np.concatenate([c1_length, c2_length, c3_length, c4_length]).reshape(-1, 1))
scaler_only_test.fit(np.concatenate([unclear_length]).reshape(-1, 1))

c1_length_ot = scaler_only_train.transform(c1_length.reshape(-1, 1))
c2_length_ot = scaler_only_train.transform(c2_length.reshape(-1, 1))
c3_length_ot = scaler_only_train.transform(c3_length.reshape(-1, 1))
c4_length_ot = scaler_only_train.transform(c4_length.reshape(-1, 1))
unclear_length_ot= scaler_only_test.transform(unclear_length.reshape(-1, 1))
plt.figure(figsize=(10, 6))
sns.kdeplot(c1_length_ot, label='C1', bw_adjust=1.2)
sns.kdeplot(c2_length_ot, label='C2', bw_adjust=1.2)
sns.kdeplot(c3_length_ot, label='C3', bw_adjust=1.2)
sns.kdeplot(c4_length_ot, label='C4', bw_adjust=1.2)
sns.kdeplot(unclear_length_ot, label='Unclear', bw_adjust=1.2)
('scaledPDF of Lengths')
plt.xlabel('Length')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.kdeplot(c1_length_ot, label='C1', bw_adjust=1.2)
sns.kdeplot(c2_length_ot, label='C2', bw_adjust=1.2)
sns.kdeplot(c3_length_ot, label='C3', bw_adjust=1.2)
sns.kdeplot(c4_length_ot, label='C4', bw_adjust=1.2)
sns.kdeplot(scaler_only_train.transform(unclear_length.reshape(-1, 1)), label='Unclear', bw_adjust=1.2)
('scaledPDF of Lengths')
plt.xlabel('Length')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


scaler_all_data = StandardScaler()
scaler_all_data.fit(np.concatenate([c1_length, c2_length, c3_length, c4_length,unclear_length]).reshape(-1, 1))
c1_length_ad = scaler_all_data.transform(c1_length.reshape(-1, 1))
c2_length_ad = scaler_all_data.transform(c2_length.reshape(-1, 1))
c3_length_ad = scaler_all_data.transform(c3_length.reshape(-1, 1))
c4_length_ad = scaler_all_data.transform(c4_length.reshape(-1, 1))
unclear_length_ad = scaler_all_data.transform(unclear_length.reshape(-1, 1))
plt.figure(figsize=(10, 6))
sns.kdeplot(c1_length_ad, label='C1', bw_adjust=1.2)
sns.kdeplot(c2_length_ad, label='C2', bw_adjust=1.2)
sns.kdeplot(c3_length_ad, label='C3', bw_adjust=1.2)
sns.kdeplot(c4_length_ad, label='C4', bw_adjust=1.2)
sns.kdeplot(unclear_length_ad, label='Unclear', bw_adjust=1.2)
('scaledPDF of Lengths')
plt.xlabel('Length')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()