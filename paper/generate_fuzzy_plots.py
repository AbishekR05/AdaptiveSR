import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

output_dir = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 2.6))

# 1. Real-Time Ratio Membership
r = np.linspace(0, 2.5, 500)
def trapz(x, a, b, c, d):
    return np.clip(np.minimum((x - a)/(b - a + 1e-9), (d - x)/(d - c + 1e-9)), 0, 1)
def tri(x, a, b, c):
    return np.clip(np.minimum((x - a)/(b - a + 1e-9), (c - x)/(c - b + 1e-9)), 0, 1)

mu_poor_r = np.where(r <= 0.5, 1.0, np.where(r >= 0.8, 0.0, (0.8 - r)/(0.3)))
mu_mod_r = tri(r, 0.6, 0.85, 1.1)
mu_good_r = np.where(r >= 1.2, 1.0, np.where(r <= 0.95, 0.0, (r - 0.95)/(0.25)))

ax1.plot(r, mu_poor_r, label='Poor', color='#d62728', lw=1.8)
ax1.plot(r, mu_mod_r, label='Moderate', color='#ff7f0e', lw=1.8)
ax1.plot(r, mu_good_r, label='Good', color='#2ca02c', lw=1.8)
ax1.set_title('(a) Real-Time Ratio ($r$)', fontsize=9, fontweight='bold')
ax1.set_xlabel('Ratio $r = t_{budget}/t_{infer}$', fontsize=8)
ax1.set_ylabel('Membership $\mu(r)$', fontsize=8)
ax1.legend(fontsize=7, loc='center right')
ax1.grid(True, linestyle=':', alpha=0.6)

# 2. Bandwidth Saving Membership
b = np.linspace(-30, 100, 500)
mu_low_b = np.where(b <= 0, 1.0, np.where(b >= 25, 0.0, (25 - b)/25.0))
mu_med_b = tri(b, 15, 35, 55)
mu_high_b = np.where(b >= 65, 1.0, np.where(b <= 45, 0.0, (b - 45)/20.0))

ax2.plot(b, mu_low_b, label='Low', color='#d62728', lw=1.8)
ax2.plot(b, mu_med_b, label='Medium', color='#ff7f0e', lw=1.8)
ax2.plot(b, mu_high_b, label='High', color='#2ca02c', lw=1.8)
ax2.set_title('(b) Bandwidth Saving ($b$)', fontsize=9, fontweight='bold')
ax2.set_xlabel('Saving Percentage (%)', fontsize=8)
ax2.legend(fontsize=7, loc='center right')
ax2.grid(True, linestyle=':', alpha=0.6)

# 3. Resource Headroom Membership
c = np.linspace(0, 1.0, 500)
mu_con_c = np.where(c <= 0.15, 1.0, np.where(c >= 0.35, 0.0, (0.35 - c)/0.20))
mu_mod_c = tri(c, 0.25, 0.50, 0.75)
mu_ava_c = np.where(c >= 0.85, 1.0, np.where(c <= 0.65, 0.0, (c - 0.65)/0.20))

ax3.plot(c, mu_con_c, label='Constrained', color='#d62728', lw=1.8)
ax3.plot(c, mu_mod_c, label='Moderate', color='#ff7f0e', lw=1.8)
ax3.plot(c, mu_ava_c, label='Available', color='#2ca02c', lw=1.8)
ax3.set_title('(c) Resource Headroom ($c$)', fontsize=9, fontweight='bold')
ax3.set_xlabel('Headroom $c = 1.0 - \max(U, M)$', fontsize=8)
ax3.legend(fontsize=7, loc='center left')
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
out_file = os.path.join(output_dir, "fuzzy_membership_plots.png")
plt.savefig(out_file, dpi=300)
plt.close()
print(f"Generated {out_file}")
