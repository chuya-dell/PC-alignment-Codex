"""v83 part A figures -> vault folder 図と表 (png).  usage: python v83_figs_A.py"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ('Yu Gothic', 'Meiryo', 'MS Gothic'):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f; break
plt.rcParams['axes.unicode_minus'] = False
ROOT = Path(__file__).resolve().parents[2]; R = ROOT / 'data/results/v83_new_shots_261008'
VAULT = Path(r'W:\GoogleDrive\chuya2816\Obsidian Vault\ラボノート\05_解析\261008_【未読】偽陽性ゼロ化_原因確定と標準経路\図と表')
t = pd.read_csv(R / 'A3_fp_by_field.csv')
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
cols = {1: '#1f77b4', 2: '#d95f02', 3: '#2ca02c'}
for a, (col, ttl) in zip(ax, [('pos91k_k10.5', '凍結 k=10.5(個/9.1万ピラー)'), ('pos91k_k8', 'k=8'), ('pos91k_k4', 'k=4')]):
    for kind, mk, ls in (('-0/-1', 'o', '--'), ('-0/-3', 's', '-')):
        for b in (1, 2, 3):
            q = t[(t.kind == kind) & (t.board == b)].sort_values('pos')
            a.plot(q.pos, q[col], marker=mk, ls=ls, color=cols[b], alpha=.85, label=f'基板{b} {kind}')
    a.set_title(ttl); a.set_xlabel('位置'); a.set_ylabel('1視野あたりの偽陽性(正側)'); a.grid(alpha=.3)
    if col == 'pos91k_k10.5': a.axhline(2, color='k', lw=1, ls=':'); a.text(1, 2.1, '合格ライン2個(仮置き)', fontsize=8)
ax[0].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(VAULT / 'A3_偽陽性_基板別位置別.png', dpi=130); plt.close()

# scar class rates
d = pd.read_csv(R / 'A3_fp_by_scar_class.csv')
order = ['in_mask', 'out_0-2', 'out_2-4', 'out_4-8', 'out_>8']
g = d.groupby(['kind', 'cls']).agg(n=('n', 'sum'), k4=('pos_k4', 'sum'), abn4=('abn_k4', 'sum')).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for kind, c in (('-0/-1', '#1f77b4'), ('-0/-3', '#d62728')):
    q = g[g.kind == kind].set_index('cls').loc[order]
    ax[0].plot(order, q.k4 / q.n * 1e4, 'o-', color=c, label=kind); ax[1].plot(order, q.abn4 / q.n * 1e4, 'o-', color=c, label=kind)
ax[0].set_ylabel('陽性率(1万ピラーあたり、k=4)'); ax[1].set_ylabel('異常ピラー(1万ピラーあたり、k=4)')
for a in ax: a.set_yscale('log'); a.set_xlabel('傷マスクからの距離'); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(VAULT / 'A3_傷からの距離別.png', dpi=130); plt.close()

# needle: focus vs apparent gouge shift
t2 = pd.read_csv(R / 'needle/landmark_residuals.csv'); fm = pd.read_csv(R / 'needle/focus_metrics_5.csv').set_index('name')
q = t2[(t2.i == '6') & (t2.variant == 'plain') & (t2['map'] == 'L') & (t2.landmark == 'gouge_whole')]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
x = [fm.loc['6', 'pillar_amp'] - fm.loc[j, 'pillar_amp'] for j in q.j]
ax[0].scatter(x, np.hypot(q.dx, q.dy), c='#d62728'); [ax[0].annotate(j, (xx, np.hypot(a_, b_))) for j, xx, a_, b_ in zip(q.j, x, q.dx, q.dy)]
ax[0].set_xlabel('ピラー像の振幅の低下(6.tif との差)'); ax[0].set_ylabel('傷(損傷部全体)の見かけのずれ(画素)'); ax[0].grid(alpha=.3)
ax[1].bar(range(4), [np.hypot(*q[q.j == j][['dx', 'dy']].values[0]) for j in q.j], tick_label=list(q.j), color='#d62728', label='損傷部全体')
hl = t2[(t2.i == '6') & (t2.variant == 'plain') & (t2['map'] == 'L') & (t2.cls == 'hline')].groupby('j').dy.median()
ax[1].bar(np.arange(4) + .35, [abs(hl[j]) for j in q.j], width=.3, color='#1f77b4', label='下端の線(線に直交する成分)'); ax[1].legend(); ax[1].set_ylabel('位置合わせ後のずれ(画素)'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(VAULT / 'A4_傷のずれとピント.png', dpi=130); plt.close()
print('ok')
