"""Paired stage summaries, including failure and unchanged/fallback cases."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.output
    names=['baseline','subpixel','lattice','iterative']
    frames={name:pd.read_csv(root/(name+'.csv')).set_index('case_id') for name in names if (root/(name+'.csv')).exists()}
    base=frames['baseline']; summaries=[]; comparisons=[]
    reference={r['case_id']:(r['sha256'],r['truth']) for r in json.loads((root/'baseline_inputs.json').read_text(encoding='utf-8'))}
    for name,df in frames.items():
        ledger={r['case_id']:(r['sha256'],r['truth']) for r in json.loads((root/(name+'_inputs.json')).read_text(encoding='utf-8'))}
        if reference!=ledger or set(base.index)!=set(df.index): raise RuntimeError(f'Unpaired inputs: {name}')
        df=df.loc[base.index]
        for label,selected in [('all',df),('nonidentity',df[df.scenario!='axis_00'])]:
            good=selected[selected.status=='ok']; row=dict(stage=name,subset=label,n=len(selected),failed=int((selected.status!='ok').sum()))
            for col in ['center_error_px','rotation_edge_error_px','scale_edge_error_px','grid_rmse_px']:
                for stat,fn in [('mean',np.mean),('median',np.median),('p95',lambda x:np.quantile(x,.95)),('max',np.max)]:
                    row[f'{col}_{stat}']=fn(good[col])
            summaries.append(row)
        if name=='baseline': continue
        previous=frames[names[names.index(name)-1]].loc[base.index]
        paired=(df.status=='ok')&(previous.status=='ok')
        delta=df.loc[paired,'grid_rmse_px']-previous.loc[paired,'grid_rmse_px']
        comparisons.append(dict(stage=name,previous=names[names.index(name)-1],n_paired=len(delta),
                         improved=int((delta< -1e-5).sum()),worse=int((delta>1e-5).sum()),
                         unchanged=int((abs(delta)<=1e-5).sum()),mean_change_px=delta.mean(),
                         median_change_px=delta.median(),worst_increase_px=delta.max()))
        df[['status','grid_rmse_px']].join(previous[['grid_rmse_px']],rsuffix='_previous').assign(change_px=delta).to_csv(root/(name+'_paired.csv'))
    pd.DataFrame(summaries).to_csv(root/'stage_summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(root/'stage_comparison.csv',index=False)
    print(pd.DataFrame(summaries)[['stage','subset','n','failed','grid_rmse_px_mean','grid_rmse_px_median','grid_rmse_px_p95']].to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    colors=['#667085','#2878b5','#d28424','#238b62']
    for (name,df),color in zip(frames.items(),colors):
        v=df.loc[(df.status=='ok')&(df.scenario!='axis_00'),'grid_rmse_px'].sort_values().to_numpy()
        axes[0].plot(v,np.arange(1,len(v)+1)/len(v),label=name,color=color,linewidth=2)
    axes[0].set(xscale='log',xlabel='Known-truth spatial RMSE (pixels)',ylabel='Cumulative fraction',title='All nonidentity cases (successful fits)')
    axes[0].grid(alpha=.2);axes[0].legend()
    if len(frames)>1:
        name=list(frames)[-1];final=frames[name].reindex(base.index)
        axes[1].scatter(base.grid_rmse_px,final.grid_rmse_px,s=13,alpha=.6,color=colors[len(frames)-1])
        axes[1].plot([1e-3,10],[1e-3,10],'--',color='#888888')
        axes[1].set(xscale='symlog',yscale='symlog',xlabel='Baseline RMSE (pixels)',ylabel=f'{name} RMSE (pixels)',title='Paired known-truth error')
        axes[1].grid(alpha=.2)
    fig.savefig(root/'precision_comparison.png',dpi=170)

if __name__=='__main__':main()
