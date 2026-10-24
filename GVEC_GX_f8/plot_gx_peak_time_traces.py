from pathlib import Path
import json
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
OUT=P/'figures'
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
data={};report={}
for j,(case,label,color) in enumerate([('SLNAM_BEST','pre-SLAM','#0057B8'),('SLAM_BEST','SLAM','#D55E00')]):
 path=P/'gx_runs/refined_alpha9'/case/'alpha0p375pi/aLT3p00/gx.out.nc'
 with Dataset(path) as n:
  t=np.asarray(n.groups['Grids'].variables['time'][:],float)
  q=np.asarray(n.groups['Diagnostics'].variables['HeatFlux_st'][:,0],float)
 assert np.all(np.isfinite(q)) and np.all(np.diff(t)>0)
 data[case+'_time']=t;data[case+'_Q_i']=q
 a=axes[0,j];b=axes[1,j]
 a.plot(t,q,color=color,lw=1)
 b.plot(t,q,color=color,lw=1)
 a.set_title(label+r', $\alpha=3\pi/8$, $a/L_{T_i}=3$')
 a.set_xlim(0,t[-1]);a.set_ylim(bottom=0)
 b.set_xlim(400,t[-1]);b.set_title('Late-time zoom')
 a.axvspan(t[-100],t[-1],color=color,alpha=.10,label='Last 100 diagnostics')
 a.legend(frameon=False,fontsize=9)
 stats={}
 for lo,hi in [(300,400),(400,500),(500,600)]:
  mask=(t>=lo)&(t<hi)
  if mask.sum()>1:
   mean=float(q[mask].mean());std=float(q[mask].std());slope=float(np.polyfit(t[mask],q[mask],1)[0])
   stats[f'{lo}-{hi}']={'samples':int(mask.sum()),'mean':mean,'std':std,'slope':slope}
   b.hlines(mean,max(lo,400),hi,colors='black',linestyles='--',lw=.9)

 late=q[t>=400]
 b.set_ylim(float(late.min())-2.,float(late.max())+2.)
 mask=t>=500
 if mask.sum()>2:
  slope,intercept=np.polyfit(t[mask],q[mask],1)
  b.plot(t[mask],slope*t[mask]+intercept,color='red',ls=':',label='Linear trend, t≥500')
  b.legend(frameon=False,fontsize=9)
 report[case]={'source':str(path.relative_to(P)),'final_time':float(t[-1]),'last_100_start_time':float(t[-100]),'last_100_mean':float(q[-100:].mean()),'windows':stats}
 for ax in [a,b]:ax.set_xlabel('GX time');ax.set_ylabel(r'$Q_i/Q_{\mathrm{GB}}$');ax.grid(alpha=.2)
for ext in ['pdf','png']:fig.savefig(OUT/f'gx_Q_vs_t_alpha3pi8_peaks.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
np.savez_compressed(OUT/'gx_Q_vs_t_alpha3pi8_peaks.npz',**data)
(OUT/'gx_Q_vs_t_alpha3pi8_peaks_stats.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
