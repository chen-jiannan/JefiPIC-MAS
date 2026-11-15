"""Vector manuscript figures from archived results; no solver reruns or fitted curves."""
from pathlib import Path
import os,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
(OUT/'preview').mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,FancyArrowPatch
from matplotlib.path import Path as MplPath
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'legend.fontsize':7.2,
 'xtick.labelsize':8,'ytick.labelsize':8,'lines.linewidth':1.25,
 'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavusans',
 'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';GRAY='#777777'
EXPORT={'fig01_workflow':'figure2_workflow','fig02_geometry':'figure1_geometry',
 'fig03_static_getdp':'figure4_static_getdp','fig04_scuff_twoarm':'figure5_scuff_twoarm',
 'fig05_warpx_sphere':'figure6_warpx_sphere','fig06_rf_accuracy_stability':'figure7_rf_accuracy_stability',
 'fig07_performance':'figure3_performance'}
def save(fig,name):
 fig.savefig(FIG/(EXPORT[name]+'.pdf'),bbox_inches='tight',pad_inches=.045)
 fig.savefig(OUT/'preview'/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.045)
 plt.close(fig)
def title(ax,text):ax.set_title(text,loc='left',pad=6)
def grid(ax):ax.grid(alpha=.17,lw=.5);ax.set_axisbelow(True)

# Figure 1: revised execution path; boxes retain the original blue/grey styling.
fig,ax=plt.subplots(figsize=(6.7,6.9));ax.set(xlim=(0,1),ylim=(0,1.055));ax.axis('off')
boxes=[('Particle advance and source deposition',
 'GPU: particle push and surface events\nCPU: charge-conserving deposition of charge and face currents'),
 ('Retarded plasma fields (GPU)',
 'Update source history; evaluate the plasma field\nEvaluate E and B directly at off-grid PEC points'),
 ('Boundary solve (CPU)',
 'Add the imposed drive; form the mixed $E_t$/$B_n$ system\nSolve the fixed matrix; update Hermite source states'),
 ('Total-field evaluation (GPU)',
 'Recompute plasma plus electric-source fields\nAdd magnetic-source feedback at the same time node'),
 ('Diagnostics and field handoff',
 'Add the imposed drive to the particle-grid field\nRecord continuity, source states and independent $E_t$/$B_n$ checks')]
ys=[.83,.65,.47,.29,.11]
for (head,body),y in zip(boxes,ys):
 ax.add_patch(FancyBboxPatch((.045,y),.79,.13,boxstyle='round,pad=0.012,rounding_size=.014',facecolor='#F1F5F8',edgecolor=BLUE,lw=1.15))
 ax.text(.065,y+.102,head,weight='bold',va='center',fontsize=9)
 ax.text(.065,y+.052,body,va='center',linespacing=1.5,fontsize=8.1)
for y0,y1 in zip(ys[:-1],ys[1:]):
 ax.add_patch(FancyArrowPatch((.44,y0-.012),(.44,y1+.142),arrowstyle='-|>',mutation_scale=10.5,lw=1.15,color='#333333',shrinkA=0,shrinkB=0))
# A continuous return path with rounded corners; no disconnected arrow segments.
verts=[(.44,.098),(.44,.066),(.44,.048),(.458,.048),
 (.922,.048),(.94,.048),(.94,.066),(.94,1.002),
 (.94,1.020),(.922,1.020),(.458,1.020),(.44,1.020),(.44,1.002),(.44,.972)]
codes=[MplPath.MOVETO,MplPath.LINETO,MplPath.CURVE3,MplPath.CURVE3,
 MplPath.LINETO,MplPath.CURVE3,MplPath.CURVE3,MplPath.LINETO,
 MplPath.CURVE3,MplPath.CURVE3,MplPath.LINETO,MplPath.CURVE3,MplPath.CURVE3,MplPath.LINETO]
ax.add_patch(FancyArrowPatch(path=MplPath(verts,codes),arrowstyle='-|>',mutation_scale=10.5,lw=1.15,color='#333333',clip_on=False))
ax.text(.71,.015,'Next time node',va='center',ha='center',fontsize=8.5)
save(fig,'fig01_workflow')

# Figure 2: the actual enriched geometry, not the obsolete 900-source cloud.
g=dict(np.load(ROOT/'data/geometry.npz'))
fig,axs=plt.subplots(1,2,figsize=(6.7,2.9),layout='constrained',gridspec_kw={'width_ratios':[1.4,1]})
for ax in axs:
 for z0,z1 in [(-30,-1),(1,30)]:
  ax.add_patch(FancyBboxPatch((z0,-1),z1-z0,2,boxstyle='round,pad=0,rounding_size=1',facecolor='#F4F4F4',edgecolor='#555555',lw=.8,zorder=0))
 for key,label,color,marker,size in [('solve_obs_points','Collocation',GRAY,'.',4),('ant_src_points','Electric source',BLUE,'o',6),('holdout_points','Independent check',ORANGE,'x',7)]:
  p=g[key]*1e3;sel=np.abs(p[:,1])<.22
  ax.scatter(p[sel,2],p[sel,0],s=size,color=color,marker=marker,label=label,linewidths=.55,alpha=.8)
 p=g['ant_src_points'][::3]*1e3;sel=np.abs(p[:,1])<.22
 ax.scatter(p[sel,2],p[sel,0],s=17,color=GREEN,marker='+',linewidths=.8,label='Magnetic source')
 ax.set(xlabel='$z$ (mm)',ylabel='$x$ (mm)');grid(ax)
axs[0].set(xlim=(-31,31),ylim=(-1.8,2.65));title(axs[0],'(a) Rounded two-arm geometry; thin $y$ slab')
axs[1].set(xlim=(-4,4),ylim=(-1.45,1.45));title(axs[1],'(b) Feed-region detail')
axs[0].legend(ncol=2,loc='upper center',frameon=False,columnspacing=.9,handletextpad=.3)
axs[0].add_patch(Rectangle((-4,-1.45),8,2.90,fill=False,edgecolor='#222222',ls=(0,(4,2.5)),lw=1.0,zorder=8))
axs[0].annotate('Panel (b)',xy=(4,1.45),xytext=(9,1.70),fontsize=7.5,color='#222222',
 arrowprops=dict(arrowstyle='-',lw=.7,color='#444444'),va='bottom')
save(fig,'fig02_geometry')

# Figure 3: common-probe static analytic/FEM comparison.
static=json.loads((ROOT/'data/static_comparison.json').read_text())['summary']
fig,axs=plt.subplots(1,2,figsize=(6.7,2.55),layout='constrained')
curve_audit=[]
for method,label,color,marker,ls in [('MAS','MAS',BLUE,'o','-'),('GetDP_P2','GetDP (P2)',ORANGE,'s','--')]:
 rows=[x for x in static if x['method']==method]
 assert len(rows)==4,(method,len(rows))
 for ax,key,dist in zip(axs,['E_error_105_pct','E_error_110_pct'],['1.05','1.10']):
  xx=np.array([x['unknowns'] for x in rows]);yy=np.array([x[key] for x in rows])
  assert len(xx)==len(yy)==4 and np.isfinite(xx).all() and np.isfinite(yy).all() and np.all(yy>0)
  ax.loglog(xx,yy,marker=marker,color=color,label=label,ls=ls,ms=3.8,markerfacecolor='white' if method=='GetDP_P2' else color)
  curve_audit.append(dict(panel_radius_over_R=float(dist),method=method,points=len(xx),unknowns=xx.tolist(),E_error_percent=yy.tolist()))
  ax.set(xlabel='Solved scalar unknowns',ylabel='Relative vector E error (%)');title(ax,f'$r={dist}R$');grid(ax)
for ax in axs:
 assert len(ax.lines)==2 and all(len(line.get_xdata())==4 for line in ax.lines)
axs[0].set_title('(a) $r=1.05R$',loc='left');axs[1].set_title('(b) $r=1.10R$',loc='left');axs[0].legend(frameon=False)
save(fig,'fig03_static_getdp')
(OUT/'figure4_curve_audit.json').write_text(json.dumps(curve_audit,indent=2),encoding='utf-8')

# Figure 4: latest compatible MOT periodic response against actual SCUFF fields.
sc=dict(np.load(ROOT/'data/scuff_periodic_fields.npz'))
assert len(sc['points_m'])==652 and len(sc['SCUFF_E'])==332
fig,axs=plt.subplots(2,2,figsize=(6.7,4.6),layout='constrained')
for ax,key,unit,label in [(axs[0,0],'E',1,'E (V/m)'),(axs[0,1],'B',1e9,'B (nT)')]:
 x=np.arange(1,13)
 ax.plot(x,np.linalg.norm(sc['SCUFF_'+key][320:],axis=1)*unit,'o-',color=BLUE,ms=3,label='SCUFF-EM')
 ax.plot(x,np.linalg.norm(sc['scattered_'+key][640:],axis=1)*unit,'s--',color=ORANGE,ms=3,markerfacecolor='none',label='JefiPIC-MAS')
 ax.set(xlabel='Fixed exterior probe index',ylabel='Scattered '+label,xticks=[1,4,8,12]);grid(ax)
title(axs[0,0],'(a) Complex electric-field vector norm');title(axs[0,1],'(b) Complex magnetic-field vector norm')
phase=np.linspace(0,2,401);ex=np.exp(2j*np.pi*phase)
for ax,key,component,unit,ylabel in [(axs[1,0],'E',2,1,'Scattered $E_z$ (V/m)'),(axs[1,1],'B',1,1e9,'Scattered $B_y$ (nT)')]:
 for field,color,ls,label in [('SCUFF_',BLUE,'-','SCUFF-EM'),('scattered_',ORANGE,'--','JefiPIC-MAS')]:
  index=320 if field=='SCUFF_' else 640
  ax.plot(phase,np.real(sc[field+key][index,component]*ex)*unit,color=color,ls=ls,label=label)
 ax.set(xlabel='Harmonic phase, $t/T$',ylabel=ylabel);grid(ax)
title(axs[1,0],'(c) Reconstructed periodic field at probe 1');title(axs[1,1],'(d) Reconstructed periodic field at probe 1')
axs[0,0].legend(frameon=False)
save(fig,'fig04_scuff_twoarm')

# Figure 5: declared primary probes; all five probe errors remain in text/table.
w=dict(np.load(ROOT/'data/sphere_waveforms.npz'))
t=w['field_time_s']*1e9;mt=w['momentum_time_s']*1e9
fig,axs=plt.subplots(2,3,figsize=(6.7,4.15),layout='constrained')
for j,p in enumerate([1,2,3]):
 for method,color,ls,label in [('W',BLUE,'-','WarpX'),('J',ORANGE,'--','JefiPIC-MAS')]:
  axs[0,j].plot(t,w['E_'+method+'_metal'][:,p,2],color=color,ls=ls,label=label)
 axs[0,j].set(xlabel='Time (ns)',ylabel='$E_z$ (V/m)');title(axs[0,j],f'({chr(97+j)}) Probe P{p}');grid(axs[0,j])
for j,p in enumerate([1,3]):
 for method,color,ls in [('W',BLUE,'-'),('J',ORANGE,'--')]:axs[1,j].plot(t,w['B_'+method+'_metal'][:,p,1]*1e9,color=color,ls=ls)
 axs[1,j].set(xlabel='Time (ns)',ylabel='$B_y$ (nT)');title(axs[1,j],f'({chr(100+j)}) Probe P{p}');grid(axs[1,j])
for method,color,ls in [('W',BLUE,'-'),('J',ORANGE,'--')]:axs[1,2].plot(mt,(w['u_'+method+'_metal']-w['u_'+method+'_no_metal'])[:,2],color=color,ls=ls)
axs[1,2].set(xlabel='Time (ns)',ylabel=r'$\Delta\langle\gamma v_z\rangle$ (m/s)');title(axs[1,2],'(f) Electron feedback');grid(axs[1,2])
axs[0,0].legend(frameon=False,fontsize=6.7)
save(fig,'fig05_warpx_sphere')

# Figure 6: complete driven RF record and full (unprojected) holdout diagnostics.
cache=dict(np.load(ROOT/'data/rf_plot_data.npz'))
a={'time_s':cache['surface_time_s'],'region_names':cache['region_names']}
d={key:cache[key] for key in ['time_s','state_norm','tail_norm']}
res=cache['residual_square_sums'];den=cache['incident_square_sums']
audit=json.loads((ROOT/'data/rf_audit.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(6.7,4.6),layout='constrained')
for name,color,ls,label in [('global',BLUE,'-','Global subset'),('feed_tips',ORANGE,'--','Feed tips'),('feed_transition',GREEN,'-.','Feed transition'),('outer_rounded',GRAY,':','Outer rounded')]:
 j=a['region_names'].tolist().index(name)
 axs[0,0].plot(a['time_s']*3e6,100*np.sqrt(np.cumsum(res[:,j])/np.cumsum(den[:,j])),color=color,ls=ls,label=label)
axs[0,0].set(xlabel='Drive cycles',ylabel='Cumulative $E_t$ residual (%)');title(axs[0,0],'(a) Independent surface residuals');axs[0,0].legend(frameon=False,fontsize=6.6)
windows=audit['time_windows'];centers=np.arange(3);width=.24
for offset,color,key,label in [(-width,BLUE,'global','Global $E_t$'),(0,ORANGE,'worst','Worst regional $E_t$'),(width,GREEN,'B',r'$B_n$ / full incident $\mathbf{B}$')]:
 vals=[100*(x['regions']['global']['gate_value'] if key=='global' else x['worst_Et'] if key=='worst' else x['Bn_over_full_B']) for x in windows]
 axs[0,1].bar(centers+offset,vals,width,color=color,label=label)
axs[0,1].set(xticks=centers,xticklabels=['0.75–4','4–8','8–12'],xlabel='Separate drive-cycle windows',ylabel='Boundary residual (%)',ylim=(0,1.95));title(axs[0,1],'(b) Early, middle and late windows');axs[0,1].legend(frameon=False,fontsize=6.6)
axs[1,0].plot(d['time_s']*3e6,d['state_norm'],color=BLUE);axs[1,0].set(xlabel='Drive cycles',ylabel='Weighted state norm (A/m)');title(axs[1,0],'(c) Complete driven coupled response')
tail=d['tail_norm'];axs[1,1].plot(np.arange(len(tail)),tail/tail[0],color=BLUE);axs[1,1].set(xlabel='Homogeneous boundary updates',ylabel='Norm / initial tail norm');title(axs[1,1],'(d) Zero-input boundary-state tail')
for ax in axs.flat:grid(ax)
save(fig,'fig06_rf_accuracy_stability')

# Figure 7: unchanged grid-kernel benchmark and current full-case stage profile.
fig,axs=plt.subplots(1,2,figsize=(6.7,3.1),layout='constrained',gridspec_kw={'width_ratios':[1,1.35]})
x=[5,9,13,17]
axs[0].semilogy(x,[4.676,5.386,17.563,95.393],'o-',color=BLUE,ms=3,label='Atomic')
axs[0].semilogy(x,[5.739,12.584,29.959,59.171],'s--',color=ORANGE,ms=3,label='Observation-parallel')
axs[0].set(xlabel='Cells per coordinate, $n$',ylabel='Full-EM field call (ms)',xticks=x);title(axs[0],'(a) Grid-source component');axs[0].legend(frameon=False);grid(axs[0])
p=json.loads((ROOT/'data/rf_profile.json').read_text());st={v['stage']:v['mean_s']*1e3 for v in p['summary'] if v['phase']=='steady'}
labels=['Plasma pass + upload','Total pass + source upload','CPU MOT update','Independent surface checks','PEC field download','Deposition + transfers','Other work / accounting']
vals=[st[k] for k in ['plasma_field_pass1_and_history_upload','source_upload_and_total_field_pass2','cpu_mot_rhs_solve_and_state_update','independent_surface_EB_diagnostics','surface_EB_sample_download_and_units','deposition_including_transfers_and_continuity']]
vals.append(st['step_wall_instrumented']-sum(vals))
axs[1].barh(np.arange(len(vals)),vals,color=[BLUE,BLUE,ORANGE,GRAY,GRAY,GREEN,'#BBBBBB']);axs[1].invert_yaxis()
axs[1].set(yticks=np.arange(len(vals)),yticklabels=labels,xlabel='Mean instrumented time (ms/node)');title(axs[1],'(b) 5976-unknown coupled case')
axs[1].grid(axis='x',alpha=.17,lw=.5);axs[1].set_axisbelow(True)
save(fig,'fig07_performance')
(OUT/'figure_sources.json').write_text(json.dumps({'generator':'work/build_revision_figures_20261115_v2.py',
 'figure1':'Frozen enriched source/collocation/holdout coordinates; dashed rectangle equals the panel-b limits',
 'figure2':'Execution order in the qualified augmented RF driver; continuous return arrow',
 'figure3':'Unchanged grid-kernel component benchmark and compatible 5976-unknown stage profile',
 'figure4':'Static comparison summary: MAS and GetDP_P2, four nonempty points each per panel; see figure4_curve_audit.json',
 'figure5':'data/scuff_periodic_fields.npz; waveforms reconstructed from phasors',
 'figure6':'data/sphere_waveforms.npz; no gain/phase/time fitting; all-probe errors retained',
 'figure7':'compatible_return_20261115_131122_714744/coupled_RF saved fields and state history'},indent=2),encoding='utf-8')
print('Generated seven vector PDF figures in',FIG)
print('SCUFF waveform fixed probe 1 position (mm):',sc['points_m'][640]*1e3)
