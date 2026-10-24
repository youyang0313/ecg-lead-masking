"""Scientific data illustrations from the first preselected pilot ECG, not results."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from check_waveform_pilot import read_record, apply_configuration, CANONICAL, CONFIG, PILOT

sys.stdout.reconfigure(encoding="utf-8")
out=PILOT/"figures"
out.mkdir(exist_ok=True)
selection=json.loads((PILOT/"selection.json").read_text(encoding="utf-8"))
audit=json.loads((PILOT/"audit.json").read_text(encoding="utf-8"))
row=selection["records"][0]
signal,info=read_record(PILOT/"originals"/row["filename_lr"])
t=np.arange(1000)/100.0
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})

fig,axes=plt.subplots(12,1,figsize=(13,12),sharex=True)
for i,ax in enumerate(axes):
    ax.plot(t,signal[i],color="#154e75",lw=0.8)
    ax.set_ylabel(CANONICAL[i],rotation=0,labelpad=24,va="center")
    ax.grid(alpha=0.18)
    ax.tick_params(axis="y",labelsize=8)
    ax.margins(x=0)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("PTB-XL 1.0.3 | one preselected training recording\nECG "+str(row["ecg_id"])+" | annotations: "+", ".join(row["labels"]),fontsize=14,y=0.99)
fig.text(0.015,0.49,"Amplitude (mV); each lead has its own vertical scale",rotation=90,va="center",fontsize=10)
fig.text(0.5,0.012,"Real source data, 100 Hz. Illustration of input data, not a model prediction or a clinical interpretation.",ha="center",fontsize=9,color="#444444")
fig.tight_layout(rect=(0.035,0.028,1,0.95))
for ext in ["png","svg"]:
    fig.savefig(out/("training_ecg_example."+ext),dpi=160)
plt.close(fig)

mean=np.array(audit["pilot_scaler"]["mean_mV"])
std=np.maximum(np.array(audit["pilot_scaler"]["std_mV"]),1e-6)
z=((signal-mean[:,None])/std[:,None]).astype(np.float32)
configs=CONFIG["configurations"]
mask=np.array([[lead in c["available"] for lead in CANONICAL] for c in configs],dtype=int)
fig=plt.figure(figsize=(13,11))
grid=fig.add_gridspec(6,1,height_ratios=[1.8,1,1,1,1,1],hspace=0.6)
ax=fig.add_subplot(grid[0])
from matplotlib.colors import ListedColormap
ax.imshow(mask,aspect="auto",cmap=ListedColormap(["#f0f0f0","#1f6d88"]),vmin=0,vmax=1)
ax.set_xticks(np.arange(12));ax.set_xticklabels(CANONICAL)
ax.set_yticks(np.arange(5));ax.set_yticklabels([c["id"] for c in configs])
ax.set_title("Availability definitions: blue = retained, grey = unavailable",fontsize=11)
for a in range(5):
    for b in range(12):ax.text(b,a,str(mask[a,b]),ha="center",va="center",color="white" if mask[a,b] else "#666666",fontsize=8)
v2=CANONICAL.index("V2")
for i,c in enumerate(configs):
    ax=fig.add_subplot(grid[i+1])
    x,keep=apply_configuration(z,c["available"])
    present=bool(keep[v2])
    ax.plot(t,x[v2],lw=1,color="#154e75" if present else "#b46c16",linestyle="-" if present else "--")
    ax.set_xlim(0,10)
    low,high=z[v2].min(),z[v2].max();pad=max(float(high-low)*0.12,0.15)
    ax.set_ylim(min(float(low)-pad,-pad),max(float(high)+pad,pad))
    ax.set_ylabel(c["id"]+" / V2",rotation=0,labelpad=35,va="center")
    ax.grid(alpha=0.18)
    if not present:ax.text(0.98,0.72,"Removed channel represented by zero",transform=ax.transAxes,ha="right",fontsize=9,color="#9a5e12")
    if i==4:ax.set_xlabel("Time (s)")
    else:ax.tick_params(labelbottom=False)
fig.suptitle("Same real ECG across five input configurations\nV2 trace after pilot-only standardization",fontsize=14,y=0.99)
fig.text(0.02,0.37,"Standardized amplitude",rotation=90,va="center",fontsize=10)
fig.text(0.5,0.02,"ECG "+str(row["ecg_id"])+" | Mask demonstration only. A zeroed channel is a missing-input convention, not a physiological flatline.\nPilot scaling is not the final full-training-cohort scaling. No model was trained.",ha="center",fontsize=9,color="#444444")
fig.subplots_adjust(left=0.12,right=0.985,top=0.92,bottom=0.08)
for ext in ["png","svg"]:fig.savefig(out/("input_mask_example."+ext),dpi=160)
plt.close(fig)

(out/"figure_sources.json").write_text(json.dumps({"ecg_id":row["ecg_id"],"source_stem":row["filename_lr"],"fold":row["fold"],"record_choice":"first metadata-selected pilot record, chosen before waveform inspection","physical_unit":"mV","normalization":"pilot-only statistics for mask illustration, not main experiment parameters","kind":"real-data input illustrations; no experimental performance results","files":["training_ecg_example.png","training_ecg_example.svg","input_mask_example.png","input_mask_example.svg"]},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"plotted_ecg_id":row["ecg_id"],"figure_directory":str(out),"models_trained":False},ensure_ascii=False))
