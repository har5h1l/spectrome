"""Show a verified replay record with the original notebook's plotting method."""
from pathlib import Path
import sys, json, hashlib, warnings
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectrome.tests.test_historical_torch_parity import prepared_brain, transformed_spectra, FREQS, LPF, score
from spectrome.forward.network_transfer import network_transfer_local_alpha

evidence = json.loads((BASE/'figures/evidence.json').read_text())
for name, digest in evidence['source_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
saved = np.load(BASE/'figures/historical_spectra.npz')
s = int(saved['example_index'])
data = ROOT/'spectrome/data'
raw_c = loadmat(data/'individual_subjects.mat')['A_all_subjs_final']
raw_p = np.squeeze(loadmat(data/'SCFC_opparam_individual.mat')['output']['param'])
raw_meg = np.squeeze(loadmat(data/'freqMEGdata.mat')['freqMEGdata']['psd'])
retained = [i for i,x in enumerate(raw_p) if x.shape == (7,1)]
idx = retained[s]
params = dict(zip(('tau_e','tau_i','alpha','speed','gei','gii','tauC'), np.squeeze(raw_p[idx]).astype(float)))
meg = raw_meg[idx].astype(float)
brain = prepared_brain(raw_c[:,:,s], np.genfromtxt(data/'mean80_fiberlength.csv',delimiter=','))
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    model = np.array([network_transfer_local_alpha(brain,params,2*np.pi*f)[3][:68] for f in FREQS]).T
transformed = transformed_spectra(model)
np.testing.assert_allclose(transformed,saved['prediction'][s],rtol=1e-9,atol=1e-9)
empirical = np.array([np.sqrt(np.abs(np.convolve(x,LPF,'same'))) for x in meg])
empirical -= empirical.mean(axis=1,keepdims=True)
np.testing.assert_allclose(empirical,saved['empirical'][s],rtol=1e-12,atol=1e-12)
# Exact plotting operations from reproduce_MEG_spectra.ipynb, applied to replay arrays.
plt.style.use('seaborn-v0_8-paper')
fig,(ax_meg,ax_sim)=plt.subplots(1,2,figsize=(12,5))
mag2db=lambda y:20*np.log10(y)
for g in range(68):
    ax_meg.plot(FREQS,mag2db(np.convolve(meg[g],LPF,'same')),color='c',alpha=.3)
    ax_sim.plot(FREQS,mag2db(np.convolve(np.abs(model[g]),LPF,'same')),color='b',alpha=.3)
ax_meg.plot(FREQS,np.convolve(mag2db(meg.mean(axis=0)),LPF,'same'),color='r',linewidth=5)
ax_sim.plot(FREQS,np.convolve(mag2db(np.abs(model).mean(axis=0)),LPF,'same'),color='k',linewidth=5)
for ax,title in [(ax_meg,'Observed Spectrum'),(ax_sim,'Model Spectrum')]:
    ax.grid(True); ax.set_xlabel('Frequency (Hz)',fontsize=12); ax.set_ylabel('Magnitude (dB)',fontsize=12)
    ax.autoscale(enable=True,axis='x',tight=True); ax.set_title(title,fontsize=12)
fig.tight_layout()
fig.savefig(BASE/'figures/replay-spectrum.png',dpi=180)
fig.savefig(BASE/'figures/replay-spectrum.pdf')
np.savez(BASE/'figures/replay-example-raw.npz',frequencies=FREQS,empirical=meg,prediction=model)
provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'record_one_based':s+1,'meg_parameter_raw_index':idx,'structural_raw_index':s,'selection':evidence['example_selection'],'scope':'Single-record forward rerun, verified against saved 36-record replay arrays. Supplied fits and historical pairing retained.','plotting':'Notebook smoothing and dB conversion. All 68 observed/model regions, original cyan/blue curves and red/black means. No unit-norm normalization.','source_sha256':evidence['source_sha256'],'score':score(transformed,empirical),'transformed_max_difference_from_prior_replay':float(np.max(np.abs(transformed-saved['prediction'][s])))}
(BASE/'figures/replay-spectrum-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({k:provenance[k] for k in ['record_one_based','score','transformed_max_difference_from_prior_replay']}))
