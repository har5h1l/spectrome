"""Extract the exact saved notebook figure; do not restyle or rerun the model."""
from pathlib import Path
import base64
import hashlib
import json

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
source = REPO / 'spectrome/notebooks/reproduce_MEG_spectra.ipynb'
notebook = json.loads(source.read_text())
data = notebook['cells'][8]['outputs'][0]['data']['image/png']
if isinstance(data, list):
    data = ''.join(data)
image = base64.b64decode(data)
output = BASE / 'figures/notebook-spectrum.png'
output.write_bytes(image)
provenance = {
    'source': str(source.relative_to(REPO)),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'cell_index': 8, 'output_index': 0, 'mime_type': 'image/png',
    'output_sha256': hashlib.sha256(image).hexdigest(),
    'transformation': 'None. Exact saved notebook PNG bytes.',
    'scope': 'Saved Figure 3 demonstration, separate from fresh cohort replay.',
    'notebook_inputs': {
        'meg': 'freqMEGdata_8002-101.h5',
        'connectome': 'Brain.add_connectome default template',
        'parameters': 'SCFC_opparam_individual.mat record 0',
    },
    'plot_preprocessing': 'Notebook smoothing and dB conversion retained.',
}
(BASE / 'figures/notebook-spectrum-provenance.json').write_text(
    json.dumps(provenance, indent=2) + '\n')
print('Extracted the original notebook figure without changes.')
