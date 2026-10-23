"""Assemble the Beamer update from current code and saved results; no training."""
from pathlib import Path
import csv, json, hashlib, shutil, statistics, textwrap
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
queue=json.loads((ROOT/'spectrome/training/experiments.json').read_text())
manifest={'sources':{},'runs':[], 'batch_status':'Runs 13–15 complete; collected and verified October 18, 2026. Run 16 was not in this batch and has no output folder.'}
def track(p):
 manifest['sources'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
def snippet(name,path,start,end):
 p=ROOT/path; track(p); source=p.read_text(); text=source[source.index(start):source.index(end,source.index(start))]
 (BASE/f'src/snippets/{name}.py').write_text(textwrap.dedent(text).rstrip()+'\n')
 return name
snippet('graphs','spectrome/training/data.py','        graphs = []','            graph = {')
snippet('local','spectrome/forward/network_transfer_torch.py','    Fe_local =','    Hid =')
snippet('layers','spectrome/training/model.py','        layers, width =','        self.network =')
p=ROOT/'spectrome/training/model.py'; track(p)
(BASE/'src/snippets/input.py').write_text('components = self.network(\n    (omega / self.omega_scale).unsqueeze(-1))\nreturn torch.complex(\n    components[..., 0], components[..., 1])\n')
snippet('response','spectrome/forward/network_transfer_torch.py','    value = response','    threshold =')
snippet('floor','spectrome/forward/network_transfer_torch.py','    threshold =','    K = response')
snippet('readout','spectrome/forward/network_transfer_torch.py','    K = response','    return frequency_response')
snippet('downstream','spectrome/training/spectra.py','        _, output = nt.evaluate_spectral_response_torch','    device = graphs[0]')
snippet('transform','spectrome/training/spectra.py','    magnitude =','\ndef spectral_correlation')
snippet('init','spectrome/training/train.py','        torch.manual_seed(seed)','        history = []')
snippet('loss','spectrome/training/train.py','            filters = model(omega)','            if not torch.isfinite(loss)')
snippet('evaluate','spectrome/training/train.py','        model.eval()','            analytical_training_score =')
(BASE/'src/snippets/update.py').write_text('optimizer.zero_grad()\n# Compute training loss\nloss.backward()\n# Check finite gradients\noptimizer.step()\n')
# This excerpt uses line wrapping; comments identify omitted checks.
old=(ROOT/'artifacts/10-5-presentation/src/main.tex').read_text()
preamble=old.split('\\newcommand{\\TanhPlot}')[0]
preamble=preamble.replace('Paper Reproduction, Torch Functions, and MLP Experiment','MLP Implementation, Training, and Results')
preamble+=r'''
\usepackage{listings}
\lstset{language=Python,basicstyle=\ttfamily\fontsize{9}{11}\selectfont,
 keywordstyle=\color{DeckBlue},commentstyle=\color{DeckGray},stringstyle=\color{DeckGreen},
 columns=fullflexible,keepspaces=true,showstringspaces=false,breaklines=true,
 aboveskip=0pt,belowskip=0pt}
\newcommand{\stepcontent}[4]{
\node[txt,font=\large\bfseries,text width=14cm] at (0.85,1.55) {#1};
\node[smalltxt,text width=6.15cm] at (0.85,2.45) {#2};
\node[txt,text width=7.25cm] at (7.75,3.2) {\begin{minipage}{7.25cm}\hyphenpenalty=0\exhyphenpenalty=0\lstinputlisting[linewidth=7.25cm]{snippets/#3.py}\end{minipage}};
\node[labeltxt,text width=7.25cm] at (7.75,2.35) {#4};}
\begin{document}
\begin{frame}[plain,noframenumbering]
\begin{canvas}
\node[txt,font=\LARGE\bfseries,text width=13.7cm] at (0.9,2.05) {SGM Graph Filter\\Progress Update};
\draw[DeckGreen,line width=1.5pt] (0.9,3.7) -- (2.4,3.7);
\node[txt,font=\large,text width=14cm] at (0.9,4.18) {MLP Implementation, Training, and Results};
\node[smalltxt] at (0.9,5.45) {Harshil Shah\\[0.18cm]Mission San Jose High School\\[0.18cm]October 19, 2026};
\end{canvas}
\end{frame}
'''
frames=[]
def stages(title,items):
 out=[r'\begin{frame}[t]{'+title+'}',r'\begin{canvas}']
 for i,(heading,left,snip,source) in enumerate(items,1):
  out.append(r'\only<'+str(i)+r'>{\stepcontent{'+heading+'}{'+left+'}{'+snip+'}{'+source+'}}')
 out += [r'\end{canvas}',r'\note{Code excerpts follow the current repository. Line wrapping and explicitly marked omissions are for readability.}',r'\end{frame}']
 frames.append('\n'.join(out))
stages('MLP Implementation',[
 ('1) Compute inputs offline',r'For each subject and frequency, we compute the delayed graph and its eigenmodes once, then save them for training.\\[0.4cm]We use 36 subjects and 40 frequencies from 2--45 Hz. The graph has 86 nodes, with 68 cortical regions used for evaluation.','graphs',r'\code{training/data.py}\\[0.12cm]\code{generate_training_data}'),
 ('2) Build the MLP',r'\textbf{Input:} normalized frequency, $f/45$.\\[0.35cm]\textbf{Output:} two real values, $a$ and $b$, which form the complex filter $a+bi$.\\[0.35cm]\textbf{Hyperparameters:} number of hidden layers, units per layer, and activation (Tanh or Sigmoid).\\[0.35cm]We vary these settings across experiments and learn only the MLP weights.','layers',r'\code{training/model.py}\\[0.12cm]\code{FeGraphMLP.__init__}'),
 ('3) Predict the complex filter',r'We scale frequency before passing it into the MLP:\\[0.25cm]$x=\dfrac{\omega}{2\pi\cdot45}=\dfrac{f}{45}$.\\[0.35cm]The two outputs give the real and imaginary parts:\\[0.25cm]$F_\theta(\omega)=a_\theta(x)+i b_\theta(x)$.\\[0.35cm]Within each fold, we use the same filter for every subject. Frequency is the only input.','input',r'\code{training/model.py}\\[0.12cm]\code{FeGraphMLP.forward}'),
 ('4) Pass the filter into the SGM',r'We pass $a+bi$ as \code{Fe_graph}, replacing $F_e$ only in graph propagation.\\[0.4cm]The existing SGM code handles the denominator safeguard, combines the graph modes, and selects the cortical regions.\\[0.4cm]This gives 68 regional spectra at 40 frequencies. The local response and fitted SGM parameters stay fixed.','downstream',r'\code{training/spectra.py}\\[0.12cm]\code{predict_spectra}')])
stages('Training and Evaluation',[
 ('1) Leave one subject out',r'One \textbf{fold} is one train-and-test round.\\[0.3cm]\textbf{Fold 1:} train on subjects 2--36; test on subject 1.\\[0.3cm]\textbf{Fold 2:} start again, train on everyone except subject 2; test on subject 2.\\[0.3cm]Repeat until every subject has been held out once. Each round starts with fresh weights and a fresh optimizer, using seed 7.','init',r'\code{training/train.py}\\[0.12cm]\code{train_filter}'),
 ('2) Prepare spectra for comparison',r'Within a fold, we compare each training subject\textquotesingle s predicted spectrum with their measured MEG spectrum.\\[0.3cm]Before calculating correlation, we process both in the same way:\\[0.2cm]1. Take the magnitude.\\[0.12cm]2. Smooth across neighboring frequencies.\\[0.12cm]3. Take the square root.\\[0.12cm]4. Subtract the mean across frequencies.','transform',r'\code{training/spectra.py}\\[0.12cm]\code{transform_spectra}'),
 ('3) Calculate the loss',r'We calculate correlation across frequency for each region, then average the 68 regions:\\[0.3cm]$r_s=\dfrac1{68}\displaystyle\sum_{r=1}^{68}\mathrm{corr}_f(\widehat x_{s,r},y_{s,r})$.\\[0.35cm]The loss averages the 35 training subjects:\\[0.3cm]$\mathcal L=1-\dfrac1{35}\displaystyle\sum_{s\in\mathrm{train}}r_s$.\\[0.35cm]The held-out subject does not enter this loss.','loss',r'\code{training/train.py}\\[0.12cm]\code{train_filter}'),
 ('4) Update the MLP weights',r'Each epoch makes one Adam update using all 35 training subjects. We record the loss before that update.\\[0.4cm]We run for a fixed 100, 500, or 5,000 epochs.\\[0.4cm]The initial learning rate is $10^{-3}$; the three follow-up runs use $3\times10^{-4}$.','update',r'\code{training/train.py}\\[0.12cm]\code{train_filter} (checks omitted)'),
 ('5) Evaluate the held-out subject',r'After training, we apply the learned filter to the subject left out of that fold and compare the predicted spectra with their MEG.\\[0.35cm]We use the same spectrum processing and correlation calculation. This score never updates the MLP or selects a checkpoint.\\[0.35cm]After all 36 folds, each subject has one held-out score. We average those 36 scores to report performance.','evaluate',r'\code{training/train.py}\\[0.12cm]\code{train_filter}')])
# All quantitative content is recomputed from saved CSVs.
track(ROOT/'spectrome/training/experiments.json')
track(ROOT/'artifacts/server-launch/basic_mlp_13_15/local_verification.json')
records=[]
for i,e in enumerate(queue):
 p=ROOT/'results/experiments'/e['results_dir']; item={'run':i,'settings':e['parameters']}
 if (p/'tables/subject_scores.csv').exists():
  c=json.loads((p/'config.json').read_text()); assert c['status']=='complete'
  assert all(c['model_config'][k]==e['parameters'][k] for k in ('hidden_sizes','omega_scale'))
  # Run 0 predates the activation field; its implementation default was Tanh.
  assert c['model_config'].get('activation','tanh')==e['parameters']['activation']
  assert all(c['training_config'][k]==e['parameters'][k] for k in ('epochs','learning_rate','seed'))
  rows=list(csv.DictReader((p/'tables/subject_scores.csv').open())); assert len(rows)==36
  for f in ['config.json','tables/subject_scores.csv']: track(p/f)
  item.update(r=statistics.mean(float(r['learned_correlation']) for r in rows),baseline=statistics.mean(float(r['analytical_correlation']) for r in rows),wins=sum(float(r['correlation_difference'])>0 for r in rows),status='complete')
  item['delta']=item['r']-item['baseline']
  for name in ['training_curves','spectra_across_subjects']:
   src=p/f'figures/summary/{name}.png'; track(src); shutil.copyfile(src,BASE/f'figures/run_{i}_{name}.png')
 else: raise ValueError(f'Missing completed subject scores: {p}')
 records.append(item)
manifest['runs']=records

def tableframe(title,body,caption,caption_y=7.65):
 frames.append(r'\begin{frame}[t]{'+title+r'}\begin{canvas}\node[txt,text width=14cm,align=center] at (0.85,1.55) {'+body+r'};\node[labeltxt,text width=14cm,align=center] at (0.85,'+str(caption_y)+r') {'+caption+r'};\end{canvas}\end{frame}')
for group,label in [(records[:12],'Initial comparison'),(records[12:],'Larger model and follow-ups')]:
 rows=[]
 for d in group:
  s=d['settings']; width='['+', '.join(map(str,s['hidden_sizes']))+']'; width=r'$5\times128$' if d['run'] in (12,13) else width
  rows.append(f"{d['run']} & {width} & {s['activation'].title()} & {s['epochs']:,} & {s['learning_rate']:g} & {d['status'].title()} \\\\")
 body=r'\footnotesize\renewcommand{\arraystretch}{1.04}\begin{tabular}{r l l r r l}\toprule Run & Hidden layers & Activation & Epochs & Learning rate & Status\\\midrule'+'\n'.join(rows)+r'\bottomrule\end{tabular}'
 cap='' if len(group)>4 else r'Runs 12 and 13 differ only in learning rate. Runs 13--15 use the same epoch budget.\\All three follow-ups are complete. Run 16 was outside this batch.'
 tableframe('Experiments --- '+label,body,cap)
rows=[]
for d in records:
 if d['status']!='complete': continue
 vals=f"{d['r']:.4f} & {d['delta']:+.4f} & {d['wins']}/36" if d['status']=='complete' else r'\multicolumn{3}{l}{'+d['status'].title()+r' --- results pending}'
 rows.append(str(d['run'])+' & '+vals+r'\\')
body=r'\fontsize{9}{10}\selectfont\renewcommand{\arraystretch}{0.78}\begin{tabular}{r r r r}\toprule Run & Mean held-out $r$ & $\Delta r$ vs. analytical & Subjects improved\\\midrule Analytical & 0.8524 & --- & ---\\\midrule'+'\n'.join(rows)+r'\bottomrule\end{tabular}'
tableframe('Performance',body,r'Exploratory single-seed comparisons; selecting a winner from these scores is model selection.',caption_y=8.1)
for d in records:
 i=d['run']; s=d['settings']; width=r'$5\times128$' if i in (12,13) else '['+', '.join(map(str,s['hidden_sizes']))+']'
 settings=f"{s['epochs']:,} epochs; Adam {s['learning_rate']:g}; seed 7"
 if d['status']!='complete':
  frames.append(r'\begin{frame}[t]{Run '+str(i)+' --- '+width+' '+s['activation'].title()+r'}\begin{canvas}'+r'\node[smalltxt] at (0.85,1.65) {'+settings+r'};\node[txt,font=\Large\bfseries] at (0.85,3.0) {'+d['status'].title()+r' --- results pending};\node[smalltxt,text width=13.7cm] at (0.85,4.15) {Same frequency-only input, fixed SGM parameters, and 36-fold LOOCV.\\[0.35cm]Training curves and held-out spectra will be added after completion.};\node[labeltxt] at (0.85,7.3) {Status checked October 18, 2026.};\end{canvas}\end{frame}')
  continue
 frames.append(r'\begin{frame}[t]{Run '+str(i)+' --- '+width+' '+s['activation'].title()+r'}\begin{canvas}'+
 r'\node[smalltxt] at (0.85,1.55) {'+settings+r'};'+
 r'\node[txt] at (0.35,2.2) {\includegraphics[width=7.45cm,height=4.45cm,keepaspectratio]{../figures/run_'+str(i)+r'_training_curves.png}};'+
 r'\node[txt] at (8.0,2.2) {\includegraphics[width=7.45cm,height=4.45cm,keepaspectratio]{../figures/run_'+str(i)+r'_spectra_across_subjects.png}};'+
 r'\node[smalltxt,text width=14cm] at (0.85,6.75) {Held-out $r$: \textbf{'+f"{d['r']:.4f}"+r'}\qquad $\Delta r$: '+f"{d['delta']:+.4f}"+r'\qquad Improved: '+str(d['wins'])+r'/36 subjects};'+
 r'\node[labeltxt,text width=14cm] at (0.85,7.45) {Left: training loss for each fold and their mean. Right: normalized spectral shape, mean $\pm$ SD across subjects.};'+r'\end{canvas}\note{Source: results/experiments/basic\_mlp\_'+str(i)+r'. The spectral plot averages normalized regions within subject, then subjects equally. SD is across subjects, not a confidence interval. Training loss is before each optimizer step.}\end{frame}')
(BASE/'src/main.tex').write_text(preamble+'\n\n'.join(frames)+'\n\\end{document}\n')
(BASE/'figures/evidence.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Assembled',len(frames)+1,'logical frames; 29 overlay pages expected.')
