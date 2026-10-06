// Mirror every Beamer overlay as a full-slide image, with editable speaker notes.
// Run a copy of this script from build/, which contains the runtime node_modules link.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const base = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const skill = '/Users/harshilshah/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const runtimePython = '/Users/harshilshah/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const { finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const pages = JSON.parse(await fs.readFile(path.join(base, 'build/page-metadata.json'), 'utf8'));
if (pages.length !== 18) throw new Error(`Expected 18 Beamer steps, found ${pages.length}`);
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
for (const page of pages) {
  const slide = presentation.slides.add();
  slide.background.fill = '#FFFFFF';
  const bytes = await fs.readFile(path.join(base, 'build/pdf-pages', `page-${String(page.page).padStart(2, '0')}.png`));
  slide.images.add({ blob: new Uint8Array(bytes), contentType: 'image/png',
    alt: `Beamer frame ${page.frame}, step ${page.step}. ${page.text}`,
    fit: 'contain', position: { left: 0, top: 0, width: 1280, height: 720 } });
  slide.speakerNotes.textFrame.setText([
    `Logical frame ${page.frame}, overlay ${page.step}.`,
    page.notes,
    'The visible slide is a rendered Beamer page. Edit src/main.tex and rebuild to change its content.',
    'Sources: Raj et al. (2020), https://pmc.ncbi.nlm.nih.gov/articles/PMC7336150/; local source hashes and fresh replay results in figures/evidence.json.',
  ].join('\n\n'));
}
const stage = path.join(base, 'build/pptx');
await fs.mkdir(stage, { recursive: true });
const candidatePath = path.join(stage, 'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const buildId = Date.now();
const finalPath = path.join(stage, `validated-${buildId}.pptx`);
const receiptPath = path.join(base, 'build', `validation-${buildId}.json`);
const result = await finalizePresentation({
  workspaceDir: base, candidatePath, finalPath,
  explicitTotalSlideCount: pages.length,
  requiredNativeTableOwnerSlides: [], requiredNativeChartOwnerSlides: [],
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000'],
  verifyArtifactToolImport: true,
  receiptPath,
});
await fs.copyFile(finalPath, path.join(base, 'presentation.pptx'));
await fs.copyFile(receiptPath, path.join(base, 'build/pptx-validation.json'));
console.log(JSON.stringify({ pages: pages.length, output: path.join(base, 'presentation.pptx'), validation: result }, null, 2));
