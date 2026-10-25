"""The production command selects only the explicitly declared nested protocol."""

import json
from pathlib import Path

import pytest

pytest.importorskip('torch')
from spectrome.mlp.training import run
from spectrome.mlp.training.helpers import DEFAULT_PARAMETERS, validate_run_inputs, make_outer_splits

PROTOCOL = Path(run.__file__).with_name('nested_protocol.json')


def test_cli_routes_nested_protocol_without_starting_batch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run, 'run_experiments', lambda: pytest.fail('Nested command must not run batch'))
    monkeypatch.setattr(run, 'run_experiment', lambda path, **kwargs: calls.append((path, kwargs)))
    cache, output = tmp_path / 'cache.pt', tmp_path / 'new-run'
    run.main(['--protocol', str(PROTOCOL), '--cache-source', str(cache), '--run-dir', str(output)])
    assert calls == [(output, {'cache_source': cache, 'protocol': json.loads(PROTOCOL.read_text()),
                               'training_runs_path': None})]
    assert not output.exists()


@pytest.mark.parametrize('arguments', [
    ['--protocol', str(PROTOCOL)], ['--run-dir', 'unused'],
    ['--protocol', str(PROTOCOL), '--run-dir', 'unused'],
])
def test_incomplete_nested_command_cannot_fall_back_to_batch(arguments, monkeypatch):
    monkeypatch.setattr(run, 'run_experiments', lambda: pytest.fail('Invalid command must stop'))
    with pytest.raises(SystemExit) as error:
        run.main(arguments)
    assert error.value.code == 2


def test_no_arguments_preserves_explicit_batch_api(monkeypatch):
    monkeypatch.setattr(run, 'run_experiments', lambda: 'batch-selected')
    assert run.main([]) == 'batch-selected'


def test_frozen_protocol_resolves_six_candidates_and_155_fits():
    requested = json.loads(PROTOCOL.read_text())
    preparation = dict.fromkeys(('data_dir', 'subjects', 'frequencies_hz', 'preprocessing_id'))
    _, protocol = validate_run_inputs(DEFAULT_PARAMETERS, preparation, 'unloaded-cache.pt', requested)
    assert [row['candidate_id'] for row in protocol['candidates']] == [
        f'basic_mlp_{i}' for i in (6, 10, 12, 13, 14, 15)]
    assert protocol['outer_strategy'] == 'grouped_kfold'
    assert protocol['inner_folds'] == 5
    assert protocol['split_seed'] == 1701 and protocol['training_seed'] == 7
    splits = make_outer_splits([f's{i}' for i in range(36)], strategy=protocol['outer_strategy'],
                               n_splits=protocol['outer_folds'], seed=protocol['split_seed'])
    assert len(splits) * (protocol['inner_folds'] * len(protocol['candidates']) + 1) == 155
    assert sorted(len(row['test_subject_ids']) for row in splits) == [7, 7, 7, 7, 8]
