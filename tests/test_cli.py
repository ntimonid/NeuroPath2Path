import json

from neuropath2path.cli import main


def test_cli_classification(dataset_dir, tmp_path):
    output = tmp_path / "results" / "classification.json"
    assert main(["classify", str(dataset_dir), "--test-proportion", "0.4",
                 "--neighbors", "1", "--output", str(output), "--quiet"]) == 0
    saved = json.loads(output.read_text())
    assert saved["result"]["accuracy"] == 1
    assert len(saved["result"]["predictions"]) == 4
    assert saved["parameters"]["seed"] == 100


def test_cli_features_and_compare(dataset_dir, capsys):
    swc = str(next(dataset_dir.rglob("*.swc")))
    assert main(["features", swc]) == 0
    assert len(json.loads(capsys.readouterr().out)["result"]["paths"]) == 2
    assert main(["compare", swc, swc, "--mode", "raw"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["distance"] == 0


def test_cli_weights(dataset_dir, capsys):
    assert main(["weights", str(dataset_dir), "--limit-per-class", "2", "--quiet"]) == 0
    saved = json.loads(capsys.readouterr().out)
    weights = saved["result"]["alpha / beta"]["weights"]
    assert len(weights) == 6 and abs(sum(weights) - 1) < 1e-10
