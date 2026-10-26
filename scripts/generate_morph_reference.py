"""Execute pinned ElasticPath2Path routines in Octave to generate morph fixtures.

python scripts/generate_morph_reference.py [--source-directory /path/to/ElasticPath2Path]
Without a source directory, the pinned public MATLAB files are downloaded.
"""

import argparse
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuropath2path import extract_features, read_swc

COMMIT = "7e2c6926d8c6723593df4b041e31517767a63938"
BASE = f"https://raw.githubusercontent.com/50-Cent/ElasticPath2Path/{COMMIT}/"
FILES = ("curve_to_q.m", "q_to_curve.m", "pathresample.m", "visualMorphingNeuron.m")


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path)
    args = parser.parse_args()
    a = np.array([[0, 0, 0], [0.1, 0.3, 0.6], [2, -1, 3], [3, 4, 5], [6, 2, 7]])
    b = np.array([[0, 0, 0], [0.4, 0.2, 0.1], [1, 2, 1], [2, 3, 4], [2, 3, 4],
                  [4, 1, 2], [3, 2, 1], [4, 5, 2]])
    inputs = {"first_0": a, "second_0": b, "first_1": a, "second_1": b}
    with ZipFile(ROOT / "Intercellular_data.zip") as archive:
        names = sorted(n for n in archive.namelist() if "/Granule/" in n and n.endswith(".swc"))[:2]
        for key, name in zip(("first_2", "second_2"), names):
            tree = read_swc(StringIO(archive.read(name).decode("utf-8-sig")))
            path = extract_features(tree).paths[0].node_ids
            lookup = {int(node_id): i for i, node_id in enumerate(tree.nodes[:, 0])}
            inputs[key] = tree.nodes[[lookup[int(node)] for node in path], 2:5]
    inputs["times"] = np.array([0, 0.125, 0.5, 0.875, 1])
    inputs["sample_counts"] = np.array([11, 3, 64])
    with tempfile.TemporaryDirectory(prefix="elasticpath2path-reference-") as temporary:
        folder = Path(temporary)
        hashes = {}
        for name in FILES:
            if args.source_directory:
                content = (args.source_directory / name).read_bytes()
            else:
                with urlopen(BASE + name, timeout=20) as response:
                    content = response.read()
            hashes[name] = hashlib.sha256(content).hexdigest()
            (folder / name).write_bytes(content)
        savemat(folder / "inputs.mat", inputs)
        # Numerical block of visualMorphingNeuron.m, excluding plotting/video.
        # SRVF conversion, inversion, and resampling call unmodified .m files.
        script = f"""addpath({quote(folder)});
data=load({quote(folder / 'inputs.mat')});
results=cell(3,5);
for k=1:3
  first=data.(sprintf('first_%d',k-1)); second=data.(sprintf('second_%d',k-1));
  n=double(data.sample_counts(k)); dt=1/(n-1);
  a=pathresample(first,n,size(first,1)); b=pathresample(second,n,size(second,1));
  qa=curve_to_q(a,dt); qb=curve_to_q(b,dt);
  pa=q_to_curve(qa,dt); pb=q_to_curve(qb,dt);
  da=pa-repmat(pa(end,:),n,1)-a; db=pb-repmat(pb(end,:),n,1)-b;
  frames=zeros(n,3,length(data.times));
  for f=1:length(data.times)
    alpha=data.times(f);
    intermediate=q_to_curve((1-alpha)*qa+alpha*qb,dt);
    frames(:,:,f)=intermediate-repmat(intermediate(end,:),n,1)-((1-alpha)*da+alpha*db);
  end
  results(k,:)={{a,b,qa,qb,frames}};
end
save('-mat7-binary',{quote(folder / 'output.mat')},'results');
"""
        (folder / "reference.m").write_text(script)
        subprocess.run(["octave", "--no-gui", "--quiet", str(folder / "reference.m")], check=True)
        results = loadmat(folder / "output.mat")["results"]
        output = dict(inputs)
        for k in range(3):
            for j, field in enumerate(("sampled_first", "sampled_second", "q_first", "q_second", "frames")):
                value = results[k, j]
                output[f"{field}_{k}"] = value.transpose(2, 0, 1) if field == "frames" else value
        target = ROOT / "tests" / "data"
        np.savez_compressed(target / "elasticpath2path_reference.npz", **output)
        metadata = {"repository": "https://github.com/50-Cent/ElasticPath2Path", "commit": COMMIT,
                    "sha256": hashes,
                    "reference": "Unmodified curve_to_q, q_to_curve, pathresample in Octave; "
                                 "numerical frame assembly from visualMorphingNeuron, excluding video/plotting",
                    "real_neurons": names}
        (target / "elasticpath2path_reference.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved ElasticPath2Path reference fixtures in {target}")


if __name__ == "__main__":
    main()
