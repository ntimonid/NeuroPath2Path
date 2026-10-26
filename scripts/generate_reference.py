"""Regenerate numerical fixtures by executing the original .m functions in Octave.

Run from the repository root: python scripts/generate_reference.py
Requires Octave, NumPy, and SciPy. No Python port code produces reference values.
"""

from pathlib import Path
import json
import subprocess
import tempfile
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parents[1]


def matlab_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    entries = []
    with tempfile.TemporaryDirectory(prefix="neuropath2path-reference-") as temporary:
        folder = Path(temporary)
        paths = []
        for archive_name in ["Intercellular_data.zip", "Intracellular_data.zip"]:
            with ZipFile(ROOT / archive_name) as archive:
                members = sorted(n for n in archive.namelist() if n.lower().endswith(".swc"))
                labels = sorted({n.split("/")[1] for n in members})
                for label in labels[:2]:
                    member = next(n for n in members if n.split("/")[1] == label)
                    target = folder / f"{label}.swc"
                    target.write_bytes(archive.read(member))
                    paths.append(str(target))
                    entries.append({"archive": archive_name, "member": member})
        savemat(folder / "inputs.mat", {"file_paths": np.array(paths, dtype=object)})
        # Octave lacks histcounts; distanceNeuronNew only needs integer degrees.
        (folder / "histcounts.m").write_text("""function [counts,edges] = histcounts(values,nbins)
if isempty(values)
  counts=zeros(1,nbins); edges=0:nbins; return;
end
assert(all(values==fix(values)) && all(values>=1) && all(values<=nbins));
counts=accumarray(values(:),1,[nbins,1])';
edges=0.5:1:(nbins+0.5);
end
""")
        script = f"""addpath({matlab_string(ROOT)});
addpath({matlab_string(folder)});
load({matlab_string(folder / 'inputs.mat')});
features=cell(length(file_paths),1);
summaries=zeros(length(file_paths),3);
for k=1:length(file_paths)
  fid=fopen(file_paths{{k}});
  data=textscan(fid,'%f %f %f %f %f %f %f','CommentStyle','#');
  fclose(fid);
  [f,num,avl,pl]=computenP2P(cell2mat(data));
  features{{k}}=f; summaries(k,:)=[num,avl,pl];
end
results=cell(2,3);
for pair=1:2
  a=features{{2*pair-1}}; b=features{{2*pair}};
  [distance,vectors,pairs]=computeDST(a,b);
  results{{pair,1}}={{distance,vectors,pairs}};
  [distance,vectors,pairs]=computeDSTInter(a,b);
  results{{pair,2}}={{distance,vectors,pairs}};
  [distance,vectors,pairs]=computeDSTIntra(a,b);
  results{{pair,3}}={{distance,vectors,pairs}};
end
weight_between=diag([1,2,3,4,5,6]);
weight_within=cat(3,eye(6)*0.5,eye(6)*0.75);
rng(100); rnum=randi([1,10],6,1); weight_initial=rnum/sum(rnum);
weight_result=getFeatureWeights(weight_between,weight_within);
engine_version=version;
save('-mat7-binary',{matlab_string(folder / 'reference.mat')},'features','summaries', ...
     'results','weight_between','weight_within','weight_initial','weight_result','engine_version');
"""
        (folder / "reference.m").write_text(script)
        subprocess.run(["octave", "--no-gui", "--quiet", str(folder / "reference.m")], check=True)
        reference = loadmat(folder / "reference.mat")
        data = {}
        for k in range(4):
            feature = reference["features"][k, 0]
            data[f"summary_{k}"] = reference["summaries"][k]
            values, ids, offsets, node_offsets = [], [], [0], [0]
            for path in feature:
                values.append(np.column_stack([path[j].ravel() for j in range(7)]))
                ids.append(path[7].ravel().astype(np.int64))
                offsets.append(offsets[-1] + len(values[-1]))
                node_offsets.append(node_offsets[-1] + len(ids[-1]))
            data[f"features_{k}"] = np.concatenate(values)
            data[f"nodes_{k}"] = np.concatenate(ids)
            data[f"offsets_{k}"] = np.array(offsets)
            data[f"node_offsets_{k}"] = np.array(node_offsets)
        for pair in range(2):
            for m, mode in enumerate(["raw", "inter", "intra"]):
                result = reference["results"][pair, m]
                data[f"distance_{pair}_{mode}"] = result[0, 0]
                data[f"vectors_{pair}_{mode}"] = result[0, 1]
                data[f"pairs_{pair}_{mode}"] = result[0, 2].astype(np.int64) - 1
        for name in ("weight_between", "weight_within", "weight_initial", "weight_result"):
            data[name] = reference[name]
        destination = ROOT / "tests" / "data"
        destination.mkdir(exist_ok=True)
        np.savez_compressed(destination / "octave_reference.npz", **data)
        metadata = {
            "engine": "GNU Octave " + str(reference["engine_version"].item()),
            "source": "Unmodified repository MATLAB functions; histcounts integer-degree shim for Octave",
            "compatibility": "Original inter/intra tortuosity calculation; pair indices converted to zero-based",
            "neurons": entries,
        }
        (destination / "reference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved reference fixtures in {destination}")


if __name__ == "__main__":
    main()
