"""SRVF path morphing from ElasticPath2Path and NeuroPath2Path §3.4.

For a piecewise-linear curve, velocity and its square-root representation
are constant on each parameter interval. Linear interpolation of these SRVFs
and exact interval integration implement the paper's formula without numerical
differentiation at vertices (method='paper'). The default 'elasticpath2path'
method reproduces the earlier tool's vertex gradients, midpoint resampling,
and endpoint reconstruction corrections. No optimal rotation is imposed:
visualMorphingNeuron.m computes a rotation but does not apply it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .distance import compute_distance
from .features import extract_features
from .io import Morphology, morphology_from_array


@dataclass(frozen=True)
class PathMorph:
    parameter: np.ndarray        # sample positions in [0, 1]
    times: np.ndarray            # morph fractions in [0, 1]
    frames: np.ndarray           # (frames, sample points, dimensions)
    method: str
    source_curve: np.ndarray | None = None
    target_curve: np.ndarray | None = None


@dataclass(frozen=True)
class NeuronMorph:
    """A collection of independently morphed paths, not a connected SWC tree."""

    times: np.ndarray
    pairs: np.ndarray
    paths: tuple[PathMorph, ...]
    added_pairs: np.ndarray      # coverage repairs after assignment reassignment
    source_node_ids: tuple[np.ndarray, ...]
    target_node_ids: tuple[np.ndarray, ...]
    source_root: np.ndarray
    target_root: np.ndarray
    source_axis_order: np.ndarray
    target_axis_order: np.ndarray
    method: str
    arbor: str = "dendrite"
    path_kinds: tuple[str, ...] = ()  # one compartment label per matched pair


def _curve(points: np.ndarray) -> np.ndarray:
    curve = np.asarray(points, dtype=float)
    if curve.ndim != 2 or len(curve) < 2 or curve.shape[1] < 1:
        raise ValueError("A curve must have shape (at least 2 points, dimensions)")
    if not np.isfinite(curve).all():
        raise ValueError("Curve coordinates must be finite")
    return curve


def _parameter(parameter: np.ndarray | None, n: int) -> np.ndarray:
    grid = np.linspace(0, 1, n) if parameter is None else np.asarray(parameter, dtype=float)
    if (grid.shape != (n,) or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)):
        raise ValueError("Parameter must be finite, strictly increasing, and match the curve")
    return grid


def _times(times: np.ndarray | None) -> np.ndarray:
    times = np.linspace(0, 1, 21) if times is None else np.asarray(times, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(times > 1)):
        raise ValueError("Morph times must be a nonempty vector of finite values in [0, 1]")
    return times.copy()


def resample_path(curve: np.ndarray, samples: int) -> np.ndarray:
    """Sequential midpoint insertion/local-length deletion from pathresample.m."""
    curve = _curve(curve)
    if not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    if len(curve) > samples:
        lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        remove = np.argsort(lengths[:-1] + lengths[1:], kind="stable")[:len(curve) - samples]
        return np.delete(curve, remove + 1, axis=0)
    result = curve.copy()
    while len(result) < samples:
        additions = min(len(result) - 1, samples - len(result))
        prefix = np.empty((2 * additions, result.shape[1]))
        prefix[::2] = result[:additions]
        prefix[1::2] = (result[:additions] + result[1:additions + 1]) / 2
        result = np.vstack((prefix, result[additions:]))
    return result


def curve_to_q(curve: np.ndarray, dt: float) -> np.ndarray:
    """Vertex SRVF counterpart of ElasticPath2Path/curve_to_q.m."""
    curve = _curve(curve)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    velocity = np.gradient(curve, dt, axis=0)
    magnitude = np.sqrt(np.linalg.norm(velocity, axis=1))
    q = velocity * 0.0001
    np.divide(velocity, magnitude[:, None], out=q, where=magnitude[:, None] > 0.0001)
    return q


def q_to_curve(q: np.ndarray, dt: float) -> np.ndarray:
    """Cumulative-sum inverse used by ElasticPath2Path/q_to_curve.m.

    This intentionally follows the original rectangular integration convention;
    use srvf_to_curve for the exact interval-based inverse.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or not len(q) or not q.shape[1] or not np.isfinite(q).all():
        raise ValueError("SRVF must be a nonempty finite point-by-dimension matrix")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    return np.cumsum(q * np.linalg.norm(q, axis=1)[:, None], axis=0) * dt


def _interpolate_elasticpath2path(first, second, times, samples, center):
    samples = max(len(first), len(second)) if samples is None else samples
    a, b = resample_path(first, samples), resample_path(second, samples)
    if center:
        a, b = a - a[0], b - b[0]
    grid = np.linspace(0, 1, samples)
    dt = 1 / (samples - 1)
    qa, qb = curve_to_q(a, dt), curve_to_q(b, dt)
    recovered_a, recovered_b = q_to_curve(qa, dt), q_to_curve(qb, dt)
    deviation_a = recovered_a - recovered_a[-1] - a
    deviation_b = recovered_b - recovered_b[-1] - b
    frames = []
    for alpha in times:
        recovered = q_to_curve((1 - alpha) * qa + alpha * qb, dt)
        deviation = (1 - alpha) * deviation_a + alpha * deviation_b
        frames.append(recovered - recovered[-1] - deviation)
    return PathMorph(grid, times, np.array(frames), "elasticpath2path", a.copy(), b.copy())


def curve_to_srvf(curve: np.ndarray, parameter: np.ndarray | None = None) -> np.ndarray:
    """Return interval SRVFs q = f' / sqrt(||f'||), with q=0 at zero speed.

    For N curve vertices, the result has N-1 rows. Values describe intervals,
    not derivatives sampled at vertices. The inverse below uses the same grid.
    """
    curve = _curve(curve)
    grid = _parameter(parameter, len(curve))
    velocity = np.diff(curve, axis=0) / np.diff(grid)[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    q = np.zeros_like(velocity)
    np.divide(velocity, np.sqrt(speed)[:, None], out=q, where=speed[:, None] > 0)
    return q


def srvf_to_curve(q: np.ndarray, parameter: np.ndarray | None = None,
                  origin: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct f(t)=origin+integral q(s)||q(s)|| ds exactly on each interval."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or not len(q) or q.shape[1] < 1 or not np.isfinite(q).all():
        raise ValueError("SRVF must be a nonempty finite interval-by-dimension matrix")
    grid = _parameter(parameter, len(q) + 1)
    origin = np.zeros(q.shape[1]) if origin is None else np.asarray(origin, dtype=float)
    if origin.shape != (q.shape[1],) or not np.isfinite(origin).all():
        raise ValueError("Origin must be a finite vector matching the curve dimensions")
    increments = q * np.linalg.norm(q, axis=1)[:, None] * np.diff(grid)[:, None]
    return np.vstack((origin, origin + np.cumsum(increments, axis=0)))


def _arc_parameter(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Remove consecutive repeated positions so normalized arc length is monotone.
    keep = np.r_[True, np.any(np.diff(curve, axis=0) != 0, axis=1)]
    points = curve[keep]
    if len(points) == 1:
        return np.repeat(points, 2, axis=0), np.array([0.0, 1.0])
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    distance /= distance[-1]
    # Coordinates closer than floating-point arc-length resolution can share a knot.
    _, indices = np.unique(distance, return_index=True)
    indices[-1] = len(points) - 1
    return points[indices], distance[indices]


def interpolate_paths(first: np.ndarray, second: np.ndarray, *,
                       times: np.ndarray | None = None, samples: int | None = None,
                       center: bool = False, method: str = "paper") -> PathMorph:
    """Interpolate curves using the earlier implementation or paper §3.4.

    method='elasticpath2path' translates visualMorphingNeuron.m, including its
    vertex derivatives, resampling, and linearly interpolated reconstruction
    errors. With samples=None, it upsamples both paths to the larger input size.
    The corrections recover endpoint inputs, but intermediate roots can drift.

    method='paper' uses normalized arc length. With samples=None,
    the common grid is the union of both inputs' knots, preserving every bend
    of both piecewise-linear endpoints. An integer samples>=2 uses that many
    uniformly spaced points, approximating the input polylines. This sampling
    convention is an implementation choice, not an estimated elastic time-warp.

    Origins interpolate linearly unless center=True, which anchors both at zero.
    Shape interpolation itself is linear in SRVF space, not coordinate space.
    Length is retained; SRVFs are not normalized to the unit sphere.
    """
    first, second = _curve(first), _curve(second)
    if first.shape[1] != second.shape[1]:
        raise ValueError("Both curves must have the same dimension")
    times = _times(times)
    if method == "elasticpath2path":
        return _interpolate_elasticpath2path(first, second, times, samples, center)
    if method != "paper":
        raise ValueError("method must be 'elasticpath2path' or 'paper'")
    a, ta = _arc_parameter(first)
    b, tb = _arc_parameter(second)
    if samples is None:
        grid = np.union1d(ta, tb)
    else:
        if not isinstance(samples, (int, np.integer)) or samples < 2:
            raise ValueError("samples must be an integer >= 2, or None")
        grid = np.linspace(0, 1, samples)
    a = np.column_stack([np.interp(grid, ta, a[:, d]) for d in range(a.shape[1])])
    b = np.column_stack([np.interp(grid, tb, b[:, d]) for d in range(b.shape[1])])
    qa, qb = curve_to_srvf(a, grid), curve_to_srvf(b, grid)
    q = (1 - times[:, None, None]) * qa + times[:, None, None] * qb
    increments = q * np.linalg.norm(q, axis=2)[:, :, None] * np.diff(grid)[None, :, None]
    origin = np.zeros((len(times), 1, first.shape[1]))
    if not center:
        origin = ((1 - times[:, None, None]) * first[0]
                  + times[:, None, None] * second[0])
    frames = np.concatenate((np.zeros_like(origin), np.cumsum(increments, axis=1)), axis=1)
    frames += origin
    source_curve = a - a[0] if center else a.copy()
    target_curve = b - b[0] if center else b.copy()
    return PathMorph(grid, times, frames, "paper", source_curve, target_curve)


def morph_neurons(first: Morphology | np.ndarray, second: Morphology | np.ndarray, *,
                   times: np.ndarray | None = None, samples: int | None = 128,
                   mode: str = "inter", axis_order: str = "original",
                   method: str = "paper",
                   arbor: str = "dendrite",
                   matlab_compatible: bool = False) -> NeuronMorph:
    """Match selected neurite paths and morph each pair independently.

    Input somata are centered at zero. axis_order='range' permutes coordinates
    in descending range order per neuron, as in Fig. 6; 'original' retains axes.
    No rotation optimization, scale normalization, or shared-branch stitching
    is performed. Coincident paths at either endpoint represent merging or
    splitting, rather than the appearance/deletion of graph edges.
    arbor='dendrite' uses tips of SWC type 3/4, 'axon' uses type 2, and 'both'
    matches the two compartments separately. A compartment present in only one
    input raises an error rather than matching axons to dendrites or omitting it.
    """
    first = first if isinstance(first, Morphology) else morphology_from_array(first)
    second = second if isinstance(second, Morphology) else morphology_from_array(second)
    if axis_order not in {"range", "original"}:
        raise ValueError("axis_order must be 'range' or 'original'")
    times = _times(times)
    fa, fb = extract_features(first, arbor=arbor), extract_features(second, arbor=arbor)
    kinds = []
    for tree, features in ((first, fa), (second, fb)):
        types = {int(row[0]): int(row[1]) for row in tree.nodes}
        kinds.append(np.array(["axon" if types[int(path.node_ids[-1])] == 2 else "dendrite"
                               for path in features.paths], dtype=str))
    groups = ("dendrite", "axon") if arbor == "both" else (arbor,)
    all_pairs, all_added, pair_kinds = [], [], []
    for kind in groups:
        ia, ib = np.flatnonzero(kinds[0] == kind), np.flatnonzero(kinds[1] == kind)
        if not len(ia) and not len(ib) and arbor == "both":
            continue
        if not len(ia) or not len(ib):
            description = "axonal paths (SWC type 2)" if kind == "axon" else "dendritic paths (SWC types 3/4)"
            raise ValueError(f"Both morphologies need {description}; "
                             f"found source={len(ia)}, target={len(ib)}")
        matching = compute_distance([fa.paths[i] for i in ia], [fb.paths[j] for j in ib],
                                    mode=mode, matlab_compatible=matlab_compatible)
        pairs = matching.pairs.copy()
        # Repair any paths orphaned by outlier reassignment within this compartment.
        additional = []
        for i in np.setdiff1d(np.arange(len(ia)), pairs[:, 0]):
            additional.append((int(i), int(np.argmin(matching.cost_matrix[i]))))
        for j in np.setdiff1d(np.arange(len(ib)), pairs[:, 1]):
            additional.append((int(np.argmin(matching.cost_matrix[:, j])), int(j)))
        if additional:
            added = np.asarray(additional, dtype=np.int64)
            pairs = np.vstack((pairs, added))
            all_added.extend(zip(ia[added[:, 0]], ib[added[:, 1]]))
        all_pairs.extend(zip(ia[pairs[:, 0]], ib[pairs[:, 1]]))
        pair_kinds.extend([kind] * len(pairs))
    if not all_pairs:
        raise ValueError("Both morphologies need selected neurite paths (SWC types 2/3/4)")
    pairs = np.asarray(all_pairs, dtype=np.int64)
    added_pairs = np.asarray(all_added, dtype=np.int64).reshape(-1, 2)

    roots = [tree.nodes[tree.root, 2:5].copy() for tree in (first, second)]
    axes, curves = [], []
    for tree, features, root in zip((first, second), (fa, fb), roots):
        coords = tree.nodes[:, 2:5] - root
        order = (np.argsort(-np.ptp(coords, axis=0), kind="stable")
                 if axis_order == "range" else np.arange(3))
        axes.append(order)
        lookup = {int(node_id): row for row, node_id in enumerate(tree.nodes[:, 0])}
        curves.append([coords[[lookup[int(node_id)] for node_id in path.node_ids]][:, order]
                       for path in features.paths])
    paths = tuple(interpolate_paths(curves[0][i], curves[1][j], times=times,
                                     samples=samples, center=True, method=method) for i, j in pairs)
    return NeuronMorph(times, pairs, paths, added_pairs,
                       tuple(p.node_ids.copy() for p in fa.paths),
                       tuple(p.node_ids.copy() for p in fb.paths),
                       roots[0], roots[1], axes[0], axes[1], method, arbor, tuple(pair_kinds))


def _endpoint_curves(morph: NeuronMorph, side: str, fraction: float,
                     column: int) -> list[np.ndarray]:
    curves, seen = [], set()
    for pair, path in zip(morph.pairs, morph.paths):
        index = int(pair[column])
        if index in seen:
            continue
        seen.add(index)
        curve = getattr(path, side)
        if curve is None:
            # Support previously constructed PathMorph objects that include endpoints.
            frames = np.flatnonzero(path.times == fraction)
            if not len(frames):
                raise ValueError("Endpoint references are missing; regenerate the morph "
                                 "with morph_neurons or include times 0 and 1")
            curve = path.frames[frames[0]]
        curves.append(curve)
    return curves


def write_morph_html(morph: NeuronMorph, destination: str | Path) -> Path:
    """Write synchronized source, intermediate, and target views in standalone HTML."""
    if not morph.paths:
        raise ValueError("Morph must contain paths")

    payload = json.dumps({"times": morph.times.tolist(),
                          "arbor": morph.arbor,
                          "path_kinds": morph.path_kinds,
                          "pairs": morph.pairs.tolist(),
                          "source": [c.tolist() for c in _endpoint_curves(morph, "source_curve", 0, 0)],
                          "target": [c.tolist() for c in _endpoint_curves(morph, "target_curve", 1, 1)],
                          "paths": [path.frames.tolist() for path in morph.paths]},
                         separators=(",", ":"), allow_nan=False)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    template = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>NeuroPath2Path — path morphing</title>
<style>
body{margin:0;background:#f4f6f8;color:#172b3a;font:16px system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:20px}h1{font-size:24px;margin:0 0 8px}
.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.view{background:white;border-radius:8px;overflow:hidden}
h2{font-size:16px;margin:0;padding:12px 16px;border-bottom:1px solid #edf0f3}
#source-title{color:#2563eb}#target-title{color:#c2410c}
canvas{display:block;width:100%;height:48vh;min-height:240px;touch-action:none}
.controls{display:flex;gap:14px;align-items:center;padding:16px 0}input[type=range]{flex:1;min-width:60px}
.overlay{display:flex;align-items:center;gap:6px;margin-bottom:12px;font-size:14px}
@media(max-width:700px){.views{grid-template-columns:1fr}canvas{height:32vh}.controls{flex-wrap:wrap}}
button{padding:8px 18px;font:inherit}p{line-height:1.5}small{color:#475867}
</style><main><h1>NeuroPath2Path — path morphing</h1>
<p>Compare the current morph with both cells. All views share the same scale.
Drag any view to rotate all three; scroll to zoom.</p>
<p id="arbor-label"></p>
<div class="views">
<section class="view"><h2 id="source-title">Source · α = 0</h2>
<canvas id="source" aria-label="Source cell at alpha zero"></canvas></section>
<section class="view"><h2 id="current-title">Current morph</h2>
<canvas id="current" aria-label="Intermediate morphed cell"></canvas></section>
<section class="view"><h2 id="target-title">Target · α = 1</h2>
<canvas id="target" aria-label="Target cell at alpha one"></canvas></section>
</div>
<div class="controls"><button id="play">Play</button>
<input id="frame" aria-label="Morph fraction" type="range" min="0" value="0" step="1">
<output id="label"></output></div>
<label class="overlay"><input id="overlay" type="checkbox">Overlay source (blue) and target (orange) on the current morph</label>
<small>Colored lines track matched paths. Paths evolve independently; this view
    does not define a branching SWC topology. Input somata are centered at the
origin; intermediate roots may drift with the original reconstruction method.</small></main>
<script>
const data=__MORPH_DATA__;
document.querySelector('#arbor-label').textContent={dendrite:'Showing dendritic paths',axon:'Showing axonal paths',both:'Showing dendritic and axonal paths'}[data.arbor];
const canvases=['source','current','target'].map(id=>document.getElementById(id));
const slider=document.querySelector('#frame'),label=document.querySelector('#label');
const overlay=document.querySelector('#overlay');
const play=document.querySelector('#play');slider.max=data.times.length-1;
let yaw=0.4,pitch=-0.3,zoom=1,timer=null,drag=null;
let low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
for(const path of data.paths)for(const frame of path)for(const p of frame)
 for(let d=0;d<3;d++){low[d]=Math.min(low[d],p[d]);high[d]=Math.max(high[d],p[d]);}
for(const curve of [...data.source,...data.target])for(const p of curve)
 for(let d=0;d<3;d++){low[d]=Math.min(low[d],p[d]);high[d]=Math.max(high[d],p[d]);}
const center=low.map((v,d)=>(v+high[d])/2);
const radius=Math.max(1e-12,Math.hypot(...high.map((v,d)=>v-low[d]))/2);
function project(p,canvas){
 const [x,y,z]=p.map((v,d)=>v-center[d]);
 const a=Math.cos(yaw)*x+Math.sin(yaw)*z;
 const b=-Math.sin(yaw)*x+Math.cos(yaw)*z;
 const c=Math.cos(pitch)*y-Math.sin(pitch)*b;
 const depth=Math.sin(pitch)*y+Math.cos(pitch)*b;
 const perspective=4*radius/(4*radius-depth);
 const scale=Math.min(canvas.width,canvas.height)*0.4*zoom/radius;
 return [canvas.width/2+a*scale*perspective,canvas.height/2-c*scale*perspective];
}
function draw(){
 const ratio=window.devicePixelRatio||1;
 const frame=Number(slider.value);
 canvases.forEach((canvas,panel)=>{
  canvas.width=Math.round(canvas.clientWidth*ratio);canvas.height=Math.round(canvas.clientHeight*ratio);
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  function stroke(curves,color,opacity=1){
   ctx.globalAlpha=opacity;ctx.lineWidth=(opacity<1?1.2:1.6)*ratio;
   curves.forEach((curve,index)=>{
    ctx.strokeStyle=color||`hsl(${index*137.508%360} 65% 40%)`;ctx.beginPath();
    curve.forEach((p,i)=>{const [x,y]=project(p,canvas);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});
    ctx.stroke();
   });ctx.globalAlpha=1;
  }
  if(panel===0)stroke(data.source,'#2563eb');
  else if(panel===2)stroke(data.target,'#c2410c');
  else{
   if(overlay.checked){stroke(data.source,'#2563eb',0.2);stroke(data.target,'#c2410c',0.2);}
   stroke(data.paths.map(path=>path[frame]),null);
  }
  const [x,y]=project([0,0,0],canvas);ctx.fillStyle='#172b3a';ctx.beginPath();ctx.arc(x,y,3*ratio,0,2*Math.PI);ctx.fill();
 });
 document.querySelector('#current-title').textContent=`Current morph · α = ${data.times[frame].toFixed(3)}`;
 label.textContent=`α = ${data.times[frame].toFixed(3)} · ${data.paths.length} matched pairs`;
}
slider.oninput=draw;
overlay.onchange=draw;
play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='Play';}
 else{play.textContent='Pause';timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%data.times.length;draw();},120);}};
canvases.forEach(canvas=>{
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*0.01;pitch+=(e.clientY-drag[1])*0.01;drag=[e.clientX,e.clientY];draw();}};
canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(0.1,Math.min(10,zoom*Math.exp(-e.deltaY*0.001)));draw();};
});
window.onresize=draw;draw();
</script></html>"""
    destination.write_text(template.replace("__MORPH_DATA__", payload), encoding="utf-8")
    return destination
