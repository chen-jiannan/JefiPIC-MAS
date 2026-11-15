"""Generate unsplit, native-Esirkepov WarpX inputs for the common sphere case."""
from pathlib import Path
import json,hashlib
import numpy as np
from warpx_inputs import write_warpx
HERE=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 case=HERE/'examples/sphere/input';cfg=json.loads((case/'case.json').read_text())
 assert hashlib.sha256((case/'particles.npz').read_bytes()).hexdigest()==cfg['particle_file_sha256']
 xyz=np.load(case/'particles.npz')['positions_m'];assert len(xyz)==42168
 for metal in [True,False]:
  folder=HERE/'outputs/warpx_inputs'/('metal' if metal else 'no_metal');folder.mkdir(parents=True,exist_ok=True)
  path=folder/'inputs';write_warpx(path,cfg,xyz,96,4,metal)
  assert 'algo.current_deposition = esirkepov' in path.read_text()
  print(path)
