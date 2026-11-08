"""Check animation endpoints against recorded states/models, allowing GIF quantization."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from coph_stagewise.evaluation import load
from coph_stagewise.visualize import frame, plt
root=Path(sys.argv[1]);manifest=load(root/'manifest.json');checked=0
for scene in dict.fromkeys(s['name'] for s in manifest['scenarios']):
    seed=manifest['animation_seed'];arm=manifest['animation_arm']
    d=load(root/'episodes'/f'{scene}_seed{seed}_{arm}.json.gz')
    with Image.open(root/'figures'/f'{scene}_seed{seed}.gif') as gif:
        for saved,physical in [(0,0),(gif.n_frames-1,len(d['trajectory'])-1)]:
            gif.seek(saved);actual=np.asarray(gif.convert('RGB')).astype(float)
            fig=frame(d,physical);fig.canvas.draw()
            expected=np.asarray(fig.canvas.buffer_rgba())[...,:3].astype(float);plt.close(fig)
            error=np.abs(actual-expected).mean()
            assert error<2.,(scene,saved,error)
            checked+=1
print(f'Validated {checked} animation endpoints against logged states and model snapshots.')
