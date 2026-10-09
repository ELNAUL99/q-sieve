"""Embed freshly generated model data in the offline demo. Run after model.py."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent
source = (ROOT / 'frontend-template.html').read_text(encoding='utf-8')
data = json.loads((ROOT / 'model-data.json').read_text(encoding='utf-8'))
encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
(ROOT / 'index.html').write_text(source.replace('__MODEL_DATA__', encoded), encoding='utf-8')
print('Created standalone index.html')
