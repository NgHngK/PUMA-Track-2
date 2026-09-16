from pathlib import Path
import urllib.request,json
p=Path(__file__).parent
items=json.loads((p/'download_refs.json').read_text())
for name,url in items.items():
    urllib.request.urlretrieve(url,p/'ingestion'/name)
    print(name,(p/'ingestion'/name).stat().st_size)
