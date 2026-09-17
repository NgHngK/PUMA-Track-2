import urllib.request,urllib.parse,re,pathlib,hashlib,json,torch
w=pathlib.Path(__file__).parent
h=(w/'stage1_download_response').read_text();action=re.search('action="([^"]+)"',h).group(1);params=dict(re.findall('name="([^"]+)" value="([^"]*)"',h));p=w/'stage1_final_fold0.pt'
if not p.exists():urllib.request.urlretrieve(action+'?'+urllib.parse.urlencode(params),p)
print('sha',hashlib.sha256(p.read_bytes()).hexdigest());s=torch.load(p,weights_only=True,map_location='cpu');print(s.keys());print(s.get('extra'));(w/'stage1_checkpoint_metadata.json').write_text(json.dumps(s.get('extra'),indent=2))
