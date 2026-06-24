import json
import urllib.request
req = urllib.request.Request(
    'http://127.0.0.1:11434/api/generate',
    data=json.dumps({'model':'llama3.2:latest','prompt':'hello','stream':False}).encode(),
    headers={'Content-Type':'application/json'},
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        print(r.read().decode())
except Exception as e:
    import traceback
    traceback.print_exc()
