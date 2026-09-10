"""Download one author-linked public checkpoint with size/time bounds."""
import hashlib,json,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'third_party/restormer_dpdd_public_v0';out.mkdir(parents=True,exist_ok=True)
    path=out/'single_image_defocus_deblurring.pth'
    url='https://drive.usercontent.google.com/download?id=10v8BH3Gktl34TYzPy0x-pAKoRSYKnNZp&export=download&confirm=t'
    started=time.monotonic();size=0;digest=hashlib.sha256()
    with requests.get(url,stream=True,timeout=(20,30)) as response:
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type',''):raise RuntimeError('HTML instead of checkpoint; no file written')
        with path.open('xb') as f:
            for chunk in response.iter_content(1024*1024):
                size+=len(chunk)
                if size>150*1024**2 or time.monotonic()-started>300:raise RuntimeError('Download budget exceeded; partial file retained')
                f.write(chunk);digest.update(chunk)
    if size!=104700429:raise RuntimeError(f'Unexpected length {size}; retained for inspection, do not load')
    result={'source_folder':'https://drive.google.com/drive/folders/1bRBG8DG_72AGA6-eRePvChlT5ZO4cwJ4',
        'url':url,'bytes':size,'sha256':digest.hexdigest(),'seconds':time.monotonic()-started,
        'downloaded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hash_is_locally_computed_not_author_signature':True}
    (out/'provenance.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
