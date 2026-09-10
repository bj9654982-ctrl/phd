"""Download exact official Restormer files; do not install or execute vendor code."""
import hashlib
import base64
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "68dc6ac472db26f16361150cb7a96a1bc87da93f"
FILES = ("LICENSE.md", "README.md", "basicsr/models/archs/restormer_arch.py",
         "Motion_Deblurring/test.py", "Motion_Deblurring/README.md",
         "Motion_Deblurring/Options/Deblurring_Restormer.yml",
         "Motion_Deblurring/generate_patches_gopro.py",
         "Motion_Deblurring/evaluate_gopro_hide.m")


def main():
    output = ROOT / "third_party/restormer_official_68dc6ac_v1"
    if output.exists():
        raise FileExistsError("Refusing to overwrite vendor download")
    output.mkdir(parents=True)
    records = []
    requests = [(name, f"https://api.github.com/repos/swz30/Restormer/contents/{name}?ref={COMMIT}", None)
                for name in FILES]
    requests += [("motion_deblurring.pth",
                  "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth",
                  104700429)]
    for relative, url, expected in requests:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {relative}", flush=True)
        digest, count = hashlib.sha256(), 0
        request = urllib.request.Request(url, headers={"User-Agent": "RestorationResearch"})
        with urllib.request.urlopen(request, timeout=45) as response:
            final_url = response.url
            if expected is None:
                payload = json.load(response)
                chunk = base64.b64decode(payload["content"])
                with target.open("xb") as handle:
                    handle.write(chunk)
                digest.update(chunk)
                count = len(chunk)
            else:
                with target.open("xb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
        if expected is not None and count != expected:
            raise ValueError(f"Unexpected byte count: {relative}, {count}")
        records.append({"path": relative, "official_url": url, "final_url": final_url,
                        "bytes": count, "sha256": digest.hexdigest()})
        (output / "download_provenance.json").write_text(json.dumps({
            "repository": "https://github.com/swz30/Restormer", "commit": COMMIT,
            "license": "MIT; retain LICENSE.md", "files": records,
            "author_published_checkpoint_digest_available": False,
            "local_hash_is_provenance_not_independent_author_signature": True,
            "complete": len(records) == len(requests)}, indent=2), encoding="utf-8")
    print("Official source and 104700429-byte checkpoint downloaded; not executed.", flush=True)


if __name__ == "__main__":
    main()
