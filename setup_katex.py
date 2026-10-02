"""Run once:  python setup_katex.py
Downloads KaTeX from the npm registry and creates static/katex (needs internet this one time only)."""
import io, tarfile, urllib.request
from pathlib import Path

URL = "https://registry.npmjs.org/katex/-/katex-0.16.11.tgz"
OUT = Path(__file__).parent / "static" / "katex"
(OUT / "fonts").mkdir(parents=True, exist_ok=True)

print("Downloading KaTeX...")
data = urllib.request.urlopen(URL, timeout=60).read()
n = 0
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
    for m in tar.getmembers():
        name = m.name  # package/dist/...
        if name in ("package/dist/katex.min.js", "package/dist/katex.min.css"):
            dest = OUT / Path(name).name
        elif name == "package/dist/contrib/auto-render.min.js":
            dest = OUT / "auto-render.min.js"
        elif name.startswith("package/dist/fonts/") and name.endswith(".woff2"):
            dest = OUT / "fonts" / Path(name).name
        else:
            continue
        dest.write_bytes(tar.extractfile(m).read())
        n += 1
print(f"Done: {n} files saved in {OUT}")
