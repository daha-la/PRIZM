import nbformat

path = "./ZP_analysis.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if "outputs" in cell:
        cell["outputs"] = []
    if "execution_count" in cell:
        cell["execution_count"] = None

with open(path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
