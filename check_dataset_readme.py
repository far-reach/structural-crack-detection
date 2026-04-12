from pathlib import Path

# Look for any readme or description files
base = Path("data/extra")
for f in base.rglob("*"):
    if f.suffix in [".txt", ".md", ".json", ".csv"] and f.is_file():
        print(f)
        if f.stat().st_size < 5000:
            print(f.read_text(errors="ignore"))
            print("---")