import subprocess

PDF_PATH = "/home/rukost/deepcock/maslova.pdf"
MARKUP_TYPE = "0"
PAGES = "1-50"

for workers in [1, 2, 4, 8, 16, 32, 64]:
    print(f"\n--- Benchmark with {workers} workers ---")
    result = subprocess.run([
        "python", "main-bench.py",
        PDF_PATH,
        MARKUP_TYPE,
        "-w", str(workers),
        "-p", PAGES
    ], capture_output=True, text=True)
    print(result.stdout)
