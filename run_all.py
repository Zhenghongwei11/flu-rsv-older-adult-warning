"""Reproduce numerical analyses and all study figures and tables."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--refit',action='store_true',help='Recompute all forecasts and paired intervals from frozen weekly inputs.')
a=p.parse_args()
scripts=['analysis.py','matched_training.py','seasonal_sensitivity.py'] if a.refit else []
scripts+=['figures.py','tables.py','workbook.py']
for script in scripts:
 subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
