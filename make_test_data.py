import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

# UV-Vis: two overlapping Gaussian absorbance peaks, values 0-2
wl = np.arange(220, 800, 1.0)
abs_y = (1.8 * np.exp(-((wl - 260) ** 2) / (2 * 15 ** 2))
         + 0.6 * np.exp(-((wl - 450) ** 2) / (2 * 30 ** 2))
         + rng.normal(0, 0.01, wl.size))
with open("uvvis_sample.txt", "w") as f:
    f.write("Instrument: Cary 60 UV-Vis\n")
    f.write("Operator: T. Churchill\n")
    f.write("Date: 2026-06-30\n")
    f.write("\n")
    f.write("Wavelength (nm)\tAbs\n")
    for a, b in zip(wl, abs_y):
        f.write(f"{a:.1f}\t{b:.4f}\n")

# Emission spectrum: single broad peak, large counts
wl2 = np.arange(400, 700, 1.0)
em_y = 50000 * np.exp(-((wl2 - 520) ** 2) / (2 * 25 ** 2)) + rng.normal(0, 200, wl2.size)
em_y = np.clip(em_y, 0, None)
df_em = pd.DataFrame({"Wavelength (nm)": wl2, "Emission Intensity (CPS)": em_y})
df_em.to_csv("emission_sample.csv", index=False)

# Lifetime decay: ~0 baseline for 0-40 ns (instrument dead time), then a
# genuine bi-exponential decay, in an xlsx with a metadata preamble
t = np.arange(0, 200, 0.5)  # ns
baseline_counts = rng.poisson(4, t.size).astype(float)
decay_mask = t >= 40
decay_t = t[decay_mask] - 40
decay_counts = 7000 * np.exp(-decay_t / 6.0) + 3000 * np.exp(-decay_t / 25.0)
counts = baseline_counts.copy()
counts[decay_mask] += decay_counts
counts += rng.poisson(3, t.size)
meta = pd.DataFrame({"A": ["FluoTime 300 TCSPC", "Channel width: 0.5 ns", ""],
                      "B": ["", "", ""]})
data = pd.DataFrame({"A": t, "B": counts})
header = pd.DataFrame({"A": ["Time (ns)"], "B": ["Counts"]})
combined = pd.concat([meta, header, data], ignore_index=True)
combined.to_excel("lifetime_sample.xlsx", index=False, header=False)

print("wrote uvvis_sample.txt, emission_sample.csv, lifetime_sample.xlsx")
