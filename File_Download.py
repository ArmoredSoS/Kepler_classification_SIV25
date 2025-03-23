import lightkurve as lk
import os
import warnings
import random

# Suppress warnings from astropy and lightkurve
warnings.simplefilter("ignore")

# Create a directory to store light curves
save_dir = "kepler_lightcurves"
os.makedirs(save_dir, exist_ok=True)

# Define number of light curves per star and number of stars
lightcurves_per_star = 100
num_stars = 20  

# List of candidate stars
targets = [
    "Kepler-10", "Kepler-11", "Kepler-12", "Kepler-22", "Kepler-90", 
    "Kepler-186", "Kepler-452", "Kepler-62", "Kepler-62f", "Kepler-452b", 
    "Kepler-20", "Kepler-68", "Kepler-16", "Kepler-62", "Kepler-37", 
    "Kepler-19", "Kepler-39", "Kepler-55", "Kepler-134", "Kepler-223"
]

# Shuffle targets for variety
random.shuffle(targets)

# Download and save light curves
for star in targets[:num_stars]:
    search_result = lk.search_lightcurve(star, mission="Kepler")

    if not search_result:
        print(f"No light curve found for {star}")
        continue

    lightcurves = []
    for sector in range(1, 21):  # Search multiple sectors
        print(f"Searching sector {sector} for {star}")
        sector_result = lk.search_lightcurve(star, mission="Kepler", sector=sector)
        
        if sector_result:
            lightcurves += sector_result.download_all(quality_bitmask="default")

        if len(lightcurves) >= lightcurves_per_star:
            break  # Stop when enough light curves are collected

    # Save light curves after NaN removal
    for i, lc in enumerate(lightcurves[:lightcurves_per_star]):
        try:
            lc = lc.remove_nans()  # Remove NaNs
            filename = os.path.join(save_dir, f"{star.replace(' ', '_')}_lc_{i+1}.fits")
            lc.to_fits(path=filename, overwrite=True)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Skipping {star} light curve {i+1} due to error: {e}")

