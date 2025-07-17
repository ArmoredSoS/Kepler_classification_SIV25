#Test to see the average length of time series in the Kepler dataset for trimming or padding

from function import *

curves = download_curves(get_kepids('all'), 10)
normalized = normalize_curves(curves)
lengths = [len(curve.flux) for curve in normalized]

print("Plotting")

plt.hist(lengths, bins=50)
plt.xlabel("Time series length")
plt.ylabel("Frequency")
plt.title("Distribution of Kepler light curve lengths")
plt.show()

#Final: ~65,000