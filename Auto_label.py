import pandas as pd
import glob
import os
from astropy.io import fits

# Load KOI data from a folder
def load_koi_labels(koi_folder):
    koi_data = []
    for file in glob.glob(f"{koi_folder}/*.fits"):
        with fits.open(file) as hdul:
            kepid = hdul[0].header.get('KEPLERID', None)  # Extract Kepler ID from header
            if kepid:
                koi_data.append({'kepid': kepid, 'label': 'POSITIVE'})  # Default label (adjust as needed)
    return pd.DataFrame(koi_data).set_index('kepid')

# Label Kepler light curves from a folder and save results
def label_kepler_data(kepler_folder, koi_df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    labeled_data = []
    for file in glob.glob(f"{kepler_folder}/*.fits"):
        with fits.open(file) as hdul:
            kepid = hdul[0].header.get('KEPLERID', None)
            label = koi_df.loc[kepid, 'label'] if kepid in koi_df.index else 'NEGATIVE'
            labeled_data.append({'kepler_id': kepid, 'file': file, 'label': label})
            
            # Save labeled file
            output_path = os.path.join(output_folder, os.path.basename(file))
            hdul.writeto(output_path, overwrite=True)
    
    labeled_df = pd.DataFrame(labeled_data)
    labeled_df.to_csv(os.path.join(output_folder, "labeled_data.csv"), index=False)
    return labeled_df

# Example usage
koi_folder = "Kepler_dataset\KOI"
kepler_folder = "Kepler_dataset\Full_Dataset"
output_folder = "Kepler_dataset\Label_dataset"
koi_df = load_koi_labels(koi_folder)
labeled_df = label_kepler_data(kepler_folder, koi_df, output_folder)
print(labeled_df)
