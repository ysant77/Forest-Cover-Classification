import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

color_map = {
    0: np.array([255, 255, 255], dtype=np.float32) / 255,  # other, cloud, shadow, snow/ice; white
    1: np.array([0, 153, 0], dtype=np.float32) / 255,  # forest; dark green
    2: np.array([198, 176, 68], dtype=np.float32) / 255,  # shrubland, savanna, grassland, cropland; mustard
    3: np.array([39, 255, 135], dtype=np.float32) / 255,  # wetlands; lime green
    4: np.array([165, 165, 165], dtype=np.float32) / 255,  # urban; grey
    5: np.array([249, 255, 164], dtype=np.float32) / 255,  # barren; light yellow
    6: np.array([28, 13, 255], dtype=np.float32) / 255,  # water; blue
    7: np.array([255, 0, 0], dtype=np.float32) / 255  # unknown; red
}
labels = [
    "Shrubland", "Savanna", "Grassland",
    "Wetland", "Corpland", "Urban",
    "Snow and Ice", "Barren"
]
# Create the colormap
color_list = [color_map[key] for key in color_map.keys()]
new_cmap = ListedColormap(color_list)


def quantify_changes(mask_starting, mask_ending):
    """Quantify the changes in the occurrence of each class between two masks"""
    # Get the masks
    

    changes = {}
    area_changes = {}  # Dictionary to store changes in square kilometers
    percent_changes = {}  # New dictionary to store percentage changes
    unique1, counts1 = np.unique(mask_starting, return_counts=True)
    unique2, counts2 = np.unique(mask_ending, return_counts=True)

    class_counts1 = dict(zip(unique1, counts1))
    class_counts2 = dict(zip(unique2, counts2))

    # Unique classes in classification
    unique_classes = np.unique(mask_starting)
    # Calculate the changes
    for class_id in unique_classes:
        count1 = class_counts1.get(class_id, 0)
        count2 = class_counts2.get(class_id, 0)
        change = count2 - count1
        changes[class_id] = change

        # Calculate the change in area
        area_change = change * 100 / 1e6  # Convert from square meters (100 per pixel) to square kilometers
        area_changes[class_id] = area_change

        # Calculate the percentage change
        if count1 > 0:  # To avoid division by zero
            percent_change = (change / count1) * 100  # Percentage change formula
        else:
            percent_change = 0 if change == 0 else float('inf')  # Set to infinity if count1 is 0 but there's a change
        percent_changes[class_id] = percent_change

   
    changes_df = pd.DataFrame(list(changes.items()), columns=['Class', 'change_in_pixels'])
    area_changes_df = pd.DataFrame(list(area_changes.items()), columns=['Class', 'change_in_area'])
    percent_changes_df = pd.DataFrame(list(percent_changes.items()), columns=['Class', 'change_in_percentage'])
    

    # Merge the data on Class
    combined_df = pd.merge(changes_df, area_changes_df, on='Class')


    combined_df = pd.merge(combined_df, percent_changes_df, on='Class')  # Merging with percent_changes_df

    combined_df['Class'] = combined_df['Class'].apply(lambda x: labels[x - 2])  # Replace class ID with class name
    # Dump results
    combined_df.to_csv('files/changes.csv', index=False)

    return combined_df.to_dict(orient="records")
