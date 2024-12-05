# Load the raster data
with rasterio.open('path to raster file.tif') as src:
    # Read the raster bands
    data = src.read()

    # # Print the current order of bands
    # print("Current band order:")
    # for idx, desc in enumerate(src.descriptions):
    #     print(f"Band {idx + 1}: {desc}")

    # Define the desired band order
    desired_order = ['BLUE_Aug', 'BLUE_Feb', 'BLUE_May', 'BLUE_Nov', 'GREEN_Aug',
                     'GREEN_Feb', 'GREEN_May', 'GREEN_Nov', 'NDVI_Aug', 'NDVI_Feb',
                     'NDVI_May', 'NDVI_Nov', 'SWIR2_Aug', 'SWIR2_Feb', 'SWIR2_May',
                     'SWIR2_Nov', 'TEMP_Aug', 'TEMP_Feb', 'TEMP_May', 'TEMP_Nov', 
                     'Slope', 'Aspect', 'DEM']

    # Create an empty array to store the rearranged bands
    rearranged_data = np.zeros_like(data)

    # Map current bands to the desired order
    band_map = {desc: i for i, desc in enumerate(src.descriptions)}

    for i, band_name in enumerate(desired_order):
        if band_name in band_map:
            rearranged_data[i] = data[band_map[band_name]]
        else:
            raise ValueError(f"Band {band_name} not found in the raster file.")
            
    # Reshape the data to match the input shape expected by the model
    reshaped_data = rearranged_data.reshape((rearranged_data.shape[0], -1)).T

# Create a mask for NaN values in the original data
nan_mask = np.isnan(reshaped_data).any(axis=1)

# Remove rows with NaN values for prediction
reshaped_data_no_nan = reshaped_data[~nan_mask]

# Predict tree cover using the trained RFR model on cleaned data
tree_cover_predictions = rf.predict(reshaped_data_no_nan)
residual_prediction = rf_residual_Train.predict(reshaped_data_no_nan)
biascorrected_treecover_prediction = tree_cover_predictions + residual_prediction

#Capping the tree cover values between 0 and 1. 
biascorrected_treecover_prediction = np.maximum(biascorrected_treecover_prediction, 0)
biascorrected_treecover_prediction = np.minimum(biascorrected_treecover_prediction, 1)

# Prepare to place predictions back into the original shape
# Create an array with the same shape as the original data but filled with NaNs
tree_cover_raster = np.full((src.height * src.width), np.nan)

# Place the predictions back into their original positions
tree_cover_raster[~nan_mask] = biascorrected_treecover_prediction

# Reshape the array to match the original raster dimensions
tree_cover_loss = tree_cover_raster.reshape((src.height, src.width))

# Sum all the pixel values
total_tree_cover = np.nansum(tree_cover_loss)

# Print the total value
print(f"Total tree cover value: {total_tree_cover}")


# Display the tree cover raster
plt.figure(figsize=(10, 5))
plt.imshow(tree_cover_loss, cmap='viridis',vmin=0, vmax=1)
plt.colorbar()
plt.title('Bias-Corrected Fractional Tree Cover - 2006')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.savefig('TreeCover_LOSS.png', dpi=300, bbox_inches='tight')
plt.show()
