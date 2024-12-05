// Mosaic the LiDAR images in the collection into a single image
var mergedImage = merged_LiDAR;

// Define the spatial resolution for aggregation (30m for Landsat)
var targetResolution = 30;

// Step 1: Count the number of valid pixels in each 30m pixel
var countPixels = mergedImage.reduceResolution({
  reducer: ee.Reducer.count(),
  bestEffort: true,
  maxPixels: 900 // 30m x 30m = 900 pixels of 1m x 1m
}).reproject({
  crs: mergedImage.projection(), // Use the original projection
  scale: targetResolution
});

// Create a mask for pixels with exactly 900 valid 1m pixels
var mask900 = countPixels.eq(900);

// Step 2: Apply a mask for values greater than or equal to 5 in the LiDAR image
var mask5 = mergedImage.gte(5);
var maskedImage = mergedImage.updateMask(mask5);

// Step 3: Count the number of valid pixels in each 30m pixel for the masked image
var countPixels5 = maskedImage.reduceResolution({
  reducer: ee.Reducer.count(),
  bestEffort: true,
  maxPixels: 900 // 30m x 30m = 900 pixels of 1m x 1m
}).reproject({
  crs: mergedImage.projection(),
  scale: targetResolution
});

// Step 4: Adjust the count by dividing by 9 and rounding to the nearest integer
var countPixels5Adjusted = countPixels5.divide(9).round().toInt16();

// Step 5: Apply the mask for 900 valid pixels to the adjusted count
var countPixels5Masked = countPixels5Adjusted.updateMask(mask900);

// Step 6: Define export parameters
var exportParams = {
  image: countPixels5Masked,
  description: 'LiDAR_5mMask_Image',
  assetId: 'users/manan_sarupria/LiDAR_5mMask_Image', // Replace with your username and desired asset name
  scale: 30, // Adjust the scale if necessary
  region: countPixels5Masked.geometry(), // Define the region to export
  maxPixels: 1e13 // Set a high enough maxPixels value to avoid errors
};

// Step 7: Export the image to an Earth Engine asset
Export.image.toAsset(exportParams);
