//*********************************************************************
//This code performs stratified sampling across aggregated LiDAR tree cover at 30m res with 7 properties: 
//Fraction of tree cover in a 30m pixel based on 5m height (b1), latitude and longitude. 
//The code also adds the Ecological region names as well as 3 Terrain properties: Elevation, Slope and Aspect
//*********************************************************************

// Function to sample an image and add latitude and longitude properties
var sampleAndAddCoordinates = function(image, numPoints, region, scale, seed) {
  var samples = image.stratifiedSample({
    numPoints: numPoints,
    classBand: 'b1', // Dummy class band for stratification
    region: region,
    scale: scale,
    geometries: true,
    seed: seed
  });

  var samplesWithCoordinates = samples.map(function(feature) {
    var coordinates = feature.geometry().coordinates();
    return feature.set({
      'longitude': coordinates.get(0),
      'latitude': coordinates.get(1)
    });
  });

  return samplesWithCoordinates;
};

// List of LiDAR images to process
var images = [
  image, image2, image3, image4, image5, image6, image7, image8,
  image9, image10, image11, image12, image13, image14, image15, image16,
  image17, image18, image19, image20, image21, image22, image23, image24,
  image25, image26, image27, image28, image29, image30, image31, image32
];

// Initialize an empty feature collection to hold all samples
var concatenatedSamples = ee.FeatureCollection([]);

// Parameters for sampling
var numPoints = 10; // Number of samples per image
var scale = 30; // Spatial resolution (30m)
var seed = 0;  // Random seed for reproducibility

// Sample each image and concatenate results
images.forEach(function(img) {
  var samples = sampleAndAddCoordinates(img, numPoints, img.geometry(), scale, seed);
  concatenatedSamples = concatenatedSamples.merge(samples);
});

// Export the concatenated samples to Google Drive
Export.table.toDrive({
  collection: concatenatedSamples,
  description: 'LiDAR_Data_Samples_With_Coordinates',
  fileFormat: 'CSV'
});

// Function to attach the GEZ_TERM property from polygons to points
var attachProperties = function(point) {
  var intersectingPolygons = polygons.filterBounds(point.geometry());
  var polygonWithProperties = ee.Feature(intersectingPolygons.first());
  var gezTerm = polygonWithProperties.get('GEZ_TERM');
  return point.set('GEZ_TERM', gezTerm);
};

// Map the function to attach GEZ_TERM to all points
var pointsWithProperties = concatenatedSamples.map(attachProperties);

// Function to sample DEM, slope, and aspect values for each point
var addDEM_Slope_Aspect = function(point) {
  var slope = ee.Terrain.slope(dem);
  var aspect = ee.Terrain.aspect(dem);

  var demValue = dem.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point.geometry(),
    scale: 30
  }).get('elevation');

  var slopeValue = slope.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point.geometry(),
    scale: 30
  }).get('slope');

  var aspectValue = aspect.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point.geometry(),
    scale: 30
  }).get('aspect');

  return point.set({
    'DEM': demValue,
    'Slope': slopeValue,
    'Aspect': aspectValue
  });
};

// Add terrain properties to all points
var pointsWithTerrain = pointsWithProperties.map(addDEM_Slope_Aspect);

// Export the final points with all properties to Google Drive
Export.table.toDrive({
  collection: pointsWithTerrain,
  description: 'LiDAR_Points_With_Terrain_Properties',
  fileNamePrefix: 'LiDAR_Points_With_Terrain',
  fileFormat: 'CSV'
});

// Print the final collection to the console
print('Points with Terrain and GEZ Properties:', pointsWithTerrain);
