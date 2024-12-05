// First load the API file
var utils = require('users/parevalo_bu/gee-ccdc-tools:ccdcUtilities/api');

// Load the CCDC results image collection and mosaic it
var ccdResultsCollection = ee.ImageCollection('projects/CCDC/v3');
var ccdResults = ccdResultsCollection.mosaic();

// **************************************************************
// ********CCDC Synthetic Bands Extraction - Seasonal************
// **************************************************************

// Spectral band names. This list contains all possible bands in this dataset
var BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP'];

// Names of the temporal segments
var SEGS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10","S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20"];

// Obtain CCDC results in 'regular' ee.Image format
var ccdImage = utils.CCDC.buildCcdImage(ccdResults, SEGS.length, BANDS)

// Define functions for calculating indices

// Calculate NDVI
function calcNDVI(image) {
   var ndvi = ee.Image(image).normalizedDifference(['NIR', 'RED']).rename('NDVI');
   return ndvi;
}

// Calculate NBR
function calcNBR(image) {
  var nbr = ee.Image(image).normalizedDifference(['NIR', 'SWIR2']).rename('NBR');
  return nbr;
}

// Calculate EVI
function calcEVI(image) {
  var evi = ee.Image(image).expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': image.select('NIR'),
      'RED': image.select('RED'),
      'BLUE': image.select('BLUE')
    }).rename('EVI');
  return evi;
}

// Calculate EVI2
function calcEVI2(image) {
  var evi2 = ee.Image(image).expression(
    '2.5 * ((NIR - RED) / (NIR + 2.4 * RED + 1))', {
      'NIR': image.select('NIR'),
      'RED': image.select('RED')
    }).rename('EVI2');
  return evi2;
}

// Tassel Cap Transformation
function tcTrans(image) {
  var brightness = image.expression(
    '0.2043 * BLUE + 0.4158 * GREEN + 0.5524 * RED + 0.5741 * NIR + 0.3124 * SWIR1 + 0.2303 * SWIR2', {
      'BLUE': image.select('BLUE'),
      'GREEN': image.select('GREEN'),
      'RED': image.select('RED'),
      'NIR': image.select('NIR'),
      'SWIR1': image.select('SWIR1'),
      'SWIR2': image.select('SWIR2')
    }).rename('BRIGHTNESS');

  var greenness = image.expression(
    '-0.1603 * BLUE - 0.2819 * GREEN - 0.4934 * RED + 0.7940 * NIR - 0.0002 * SWIR1 - 0.1446 * SWIR2', {
      'BLUE': image.select('BLUE'),
      'GREEN': image.select('GREEN'),
      'RED': image.select('RED'),
      'NIR': image.select('NIR'),
      'SWIR1': image.select('SWIR1'),
      'SWIR2': image.select('SWIR2')
    }).rename('GREENNESS');

  var wetness = image.expression(
    '0.0315 * BLUE + 0.2021 * GREEN + 0.3102 * RED + 0.1594 * NIR - 0.6806 * SWIR1 - 0.6109 * SWIR2', {
      'BLUE': image.select('BLUE'),
      'GREEN': image.select('GREEN'),
      'RED': image.select('RED'),
      'NIR': image.select('NIR'),
      'SWIR1': image.select('SWIR1'),
      'SWIR2': image.select('SWIR2')
    }).rename('WETNESS');

  return ee.Image([brightness, greenness, wetness]);
}

// Calculate indices and add as bands
function addIndices(image) {
  var ndvi = calcNDVI(image);
  var nbr = calcNBR(image);
  var evi = calcEVI(image);
  var evi2 = calcEVI2(image);
  var tc = tcTrans(image);
  
  return image.addBands([ndvi, nbr, evi, evi2, tc]);
}


var inputDate = '2013-01-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var JAN_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
// Add indices to the image
var JAN_SyntheticBands = addIndices(JAN_SyntheticBands);
// // Print to console to inspect
// print(JANimageWithIndices);

var inputDate = '2013-02-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var FEB_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var FEB_SyntheticBands = addIndices(FEB_SyntheticBands);

var inputDate = '2013-03-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var MAR_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var MAR_SyntheticBands = addIndices(MAR_SyntheticBands);

var inputDate = '2013-04-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var APR_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var APR_SyntheticBands = addIndices(APR_SyntheticBands);

var inputDate = '2013-05-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var MAY_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var MAY_SyntheticBands = addIndices(MAY_SyntheticBands);

var inputDate = '2013-06-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var JUN_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var JUN_SyntheticBands = addIndices(JUN_SyntheticBands);

var inputDate = '2013-07-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var JULY_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var JULY_SyntheticBands = addIndices(JULY_SyntheticBands);

var inputDate = '2013-08-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var AUG_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var AUG_SyntheticBands = addIndices(AUG_SyntheticBands);

var inputDate = '2013-09-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var SEPT_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var SEPT_SyntheticBands = addIndices(SEPT_SyntheticBands);

var inputDate = '2013-10-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var OCT_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var OCT_SyntheticBands = addIndices(OCT_SyntheticBands);

var inputDate = '2013-11-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var NOV_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var NOV_SyntheticBands = addIndices(NOV_SyntheticBands);

var inputDate = '2013-12-01'
var dateParams = {inputFormat: 3, inputDate: inputDate, outputFormat: 1}
var formattedDate = utils.Dates.convertDate(dateParams)

// Obtain synthetic image
var DEC_SyntheticBands = utils.CCDC.getMultiSynthetic(ccdImage, formattedDate, 1, BANDS, SEGS)
var DEC_SyntheticBands = addIndices(DEC_SyntheticBands);



// Define the function to rename the bands of an image
var renameBands = function(image, suffix) {
  var bandNames = image.bandNames();
  var newBandNames = bandNames.map(function(band) {
    return ee.String(band).cat('_').cat(suffix);
  });
  return image.rename(newBandNames);
};

// Rename bands of each image
var Jan_renamed = renameBands(JAN_SyntheticBands, 'Jan');
var Feb_renamed = renameBands(FEB_SyntheticBands, 'Feb');
var Mar_renamed = renameBands(MAR_SyntheticBands, 'Mar');
var Apr_renamed = renameBands(APR_SyntheticBands, 'Apr');
var May_renamed = renameBands(MAY_SyntheticBands, 'May');
var Jun_renamed = renameBands(JUN_SyntheticBands, 'Jun');
var Jul_renamed = renameBands(JULY_SyntheticBands, 'Jul');
var Aug_renamed = renameBands(AUG_SyntheticBands, 'Aug');
var Sep_renamed = renameBands(SEPT_SyntheticBands, 'Sep');
var Oct_renamed = renameBands(OCT_SyntheticBands, 'Oct');
var Nov_renamed = renameBands(NOV_SyntheticBands, 'Nov');
var Dec_renamed = renameBands(DEC_SyntheticBands, 'Dec');

// Stack the images together
var stackedImage = Jan_renamed.addBands(Feb_renamed)
  .addBands(Mar_renamed)
  .addBands(Apr_renamed)
  .addBands(May_renamed)
  .addBands(Jun_renamed)
  .addBands(Jul_renamed)
  .addBands(Aug_renamed)
  .addBands(Sep_renamed)
  .addBands(Oct_renamed)
  .addBands(Nov_renamed)
  .addBands(Dec_renamed);
  
var Masked_Stack = stackedImage
  
// var Masked_Stack = stackedImage.updateMask(FinalMask_scalarBand)
// print(Masked_Stack)
// Map.addLayer(Masked_Stack,{},'MonthlyStack_Masked')

// Define batch size
var batchSize = 1000;  // Adjust the batch size according to your requirements
// var numSamples = RAP_Sample_Points.size().getInfo();
var numSamples = LiDAR_2mMasked.size().getInfo();

// Initialize an empty FeatureCollection to store the results
var CCDC_featureCollection = ee.FeatureCollection([]);

// Process the data in batches
for (var i = 0; i < numSamples; i += batchSize) {
  // Get the current batch of sample points
  var batch = LiDAR_2mMasked.toList(batchSize, i);

  // Perform reduceRegions on the current batch
  var extractedData = Masked_Stack.reduceRegions({
    collection: ee.FeatureCollection(batch),
    reducer: ee.Reducer.first(),
    scale: 30,
    crs: 'EPSG:32615',  // Specify CRS to ensure proper geometry handling
    tileScale: 16  // Increase tile scale to handle larger computations
  });

  // Add lat-lon properties to each feature in extractedData
  extractedData = extractedData.map(function(feature) {
    var lat = feature.geometry().coordinates().get(1);
    var lon = feature.geometry().coordinates().get(0);
    return feature.set({
      'longitude': lon,
      'latitude': lat
    });
  });
