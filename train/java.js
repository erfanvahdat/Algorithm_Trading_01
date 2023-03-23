// Initialize the chart data
let chartData = [];

// Set the initial value of the chart
let currentValue = Math.floor(Math.random() * 1000);

// Set the number of data points to display in the chart
const numDataPoints = 100;

// Generate the OHLC data points
for (let i = 0; i < numDataPoints; i++) {
  // Generate a random price fluctuation between -1% and 1%
  const priceFluctuation = (Math.random() * 2 - 1) / 100;

  // Calculate the open, high, low, and close values
  const openValue = currentValue;
  const highValue = currentValue + Math.max(priceFluctuation, 0);
  const lowValue = currentValue + Math.min(priceFluctuation, 0);
  const closeValue = highValue - lowValue > 0 ? currentValue + priceFluctuation : currentValue;

  // Add the OHLC data point to the chart data array
  chartData.push({
    open: openValue,
    high: highValue,
    low: lowValue,
    close: closeValue
  });

  // Update the current value with the close value of the previous data point
  currentValue = closeValue;
}

// Log the generated chart data
console.log(chartData);
