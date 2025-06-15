# Linear Regression: Study Hours vs Scores

This project demonstrates a simple linear regression model that predicts student scores based on the number of hours they studied.

## Project Files
- `Linear_regression.py`: Python script containing the code for data loading, visualization, model training, and evaluation.
- `student_scores - student_scores.csv`: Dataset containing study hours and corresponding scores of students.
- `Linear_regression_Plot.png`: Output plot showing the data points and regression line (generated when running the script).

## Requirements
To run this code, you need the following Python libraries installed:
- pandas
- numpy
- matplotlib
- scikit-learn

You can install these packages using pip:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## How to Run the Code
1. Clone this repository or download the files to your local machine.
2. Ensure all required libraries are installed (see Requirements above).
3. Run the Python script:
   ```bash
   python Linear_regression.py
   ```

## Expected Output
When you run the script, it will:
1. Display the first few rows of the dataset in the console.
2. Print the Mean Squared Error (MSE) of the model (expected around 18.94).
3. Show a plot with two subplots:
   - Top: Scatter plot of all data points (Hours vs Scores)
   - Bottom: Regression line plotted against the test data points

## Code Overview
The script performs the following steps:
1. Loads the dataset from a URL
2. Creates visualizations of the data
3. Splits the data into training and test sets
4. Trains a linear regression model
5. Evaluates the model using Mean Squared Error
6. Plots the regression line against the test data

## Dataset
The dataset contains two columns:
- `Hours`: Number of hours studied (feature)
- `Scores`: Percentage score achieved (target)

## License
This project uses a standard dataset for educational purposes. The code is free to use and modify.
