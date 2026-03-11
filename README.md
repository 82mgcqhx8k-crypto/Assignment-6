# Assignment-6

# Answer for 2.
  Based on the evaluated metrics, the Linear Regression model demonstrates the best overall performance among the three models. 
It achieves the lowest Mean Squared Error, indicating that its predictions are, on average, closer to the actual values. 
Additionally, it has the highest R-squared and explained variance scores which suggest that,
it explains a greater proportion of the variance in the test data and provides a better fit overall. 
These metrics collectively show that the linear regression model captures complex relationships 
more effectively than the random forest and decision tree models. 
Therefore, the linear regression model approach offers a robust balance of accuracy and generalization, 
making it the best performing model in this scenario despite its reduced interpretability compared to the simpler models.

# Pros and Cons:
  
Linear Regression:
  - Pros: Simple, interpretable, fast
  - Cons: Assumes linear relationships, might underperform with complex data
  
Decision Tree:
  - Pros: Handles non-linear patterns well, it is easy to interpret
  - Cons: Prone to overfitting, can be at times less stable
  
Random Forest:
  - Pros: Reduces overfitting, captures complex patterns well, it is robust
  - Cons: Less interpretable, computationally intensive


# Purpose
   This project aims to demonstrate the process of building and evaluating multiple regression models using the built in diabetes dataset. 
The primary goal was to compare different machine learning algorithms by training them on the dataset and 
assessing their performance through key metrics such as r-squared, mean squared error, and explained variance score. 
This comparison helps in understanding which model provides the most accurate predictions, 
while providing valueable information about each model. 
Additionally, ensemble models like Random Forest, while highly effective, are less interpretable than simpler linear models. 
Overall, this project provides a foundational understanding of applying different regression techniques 
to medical data and evaluating their relative performances.
