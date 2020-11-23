# ML Tutorials
Welcome to ML tutorial

## Regression
  In regression what we need to do is finding **Best Slope** for given linear data set. So how can we find the best slope?
  If we think data as x set and y set like in the given image,
  
  ![example data set](https://github.com/EnsarErayAkkaya/MLTutorial/blob/master/Regression/Images/RegressionExampleDataSet.jpg)
  
  We can calculate best slope with following function
  
  ![BestSlope](https://github.com/EnsarErayAkkaya/MLTutorial/blob/master/Regression/Images/RegressionMSlope.jpg)
  
  This expression means that avarage of all Xs  
  ![Mean X]( https://github.com/EnsarErayAkkaya/MLTutorial/blob/master/Regression/Images/mean(X).jpg)
 
  Now we need to calculate **Y intercept**, formula of intercept is 
  ![Mean X](https://github.com/EnsarErayAkkaya/MLTutorial/blob/master/Regression/Images/RegressionYIntercept.jpg)
  
  #### Now Finally our function is ready, we can calculate every y for every is with: **y = mx + b** 
  
  When we connect our ys we calcualte with our function we will have following slope:
  ![Mean X](https://github.com/EnsarErayAkkaya/MLTutorial/blob/master/Regression/Images/RegressionSlopeX.jpg)
  
  But **how can we sure this is a good line?** We need to use **Coefficient of determination**.
  ![Mean X](https://github.com/EnsarErayAkkaya/MLTutorial/blob/master/Regression/Images/CoefficientOfDeterminationFormula.jpg)
  
  This formula will give us accuracy of line with comparison with mean of ys line with using **squared error(distance)**. As an example if squarred error of mean line of ys is 0.5   and squared error of our line is 0.1 it means that 1 - (0.1/0.5) = 0.8 and this is our accuracy.
  
  
## K Nearest Neighbours
  
  
