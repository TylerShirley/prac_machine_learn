R Markdown
----------

Download the testing and training sets from the provided links. This
ensures any updates get captured when the prediction methods are run.

    train_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    test_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

    train_set <- read.csv(train_url)
    test_set <- read.csv(test_url)

    set.seed(1221)

First all columns with NA are removed fromt he training data. Then the
timestamps, window data, names, and X variable are removed because they
have no connection to the outcome of our prediction model. The testing
set then has 86 unique variables however it’s likely not all are useful
for our prediction model. A covariate is created to have only near zero
variance variables.

This results in 53 unique variables to train our prediction model.

An evaluation testing set will be made from the training set. This is to
determine how accurate the training set is for the different models.

    train_set <- train_set[,colSums(is.na(train_set)) == 0]
    train_set <- train_set[, -c(1:7)]

    nz_cov <- nearZeroVar(train_set)
    train_set <- train_set[ ,-nz_cov]

    eval_list <- createDataPartition(train_set$classe, p = 0.8, list = FALSE)

    eval_set <- train_set[-eval_list,]
    train_set <- train_set[eval_list,]

Correlary Data
==============

With a large number of variables there could be many variables that
don’t corrilate which could account for added noise in the model. So we
can check if a large number of uncorrelated data exists so we can take
it out to slim down the amount of processing required.

    cor_var <- abs(cor(train_set[,-53]))
    diag(cor_var) <- 0
    dim(which(cor_var > 0.75, arr.ind = T))

    ## [1] 64  2

    corrplot(cor_var, type = "lower", method = "color",
             tl.cex = 0.8,
             tl.col = rgb(0,0,0))

![](Exercise_ml_files/figure-markdown_strict/unnamed-chunk-3-1.png) From
the correlation analysis we find there are only 32 unique combinations
of variables that give a correlation of greater than 0.75. With this
information we could preprocess our data just to reduce the number of
calculations needed to be performed. But because the dataset is
relatively small, there is no real need to reduce the number of
uncorrelated variables.

Prediction Modeling
===================

2 methods of prediction modeling will be explored to determine the best
fit for the testing set. This will show how a single classification tree
will compare with multiple weighed average trees in a random forest.

Regression and Classification Trees
-----------------------------------

    mod_tree <- train(classe ~., method = "rpart", data = train_set)
    pred_tree <- predict(mod_tree, eval_set)
    mat_tree <- confusionMatrix(pred_tree, eval_set$classe)
    mat_tree

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1000  326  333  288  106
    ##          B   17  247   27  111  107
    ##          C   94  186  324  244  186
    ##          D    0    0    0    0    0
    ##          E    5    0    0    0  322
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4825          
    ##                  95% CI : (0.4668, 0.4983)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3234          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8961  0.32543  0.47368   0.0000  0.44660
    ## Specificity            0.6249  0.91719  0.78080   1.0000  0.99844
    ## Pos Pred Value         0.4871  0.48527  0.31335      NaN  0.98471
    ## Neg Pred Value         0.9380  0.85003  0.87539   0.8361  0.88904
    ## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
    ## Detection Rate         0.2549  0.06296  0.08259   0.0000  0.08208
    ## Detection Prevalence   0.5233  0.12975  0.26357   0.0000  0.08335
    ## Balanced Accuracy      0.7605  0.62131  0.62724   0.5000  0.72252

    fancyRpartPlot(mod_tree$finalModel)

![](Exercise_ml_files/figure-markdown_strict/unnamed-chunk-4-1.png)

Random Forest
-------------

    tune_for <- tuneRF(train_set[,-53], train_set$classe, ntreeTry=500, stepFactor=1.5,improve=0.01, 
                   plot=FALSE, trace=TRUE, dobest=FALSE)

    ## mtry = 7  OOB error = 0.45% 
    ## Searching left ...
    ## mtry = 5     OOB error = 0.43% 
    ## 0.02857143 0.01 
    ## mtry = 4     OOB error = 0.47% 
    ## -0.08823529 0.01 
    ## Searching right ...
    ## mtry = 10    OOB error = 0.43% 
    ## 0.01470588 0.01 
    ## mtry = 15    OOB error = 0.47% 
    ## -0.1044776 0.01

    tune_for

    ##        mtry    OOBError
    ## 4.OOB     4 0.004713676
    ## 5.OOB     5 0.004331486
    ## 7.OOB     7 0.004458883
    ## 10.OOB   10 0.004267788
    ## 15.OOB   15 0.004713676

    mod_for <- randomForest(classe ~., data = train_set, tune_for = 4, ntree = 250)
    pred_for <- predict(mod_for,eval_set)
    mat_for <- confusionMatrix(pred_for, eval_set$classe)
    mat_for

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    1    0    0    0
    ##          B    0  756    2    0    0
    ##          C    0    2  678    3    0
    ##          D    0    0    4  640    3
    ##          E    0    0    0    0  718
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9962          
    ##                  95% CI : (0.9937, 0.9979)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9952          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9960   0.9912   0.9953   0.9958
    ## Specificity            0.9996   0.9994   0.9985   0.9979   1.0000
    ## Pos Pred Value         0.9991   0.9974   0.9927   0.9892   1.0000
    ## Neg Pred Value         1.0000   0.9991   0.9981   0.9991   0.9991
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1927   0.1728   0.1631   0.1830
    ## Detection Prevalence   0.2847   0.1932   0.1741   0.1649   0.1830
    ## Balanced Accuracy      0.9998   0.9977   0.9948   0.9966   0.9979

Accuracy Findings
-----------------

Decision Tree: 0.48 Random Forest: 0.99

From our analysis, the random forest is twice as accurate as a single
tree. So we will use the random forest prediction model moving forward
to evaluate the testing set.

This results in the prediction below for the week 4 quiz:

    prediction <- predict(mod_for, test_set)
    prediction

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
