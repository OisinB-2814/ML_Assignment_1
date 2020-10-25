LDA\_OB
================

-   Import the necessary packages

``` r
library(klaR)
library(tidyverse)
library(caret)
library(MASS)
library(scales)
theme_set(theme_classic())
```

-   Import the training dataset and clean

``` r
beers_train <- read.table("C:/Users/Oisin/Desktop/ML Assignment 1/Data/beer_training.txt")

colnames(beers_train) <- c('calorific_value', 'nitrogen', 'turbidity', 'style', 'alcohol', 'sugars', 'bitterness', 'beer_id', 'colour', 'degree_of_fermentation')

rownames(beers_train) <- beers_train$beer_id

beers_train <- arrange(beers_train,by=style)

beers_train <- beers_train %>% relocate(style)

beers_train <- subset(beers_train, select = -c(beer_id))

head(beers_train)
```

    ##   style calorific_value  nitrogen turbidity  alcohol sugars bitterness
    ## 1   ale        45.30531 0.4595482  1.917273 4.227692  16.67   12.56895
    ## 2   ale        43.88938 0.5489770  3.186364 4.289231  16.73   14.97400
    ## 3   ale        41.58850 0.5428470  1.568182 4.344615  16.48   11.84879
    ## 4   ale        44.55310 0.4803009  1.871818 4.424615  18.59   13.87963
    ## 5   ale        41.01327 0.4418605  2.345455 4.264615  16.35   12.18605
    ## 6   ale        42.78319 0.3604634  1.889091 4.172308  16.71   15.09374
    ##   colour degree_of_fermentation
    ## 1  11.04               62.17857
    ## 2  13.44               63.03286
    ## 3  14.04               63.46857
    ## 4  12.48               63.53143
    ## 5  12.12               63.74714
    ## 6  11.40               63.91286

-   Import the test dataset and clean

``` r
beers_test <- read.table("C:/Users/Oisin/Desktop/ML Assignment 1/Data/beer_test.txt")

colnames(beers_test) <- c('calorific_value', 'nitrogen', 'turbidity', 'style', 'alcohol', 'sugars', 'bitterness', 'beer_id', 'colour', 'degree_of_fermentation')

rownames(beers_test) <- beers_test$beer_id

beers_test <- arrange(beers_test,by=style)

beers_test <- beers_test %>% relocate(style)

beers_test <- subset(beers_test, select = -c(beer_id))

head(beers_test)
```

    ##   style calorific_value  nitrogen turbidity  alcohol sugars bitterness
    ## 1   ale        41.72124 0.5032758 2.6281818 4.015385  16.73  10.452789
    ## 2   ale        42.42920 0.5255122 1.7763636 4.092308  16.72  10.999526
    ## 3   ale        45.88053 0.4432328 2.6281818 4.276923  16.68  13.456368
    ## 4   ale        45.30531 0.4716678 1.8063636 4.126154  18.84   9.202737
    ## 5   ale        38.97788 0.3928458 2.2727273 4.015385  16.77   9.457895
    ## 6   ale        41.14602 0.3964436 0.8854545 4.021538  16.50  13.026105
    ##   colour degree_of_fermentation
    ## 1  13.44               55.33714
    ## 2  12.24               58.38000
    ## 3  10.92               58.38286
    ## 4  10.92               58.52571
    ## 5  10.56               58.90000
    ## 6  14.16               59.41429

-   Apply a preprocessing technique to my data to get it ready for LDA:
    “center“: subtract mean from values. “scale“: divide values by
    standard deviation.

``` r
preproc.param <- beers_train %>% 
  preProcess(method = c("center", "scale"))
# Transform the data using the estimated parameters
train.transformed <- preproc.param %>% predict(beers_train)
test.transformed <- preproc.param %>% predict(beers_test)
```

``` r
head(train.transformed)
```

    ##   style calorific_value  nitrogen   turbidity   alcohol     sugars
    ## 1   ale      1.39766656 1.0165582  0.06541696 0.9149941 -0.6246594
    ## 2   ale      0.85802066 1.6788234  1.48820089 1.1606566 -0.5707820
    ## 3   ale     -0.01890392 1.6334274 -0.32595054 1.3817530 -0.7952712
    ## 4   ale      1.11097968 1.1702426  0.01445765 1.7011143  1.0994179
    ## 5   ale     -0.23813507 0.8855719  0.54545366 1.0623916 -0.9120056
    ## 6   ale      0.43642230 0.2827859  0.03382219 0.6938977 -0.5887411
    ##   bitterness      colour degree_of_fermentation
    ## 1  1.0624650 -0.09892468             -0.8468714
    ## 2  1.7657103  0.74530518             -0.7011621
    ## 3  0.8518885  0.95636264             -0.6268455
    ## 4  1.4457134  0.40761324             -0.6161244
    ## 5  0.9505053  0.28097876             -0.5793316
    ## 6  1.8007218  0.02770980             -0.5510669

``` r
head(test.transformed)
```

    ##   style calorific_value  nitrogen   turbidity    alcohol     sugars
    ## 1   ale      0.03168788 1.3403828  0.86242057 0.06745815 -0.5707820
    ## 2   ale      0.30151083 1.5050546 -0.09255690 0.37453638 -0.5797615
    ## 3   ale      1.61689771 0.8957346  0.86242057 1.11152413 -0.6156798
    ## 4   ale      1.39766656 1.1063100 -0.05892375 0.50965080  1.3239071
    ## 5   ale     -1.01387606 0.5225935  0.46391877 0.06745815 -0.5348637
    ## 6   ale     -0.18754327 0.5492373 -1.09135937 0.09202441 -0.7773120
    ##   bitterness     colour degree_of_fermentation
    ## 1 0.44369346  0.7453052              -2.013764
    ## 2 0.60356111  0.3231903              -1.494766
    ## 3 1.32194984 -0.1411362              -1.494279
    ## 4 0.07817395 -0.1411362              -1.469913
    ## 5 0.15278296 -0.2677706              -1.406073
    ## 6 1.19613947  0.9985741              -1.318355

-   Fit the model to the transformed data and look at the predictions on
    the test set

``` r
# Fit the model
model <- lda(style~., data = train.transformed)
# Make predictions
predictions <- model %>% predict(test.transformed)
```

``` r
model
```

    ## Call:
    ## lda(style ~ ., data = train.transformed)
    ## 
    ## Prior probabilities of groups:
    ##       ale     lager     stout 
    ## 0.3387097 0.3548387 0.3064516 
    ## 
    ## Group means:
    ##       calorific_value   nitrogen   turbidity    alcohol     sugars
    ## ale        0.06300661  1.1381186  0.07514776  0.9155790 -0.3127262
    ## lager     -0.17182915 -0.7624341 -0.87739976 -0.9774453 -0.5003740
    ## stout      0.12932118 -0.3751021  0.93287851  0.1198231  0.9250252
    ##        bitterness     colour degree_of_fermentation
    ## ale    0.86329768  0.5000765              0.4972208
    ## lager  0.03597728  0.5031647             -0.1898339
    ## stout -0.99582903 -1.1353279             -0.3297522
    ## 
    ## Coefficients of linear discriminants:
    ##                                LD1         LD2
    ## calorific_value        -0.02880087  0.06814431
    ## nitrogen               -0.82797931  0.85051275
    ## turbidity               0.69967744  0.50491692
    ## alcohol                -0.31313101  0.94486221
    ## sugars                  0.33425179  0.23847579
    ## bitterness             -0.98852273 -0.31520076
    ## colour                 -0.58409941 -0.43812625
    ## degree_of_fermentation -0.06004528  0.29024988
    ## 
    ## Proportion of trace:
    ##    LD1    LD2 
    ## 0.5864 0.4136

``` r
plot(model)
```

![](LDA2_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

-   Linear discriminants are calculated by the prediction and are the
    result on dimensionality reduction on the feature attributes

``` r
# Predicted classes
(predictions$class)
```

    ##  [1] ale   ale   ale   ale   ale   ale   ale   ale   ale   ale   lager
    ## [12] lager lager lager lager lager lager lager lager lager stout stout
    ## [23] stout stout stout stout stout stout stout stout
    ## Levels: ale lager stout

``` r
# Predicted probabilities of class membership.
head(predictions$posterior) 
```

    ##         ale        lager        stout
    ## 1 0.9819117 0.0179612435 1.271054e-04
    ## 2 0.9990278 0.0009709113 1.254089e-06
    ## 3 0.9998192 0.0001688323 1.192988e-05
    ## 4 0.9872715 0.0030689841 9.659546e-03
    ## 5 0.4877489 0.4474218643 6.482927e-02
    ## 6 0.6840280 0.3159720353 1.414997e-08

``` r
# Linear discriminants
head(predictions$x)
```

    ##          LD1        LD2
    ## 1 -1.4722301  0.4543587
    ## 2 -2.3263186  0.7038103
    ## 3 -1.8732618  1.4223209
    ## 4 -0.6211304  1.4142448
    ## 5 -0.1890016  0.2068570
    ## 6 -3.1881055 -1.3922903

-   Look at the resulting plot of LD1 vs LD2

``` r
predictions <- as.data.frame(predictions)

head(predictions)
```

    ##   class posterior.ale posterior.lager posterior.stout      x.LD1
    ## 1   ale     0.9819117    0.0179612435    1.271054e-04 -1.4722301
    ## 2   ale     0.9990278    0.0009709113    1.254089e-06 -2.3263186
    ## 3   ale     0.9998192    0.0001688323    1.192988e-05 -1.8732618
    ## 4   ale     0.9872715    0.0030689841    9.659546e-03 -0.6211304
    ## 5   ale     0.4877489    0.4474218643    6.482927e-02 -0.1890016
    ## 6   ale     0.6840280    0.3159720353    1.414997e-08 -3.1881055
    ##        x.LD2
    ## 1  0.4543587
    ## 2  0.7038103
    ## 3  1.4223209
    ## 4  1.4142448
    ## 5  0.2068570
    ## 6 -1.3922903

``` r
ggplot(predictions, aes(x=x.LD1,y=x.LD2,color=class)) +
  geom_point()
```

![](LDA2_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

-   Add LDA1 and LDA2 to our transformed training data

``` r
lda.data <- cbind(train.transformed, predict(model)$x)

head(lda.data)
```

    ##   style calorific_value  nitrogen   turbidity   alcohol     sugars
    ## 1   ale      1.39766656 1.0165582  0.06541696 0.9149941 -0.6246594
    ## 2   ale      0.85802066 1.6788234  1.48820089 1.1606566 -0.5707820
    ## 3   ale     -0.01890392 1.6334274 -0.32595054 1.3817530 -0.7952712
    ## 4   ale      1.11097968 1.1702426  0.01445765 1.7011143  1.0994179
    ## 5   ale     -0.23813507 0.8855719  0.54545366 1.0623916 -0.9120056
    ## 6   ale      0.43642230 0.2827859  0.03382219 0.6938977 -0.5887411
    ##   bitterness      colour degree_of_fermentation       LD1      LD2
    ## 1  1.0624650 -0.09892468             -0.8468714 -2.273117 1.171093
    ## 2  1.7657103  0.74530518             -0.7011621 -3.066380 2.111687
    ## 3  0.8518885  0.95636264             -0.6268455 -3.641533 1.469833
    ## 4  1.4457134  0.40761324             -0.6161244 -2.786219 2.134710
    ## 5  0.9505053  0.28097876             -0.5793316 -2.051172 1.207839
    ## 6  1.8007218  0.02770980             -0.5510669 -2.400265 0.062891

-   Check that the resulting plot is the same as above

``` r
ggplot(lda.data, aes(LD1, LD2)) +
  geom_point(aes(color = style))
```

![](LDA2_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
ggsave("LDA_OB.png")
```

    ## Saving 7 x 5 in image

-   Use the model to give a score for the accuracy on the test data -
    100%

``` r
mean(predictions$class==test.transformed$style)
```

    ## [1] 1

-   Look at the confusion matrix for the training & test data - note
    that it is 98% accurate while the test set is 100%

``` r
beers.train.lda.predict <- train(style ~ ., method = "lda", data = train.transformed)
confusionMatrix(train.transformed$style, predict(beers.train.lda.predict, train.transformed))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction ale lager stout
    ##      ale    42     0     0
    ##      lager   2    41     1
    ##      stout   0     0    38
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9758         
    ##                  95% CI : (0.9309, 0.995)
    ##     No Information Rate : 0.3548         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9637         
    ##                                          
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: ale Class: lager Class: stout
    ## Sensitivity              0.9545       1.0000       0.9744
    ## Specificity              1.0000       0.9639       1.0000
    ## Pos Pred Value           1.0000       0.9318       1.0000
    ## Neg Pred Value           0.9756       1.0000       0.9884
    ## Prevalence               0.3548       0.3306       0.3145
    ## Detection Rate           0.3387       0.3306       0.3065
    ## Detection Prevalence     0.3387       0.3548       0.3065
    ## Balanced Accuracy        0.9773       0.9819       0.9872

``` r
beers.test.lda.predict <- train(style ~ ., method = "lda", data = beers_test)
confusionMatrix(beers_test$style, predict(beers.test.lda.predict, beers_test))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction ale lager stout
    ##      ale    10     0     0
    ##      lager   0    10     0
    ##      stout   0     0    10
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.8843, 1)
    ##     No Information Rate : 0.3333     
    ##     P-Value [Acc > NIR] : 4.857e-15  
    ##                                      
    ##                   Kappa : 1          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: ale Class: lager Class: stout
    ## Sensitivity              1.0000       1.0000       1.0000
    ## Specificity              1.0000       1.0000       1.0000
    ## Pos Pred Value           1.0000       1.0000       1.0000
    ## Neg Pred Value           1.0000       1.0000       1.0000
    ## Prevalence               0.3333       0.3333       0.3333
    ## Detection Rate           0.3333       0.3333       0.3333
    ## Detection Prevalence     0.3333       0.3333       0.3333
    ## Balanced Accuracy        1.0000       1.0000       1.0000

-   Plot the boundaries computed by LDA to visually see where the
    mistakes were made in the training data

``` r
fit <- lda(style ~ ., data = train.transformed, prior = rep(1, 3)/3)
datPred <- data.frame(style=predict(fit)$class,predict(fit)$x)
#Create decision boundaries
fit2 <- lda(style ~ LD1 + LD2, data=datPred, prior = rep(1, 3)/3)
datPred <- datPred[-65,]
ld1lim <- expand_range(c(min(datPred$LD1),max(datPred$LD1)),mul=0.05)
ld2lim <- expand_range(c(min(datPred$LD2),max(datPred$LD2)),mul=0.05)
ld1 <- seq(ld1lim[[1]], ld1lim[[2]], length.out=300)
ld2 <- seq(ld2lim[[1]], ld1lim[[2]], length.out=300)
newdat <- expand.grid(list(LD1=ld1,LD2=ld2))
preds <-predict(fit2,newdata=newdat)
predclass <- preds$class
postprob <- preds$posterior
df <- data.frame(x=newdat$LD1, y=newdat$LD2, class=predclass)
df$classnum <- as.numeric(df$class)
df <- cbind(df,postprob)
```

``` r
colorfun <- function(n,l=65,c=100) { hues = seq(15, 375, length=n+1); hcl(h=hues, l=l, c=c)[1:n] } # default ggplot2 colours
colors <- colorfun(3)
colorslight <- colorfun(3,l=90,c=50)
ggplot(datPred, aes(x=LD1, y=LD2) ) +
    geom_raster(data=df, aes(x=x, y=y, fill = factor(class)),alpha=0.7,show_guide=FALSE) +
    geom_contour(data=df, aes(x=x, y=y, z=classnum), colour="red2", alpha=0.5, breaks=c(1.5,2.5)) +
    geom_point(data = datPred, size = 3, aes(colour=style)) +
    scale_x_continuous(limits = ld1lim, expand=c(0,0)) +
    scale_y_continuous(limits = ld2lim, expand=c(0,0)) +
    scale_fill_manual(values=colorslight,guide=F)
```

![](LDA2_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
ggsave("LDA_OB1.png")
```

-   Plot a similar graph for the test data

``` r
fit <- lda(style ~ ., data = test.transformed, prior = rep(1, 3)/3)
datPred <- data.frame(style=predict(fit)$class,predict(fit)$x)
#Create decision boundaries
fit2 <- lda(style ~ LD1 + LD2, data=datPred, prior = rep(1, 3)/3)
ld1lim <- expand_range(c(min(datPred$LD1),max(datPred$LD1)),mul=0.05)
ld2lim <- expand_range(c(min(datPred$LD2),max(datPred$LD2)),mul=0.05)
ld1 <- seq(ld1lim[[1]], ld1lim[[2]], length.out=300)
ld2 <- seq(ld2lim[[1]], ld1lim[[2]], length.out=300)
newdat <- expand.grid(list(LD1=ld1,LD2=ld2))
preds <-predict(fit2,newdata=newdat)
predclass <- preds$class
postprob <- preds$posterior
df <- data.frame(x=newdat$LD1, y=newdat$LD2, class=predclass)
df$classnum <- as.numeric(df$class)
df <- cbind(df,postprob)
datPred
```

    ##    style        LD1        LD2
    ## 1    ale -2.8530990  3.1577613
    ## 2    ale -1.8509325  4.5232157
    ## 3    ale -2.6856989  2.7388200
    ## 4    ale -1.2457266  2.7321449
    ## 5    ale -0.6183876  2.0761018
    ## 6    ale -2.7637474  2.3081922
    ## 7    ale -0.9957294  3.4553289
    ## 8    ale -0.8460303  4.2781295
    ## 9    ale -1.4419561  2.6943110
    ## 10   ale -0.7847103  2.3048732
    ## 11 lager -6.2918589 -2.5201079
    ## 12 lager -4.9039466 -4.1757194
    ## 13 lager -5.2607622 -2.2290151
    ## 14 lager -6.2891975 -1.4054984
    ## 15 lager -4.8863925 -0.8856314
    ## 16 lager -3.9171900 -1.8840988
    ## 17 lager -3.1552451 -3.7298043
    ## 18 lager -2.5703427 -3.0713014
    ## 19 lager -3.7993549 -0.7360399
    ## 20 lager -3.7379582 -1.4062665
    ## 21 stout  4.9526098 -1.1590140
    ## 22 stout  5.2421662 -1.6900392
    ## 23 stout  6.3315396 -1.3045908
    ## 24 stout  6.2516194 -0.5248131
    ## 25 stout  5.0315504 -2.0448128
    ## 26 stout  7.2667285 -0.5951065
    ## 27 stout  5.8550898  1.4099115
    ## 28 stout  6.2384356 -0.9758494
    ## 29 stout  6.8412371 -1.2400007
    ## 30 stout  6.8872903 -0.1010802

``` r
colorfun <- function(n,l=65,c=100) { hues = seq(15, 375, length=n+1); hcl(h=hues, l=l, c=c)[1:n] } # default ggplot2 colours
colors <- colorfun(3)
colorslight <- colorfun(3,l=90,c=50)
ggplot(datPred, aes(x=LD1, y=LD2) ) +
    geom_raster(data=df, aes(x=x, y=y, fill = factor(class)),alpha=0.7,show_guide=FALSE) +
    geom_contour(data=df, aes(x=x, y=y, z=classnum), colour="red2", alpha=0.5, breaks=c(1.5,2.5)) +
    geom_point(data = datPred, size = 3, aes(colour=style)) +
    scale_x_continuous(limits = ld1lim, expand=c(0,0)) +
    scale_y_continuous(limits = ld2lim, expand=c(0,0)) +
    scale_fill_manual(values=colorslight,guide=F)
```

![](LDA2_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` r
ggsave("LDA_TEST_OB.png")
```
