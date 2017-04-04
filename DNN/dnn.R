## Load all packages 
suppressMessages(library(h2o))
suppressMessages(library(caret))
#h2o.shutdown(prompt = TRUE)
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE,min_mem_size = "3g")
### note: unzip the file and point it to your folder ###
mnistPath = './mnist_train.csv'
mnist.hex = h2o.importFile(path = mnistPath, destination_frame = "mnist.hex")
train <- as.data.frame(mnist.hex)
train$C1 <- as.factor(train$C1)
train_h2o <- as.h2o(train)

#############################################################
######## training the 2-hidden layer DNN ####################
#############################################################
model <- h2o.deeplearning(x = 2:785, y = 1, training_frame = train_h2o, activation = "RectifierWithDropout", 
                   input_dropout_ratio = 0.2, hidden_dropout_ratios = c(0.5,0.5),
                   balance_classes = TRUE, hidden = c(800,800), epochs = 200)

summary(model)

Model Details:
==============

H2OMultinomialModel: deeplearning
Model Key:  DeepLearning_model_R_1457383977741_20 
Status of Neuron Layers: predicting C1, 10-class classification, multinomial distribution, CrossEntropy loss, 1,223,210 weights/biases, 14.1 MB, 3,103,670 training samples, mini-batch size 1
  layer units             type dropout       l1       l2 mean_rate rate_RMS momentum mean_weight weight_RMS mean_bias bias_RMS
1     1   717            Input 20.00 %                                                                                        
2     2   800 RectifierDropout 50.00 % 0.000000 0.000000  0.151142 0.265777 0.000000    0.062652   0.128986 -0.009662 0.139950
3     3   800 RectifierDropout 50.00 % 0.000000 0.000000  0.001894 0.001872 0.000000   -0.009936   0.045010  0.918800 0.060404
4     4    10          Softmax         0.000000 0.000000  0.058431 0.184258 0.000000   -0.220251   0.298105 -3.438205 0.466711

H2OMultinomialMetrics: deeplearning
** Reported on training data. **
Description: Metrics reported on temporary training frame with 10043 samples

Training Set Metrics: 
=====================
Metrics reported on temporary training frame with 10043 samples 

MSE: (Extract with `h2o.mse`) 0.004803317
R^2: (Extract with `h2o.r2`) 0.9994204
Logloss: (Extract with `h2o.logloss`) 0.0152678
Confusion Matrix: Extract with `h2o.confusionMatrix(<model>,train = TRUE)`)
=========================================================================
        X0  X1   X2   X3  X4   X5  X6   X7  X8   X9       Error        Rate
0      992   0    3    0   0    0   0    0   0    0 0.003015075     3 / 995
1        0 993    6    0   0    0   0    1   0    0 0.007000000   7 / 1,000
2        0   0  985    0   0    0   0    2   0    0 0.002026342     2 / 987
3        0   0    2 1006   0    0   0    2   0    0 0.003960396   4 / 1,010
4        0   0    1    0 965    0   0    1   0    0 0.002068252     2 / 967
5        0   0    1    1   0 1016   3    1   1    0 0.006842620   7 / 1,023
6        0   1    2    0   0    0 973    0   0    0 0.003073770     3 / 976
7        0   0    1    0   0    0   0 1050   0    4 0.004739336   5 / 1,055
8        0   2    1    1   0    2   1    0 993    0 0.007000000   7 / 1,000
9        0   0    2    0   0    3   0    6   0 1019 0.010679612  11 / 1,030
Totals 992 996 1004 1008 965 1021 977 1063 994 1023 0.005078164 51 / 10,043

Hit Ratio Table: Extract with `h2o.hit_ratio_table(<model>,train = TRUE)`
=======================================================================
Top-10 Hit Ratios: 
    k hit_ratio
1   1  0.994922
2   2  0.997909
3   3  0.998506
4   4  0.999004
5   5  0.999303
6   6  0.999403
7   7  0.999602
8   8  0.999801
9   9  1.000000
10 10  1.000000



Scoring History: 
            timestamp          duration training_speed  epochs iterations      samples training_MSE training_r2 training_logloss training_classification_error
1 2016-03-07 13:23:10         0.000 sec                0.00000          0     0.000000                                                                        
2 2016-03-07 13:23:19        12.653 sec   252 rows/sec 0.01564          1  1053.000000      0.15734     0.98101          0.82775                       0.18351
3 2016-03-07 13:24:45  1 min 39.899 sec   356 rows/sec 0.43510         28 29288.000000      0.05277     0.99363          0.18903                       0.06024
4 2016-03-07 13:26:17  3 min 11.410 sec   381 rows/sec 0.93502         60 62940.000000      0.03894     0.99530          0.14292                       0.04371
5 2016-03-07 13:27:44  4 min 39.398 sec   387 rows/sec 1.40415         90 94519.000000      0.03247     0.99608          0.11907                       0.03515

---
             timestamp     duration training_speed   epochs iterations        samples training_MSE training_r2 training_logloss training_classification_error
74 2016-03-07 15:17:30  1:54:25.025   467 rows/sec 42.89270       2749 2887279.000000      0.00500     0.99940          0.01671                       0.00597
75 2016-03-07 15:19:04  1:55:58.971   467 rows/sec 43.51456       2789 2929139.000000      0.00517     0.99938          0.01791                       0.00558
76 2016-03-07 15:20:41  1:57:35.599   468 rows/sec 44.15401       2830 2972183.000000      0.00493     0.99940          0.01697                       0.00597
77 2016-03-07 15:22:17  1:59:12.988   468 rows/sec 44.80814       2872 3016215.000000      0.00518     0.99938          0.01748                       0.00587
78 2016-03-07 15:24:00  2:00:55.642   468 rows/sec 45.46451       2914 3060398.000000      0.00465     0.99944          0.01547                       0.00518
79 2016-03-07 15:25:43  2:02:38.456   468 rows/sec 46.10735       2955 3103670.000000      0.00480     0.99942          0.01527                       0.00508

#############################################################
######## scoring test set, confusion matrix #################
#############################################################

### note: unzip the file and point it to your folder ###
test_h2o <- h2o.importFile(path = './mnist_test.csv', destination_frame = "test_h2o")
y2 <- as.factor(as.matrix(test_h2o[, 1]))
yhat <- h2o.predict(model, test_h2o)
yhat2 <-as.factor(as.matrix(yhat[,1]))
confusionMatrix(yhat2,y2)

Confusion Matrix and Statistics

          Reference
Prediction    0    1    2    3    4    5    6    7    8    9
         0  969    1    2    0    0    2    4    1    0    1
         1    1 1127    1    0    0    0    4    4    0    2
         2    2    2 1018    5    5    2    3    9    5    6
         3    0    1    1  994    0    8    0    0    4    6
         4    1    0    1    0  956    1    4    0    4    7
         5    1    1    0    2    0  868    3    0    4    4
         6    3    2    2    0    7    2  940    0    1    0
         7    1    1    4    5    4    1    0 1008    3    8
         8    1    0    3    2    3    6    0    2  951    1
         9    1    0    0    2    7    2    0    4    2  974

Overall Statistics
                                          
               Accuracy : 0.9805          
                 95% CI : (0.9776, 0.9831)
    No Information Rate : 0.1135          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9783          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
Sensitivity            0.9888   0.9930   0.9864   0.9842   0.9735   0.9731   0.9812   0.9805   0.9764   0.9653
Specificity            0.9988   0.9986   0.9957   0.9978   0.9980   0.9984   0.9981   0.9970   0.9980   0.9980
Pos Pred Value         0.9888   0.9895   0.9631   0.9803   0.9815   0.9830   0.9822   0.9739   0.9814   0.9819
Neg Pred Value         0.9988   0.9991   0.9984   0.9982   0.9971   0.9974   0.9980   0.9978   0.9975   0.9961
Prevalence             0.0980   0.1135   0.1032   0.1010   0.0982   0.0892   0.0958   0.1028   0.0974   0.1009
Detection Rate         0.0969   0.1127   0.1018   0.0994   0.0956   0.0868   0.0940   0.1008   0.0951   0.0974
Detection Prevalence   0.0980   0.1139   0.1057   0.1014   0.0974   0.0883   0.0957   0.1035   0.0969   0.0992
Balanced Accuracy      0.9938   0.9958   0.9910   0.9910   0.9858   0.9857   0.9897   0.9888   0.9872   0.9817





