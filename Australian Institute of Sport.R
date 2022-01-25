library("PerformanceAnalytics")

library(tidyverse) #sudo apt install libcurl4-openssl-dev was needed :libxml2-dev,libcurl4-openssl-dev
library(DAAG)
library(plotly)

library(caret) 
library(keras) #install.packages("keras") && install_keras & same for tensorflow
library(tensorflow)
set.seed(123)

#Keep only numerical variables
df_x=ais[,1:11]

#Scale data to have mean 0 and standard deaviation 1
preproc1 <- preProcess(df_x, method=c("center", "scale"))
df_x <- predict(preproc1, df_x)
#PCA
pca <- prcomp(df_x)
# plot cumulative plot
qplot(x = 1:11, y = cumsum(pca$sdev)/sum(pca$sdev), geom = "line",xlab = "Number of Principal Components",ylab="Cumulative Variance explained",xlim=c(1,11))
#individual variance plot
qplot(x = 1:11, y = pca$sdev/sum(pca$sdev), geom = "line",xlab = "Principal Components",ylab="Individual Variance explained")
#correlation chart
chart.Correlation(df_x, histogram=TRUE, pch=25)
#3d pca plot
plot_ly(as.data.frame(pca$x), x = ~PC1, y = ~PC2, z = ~PC3, color = ~ais$sex) %>% add_markers()

#AE
x_train <- as.matrix(df_x)
# set model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 6, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 3, activation = "tanh", name = "bottleneck") %>%
  layer_dense(units = 6, activation = "tanh") %>%
  layer_dense(units = ncol(x_train))
# view model layers
#summary(model)


# compile model
model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)
# fit model
model %>% fit(
  x = x_train, 
  y = x_train, 
  epochs = 1000,
  verbose = 0
)
# evaluate the performance of the model
mse.ae3 <- evaluate(model, x_train, x_train)
mse.ae3


intermediate_layer_model <- keras_model(inputs = model$input, outputs = get_layer(model, "bottleneck")$output)
intermediate_output <- predict(intermediate_layer_model, x_train)
# plot the reduced data set
aedf3 <- data.frame(node1 = intermediate_output[,1], node2 = intermediate_output[,2], node3 = intermediate_output[,3])
plot_ly(aedf3, x = ~node1, y = ~node2, z = ~node3, color = ~ais$sex) %>% add_markers()

#######################RECONSTRUCTION ERROR multiple k############################3
###tried tanh and linear activation functions
# pCA reconstruction
pca.recon <- function(pca, x, k){
  mu <- matrix(rep(pca$center, nrow(pca$x)), nrow = nrow(pca$x), byrow = T)
  recon <- pca$x[,1:k] %*% t(pca$rotation[,1:k]) + mu
  mse <- mean((recon - x)^2)
  return(list(x = recon, mse = mse))
}
xhat <- rep(NA, 11)
for(k in 1:11){
  xhat[k] <- pca.recon(pca, x_train, k)$mse
}
ae.mse <- rep(NA, 8)
for(k in 1:8){
  modelk <- keras_model_sequential()
  modelk %>%
    layer_dense(units = 9, activation = "linear", input_shape = ncol(x_train)) %>%
    layer_dense(units = k, activation = "linear",name = "bottleneck") %>%
    layer_dense(units = 9, activation = "linear") %>%
    layer_dense(units = ncol(x_train))
  modelk %>% compile(
    loss = "mean_squared_error", 
    optimizer = "adam"
  )
  modelk %>% fit(
    x = x_train, 
    y = x_train, 
    epochs = 2000,
    verbose = 0
  )
  ae.mse[k] <- unname(evaluate(modelk, x_train, x_train))
}
df <- data.frame(k = c(1:8, 1:8), mse = c(xhat[1:8], ae.mse), method = c(rep("pca", 8), rep("autoencoder", 8)))
ggplot(df, aes(x = k, y = mse, col = method)) + geom_line()



