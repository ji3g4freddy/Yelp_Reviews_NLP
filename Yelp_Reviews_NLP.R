# Yelp_Reviews_NLP

# Data Preprocessing & Data Cleaning

## Import Library
# data visualization
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
library(ROCR)

# data processing
library(caret)
library(tm)
library(SnowballC)
library(NLP)
library(caTools)

# data modeling
library(e1071)
library(randomForest)

## Import dataset and data overview
reviews = read.csv('yelp_review.csv',stringsAsFactors=FALSE, header=T, nrows=20000)

names(reviews)
dim(reviews)
summary(reviews)

## leave only the information we want to analyze (text and stars)
reviews = subset(reviews, select = c("text","stars"))
summary(reviews)

## See the distribution of the stars
table(reviews$stars)
## transform stars to categorical data
reviews$stars = factor(reviews$stars, levels = c(1, 2, 3, 4, 5))

# distribution of stars
ggplot(data = reviews, aes(stars)) +
  geom_bar(aes(y = (..count..), fill = stars)) +
  ggtitle('Stars Distribution') + 
  xlab('Number of stars') +
  ylab("Count")

## Use downsample function to deal with the class unbalance
### downsample
set.seed(123)
reviews = downSample(x = reviews, y = reviews$stars)

table(reviews$stars)

# distribution of stars
ggplot(data = reviews, aes(stars)) +
  geom_bar(aes(y = (..count..), fill = stars)) +
  ggtitle('Stars Distribution(Downsample)') + 
  xlab('Number of stars') +
  ylab("Count")

## Only left pos and neg text(exclude the neutral)
pos = subset(reviews, as.numeric(stars) > 3)
neg = subset(reviews, as.numeric(stars) < 3)
reviews_sent = rbind(pos, neg)

## create new dataset reviews_sent to focus on binary sentiment analysis
reviews_sent$sentiment[as.numeric(reviews_sent$stars) < 3] = 0
reviews_sent$sentiment[as.numeric(reviews_sent$stars) > 3] = 1

## transform sentiment into categorical data
reviews_sent$sentiment = factor(reviews_sent$sentiment, levels = c(0, 1))

# distribution of sentiment
ggplot(data = reviews_sent, aes(sentiment)) +
  geom_bar(aes(y = (..count..), fill = sentiment)) +
  ggtitle('Sentiment Distribution') + 
  xlab('Sentiment(0=neg, 1=pos)') +
  ylab("Count")

# Feature Selection

# Cleaning the texts

cleantext = function(dataset){
  corpus = VCorpus(VectorSource(dataset$text))
  corpus = tm_map(corpus, content_transformer(tolower))
  corpus = tm_map(corpus, removeNumbers)
  corpus = tm_map(corpus, removePunctuation)
  corpus = tm_map(corpus, removeWords, stopwords())
  corpus = tm_map(corpus, stemDocument)
  corpus = tm_map(corpus, stripWhitespace)
  corpus
  return(corpus)
}


# Creating the Bag of Words model
bagofwords = function(corpus){
  dtm = DocumentTermMatrix(corpus)
  dtm = removeSparseTerms(dtm, 0.99)
  dtm
  return(dtm)
}

## Create wordcould

pos_text = cleantext(pos)
pos_words = bagofwords(pos_text)

m = as.matrix(pos_words)
v = sort(colSums(m),decreasing=TRUE)
d = data.frame(word = names(v),freq=v)
head(d, 10)

set.seed(123)
wordcloud(words = d$word, freq = d$freq, scale=c(2,0.2),
          min.freq=sort(d$freq, decreasing=TRUE)[[400]],
          colors=brewer.pal(8, "Dark2"),
          random.order=FALSE) 

neg_text = cleantext(neg)
neg_words = bagofwords(neg_text)

m = as.matrix(neg_words)
v = sort(colSums(m),decreasing=TRUE)
d = data.frame(word = names(v),freq=v)
head(d, 10)

set.seed(123)
wordcloud(words = d$word, freq = d$freq, scale=c(2,0.2),
          min.freq=sort(d$freq, decreasing=TRUE)[[400]],
          colors=brewer.pal(8, "Dark2"),
          random.order=FALSE) 

## focus on sentiment column and remove stars column
reviews_sent = subset(reviews_sent, select = c("text", "sentiment"))

## clean text and feature selection
text = cleantext(reviews_sent)
words = bagofwords(text)

dataset = as.data.frame(as.matrix(words))
dataset$Sentiment = reviews_sent$sentiment
dim(dataset)

# Fit the Model

## split into train set and test set
set.seed(123)
split = sample.split(dataset$Sentiment, SplitRatio = 0.5)
train = subset(dataset, split == TRUE)
test = subset(dataset, split == FALSE)

## Fit classification method

### Random forest

# Fitting Random Forest Classification to the Training set
classifier = randomForest(x = train[-989],
y = train$Sentiment,
ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test[-989])

# Making the Confusion Matrix
cm = table(y_pred, y_test=test[, 989])
cm

accuracy = (1423+1364)/3484
precision = 1364 / (1364 + 319)
recall = 1364 / (1364 + 379)
accuracy
precision
recall


### SVM
classifier = svm(Sentiment ~ . , data=train, kernel="linear", cost=0.1, scale = FALSE)
summary(classifier)
y_pred = predict(classifier, newdata = test[-989])
cm = table(y_pred, y_test = test[, 989])
cm

accuracy = (1493+1479)/3484
precision = 1479 / (1479 + 249)
recall = 1479 / (1479 + 263)
accuracy
precision

## ROC Curve
pred1 = prediction(as.numeric(y_pred), as.numeric(test[, 989]))
perf1 <- performance(pred1,"tpr","fpr")
plot(perf1)


# Multiclass classification

## clean text and feature selection
summary(reviews)

text = cleantext(reviews)
words = bagofwords(text)

dataset = as.data.frame(as.matrix(words))
dataset$Stars = reviews$stars

# Fit the Model

## split into train set and test set
set.seed(123)
split1 = sample.split(dataset$Stars, SplitRatio = 0.5)
train = subset(dataset, split1 == TRUE)
test = subset(dataset, split1 == FALSE)

## Fit classification method

### Random Forest

# Fitting Random Forest Classification to the Training set
classifier = randomForest(x = train[-993],
y = train$Stars,
ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test[-993])

# Making the Confusion Matrix
cm = table(y_test=test[, 993], y_pred)
cm

confusion = as.data.frame(as.table(cm))

# Confusion Matrix Viz
ggplot(confusion, aes(x=y_pred, y=y_test, fill=Freq)) + 
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette="Greens", direction=1) +
  ggtitle("Confusion Matirx (Random Forest)")

accuracy = (526+226+269+237+486)/4355
accuracy

mse = sum((as.numeric(y_pred) - as.numeric(test[, 993]))**2)/length(y_pred)
mse

### Support Vector Machine
classifier = svm(Stars ~ . , data=train, kernel="linear", scale=FALSE, cost=0.01)
summary(classifier)
y_pred = predict(classifier, newdata = test[-993])
cm = table(y_pred, test[, 993])
cm

# Confusion Matrix Viz
confusion = as.data.frame(as.table(cm))

ggplot(confusion, aes(x=y_test, y=y_pred, fill=Freq)) + 
  geom_tile() + theme_bw() + coord_equal() +
  scale_fill_distiller(palette="Reds", direction=1) +
  ggtitle("Confusion Matirx (SVM)")

accuracy = (633+315+297+306+604)/4355
accuracy


mse = sum((as.numeric(y_pred) - as.numeric(test[, 993]))**2)/length(y_pred)
mse

