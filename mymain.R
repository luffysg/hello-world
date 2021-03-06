
myvocab <- scan(file = "myvocab.txt", what = character())
train <- read.table("train.tsv", stringsAsFactors = FALSE,
                    header = TRUE)

train$review <- gsub('<.*?>', ' ', train$review)
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab, 
                                                ngram = c(1L, 2L)))
dtm_train = create_dtm(it_train, vectorizer)

mylogit.cv = cv.glmnet(x = dtm_train, 
                       y = train$sentiment, 
                       alpha = 0.6,
                       family='binomial', 
                       type.measure = "auc")
mylogit.fit = glmnet(x = dtm_train, 
                     y = train$sentiment, 
                     alpha = 0.6,
                     lambda = mylogit.cv$lambda.min, 
                     family='binomial')

test = read.table("test.tsv",
                  stringsAsFactors = FALSE,
                  header = TRUE)
test$review <- gsub('<.*?>', ' ', test$review)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
dtm_test = create_dtm(it_test, vectorizer)
mypred = predict(mylogit.fit, dtm_test, type = "response")
output = data.frame(id = test$id, prob = as.vector(mypred))
write.table(output, file = "mysubmission.txt", 
            row.names = FALSE, sep='\t')

