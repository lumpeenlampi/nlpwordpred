#
# This is the NLP Word Predictor Application developed by Lumpeenlampi for the
# Coursera Capstone project. It deploys prediction based on 5-grams and includes
# a model of approx 150 MB.
# 
#

library(shiny)
library(tm)
library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)
library(caret)
library(kernlab)
library(data.table)
library(RWeka)
library(wordcloud)

options(shiny.maxRequestSize=300*1024^2)

#inputdir <- "../tdmall3/en_US/"
inputdir <- ""

readTdm <- function(dir, filenm)
{
  fnm <- paste(dir, filenm, sep="")
  tdm <- fread(fnm)
  return(tdm)
}

tokenize <- function(t) {
  return(Boost_tokenizer(t))
}

tdt2all <- readTdm(inputdir, "en_US.all_tdm_scores_redalt_final_1.csv")
tdt3all <- readTdm(inputdir, "en_US.all_tdm_scores_redalt_final_2.csv")
tdt4all <- readTdm(inputdir, "en_US.all_tdm_scores_redalt_final_3.csv")
tdt5all <- readTdm(inputdir, "en_US.all_tdm_scores_redalt_final_4.csv")

setkey(tdt2all, word1)
setkey(tdt3all, word2, word1)
setkey(tdt4all, word3, word2, word1)
setkey(tdt5all, word4, word3, word2, word1)

predictwordn <- function(wordstr, n=5)
{
  nextw <- ""
  ans <- data.table()
  words <- tokenize(removePunctuation(tolower(wordstr)))
  lenwords <- length(words)
  matchl <- data.table()
  if(lenwords >= 4)
  {
    if(n>=5)
      matchl <- tdt5all[.(words[lenwords-3], words[lenwords-2], words[lenwords-1], words[lenwords])]
    if(nrow(matchl)>0)
    {
      matchl <- cbind(matchl, match = rep(4, nrow(matchl)))
      matchl <- matchl[,term:=paste(word4, word3, word2, word1, nextword)]
      matchl <- matchl[, .(nextword, count, score, match, term)]
      ans <- matchl[!is.na(matchl$nextword),]
      #      return(c(matchl[1, "nextword"], "4", matchl[1, "freq"]))
    }
  }
  if(lenwords >= 3)
  {
    if(n>=4)
      matchl <- tdt4all[.(words[lenwords-2], words[lenwords-1], words[lenwords])]
    if(nrow(matchl)>0)
    {
      matchl <- cbind(matchl, match = rep(3, nrow(matchl)))
      matchl <- matchl[,term:=paste(word3, word2, word1, nextword)]
      matchl <- matchl[, .(nextword, count, score, match, term)]
      if(nrow(matchl) > 0)
        for(i in 1:nrow(matchl))
          if(!is.na(matchl[i, "nextword"]) && ((nrow(ans)==0) || !any(ans$nextword == as.character(matchl[i, "nextword"]))))
            ans <- rbind(ans, matchl[i, ])
      #      return(c(matchl[1, "nextword"], "3", matchl[1, "freq"]))
    }
  }
  if(lenwords >= 2)
  {
    if(n>=3)
      matchl <- tdt3all[.(words[lenwords-1], words[lenwords])]
    if(nrow(matchl)>0)
    {
      matchl <- cbind(matchl, match = rep(2, nrow(matchl)))
      matchl <- matchl[,term:=paste(word2, word1, nextword)]
      matchl <- matchl[, .(nextword, count, score, match, term)]
      if(nrow(matchl) > 0)
        for(i in 1:nrow(matchl))
          if(!is.na(matchl[i, "nextword"]) && ((nrow(ans)==0) || !any(ans$nextword == as.character(matchl[i, "nextword"]))))
            ans <- rbind(ans, matchl[i, ])
      #      return(c(matchl[1, "nextword"], "2", matchl[1, "freq"]))
    }
  }
  matchl <- tdt2all[.(words[lenwords])]
  if(nrow(matchl)>0)
  {
    matchl <- cbind(matchl, match = rep(1, nrow(matchl)))
    matchl <- matchl[,term:=paste(word1, nextword)]
    matchl <- matchl[, .(nextword, count, score, match, term)]
    if(nrow(matchl) > 0)
      for(i in 1:nrow(matchl))
        if(!is.na(matchl[i, "nextword"]) && ((nrow(ans)==0) || !any(ans$nextword == as.character(matchl[i, "nextword"]))))
          ans <- rbind(ans, matchl[i, ])
    #    return(c(matchl[1, "nextword"], "1", matchl[1, "freq"]))
  }
  if(nrow(ans)>0)
    ans <- data.table(arrange(ans, desc(score)))
  return(ans)
}

predictword <- function(x) predictwordn(x, 5)





# Define UI for application that draws a histogram
ui <- fluidPage(
   
   # Application title
   titlePanel("NLP Word Prediction Tool"),
   
   # Sidebar with a slider input for number of bins 
   sidebarLayout(
      sidebarPanel(
         h3("Predicting words based on NGrams"),
         textInput("text",
                     "Give text:",
                      value="",
                      width="1500px"),
         sliderInput("ngrams", "Max NGram", min=2, max=5, value=5),
         submitButton("Predict Next Word")
         
      ),
      
      mainPanel(
        tabsetPanel(type="tabs",
                  tabPanel("Word Prediction", br(),
                           htmlOutput("textout"),
                           plotOutput("myPlot")),
                  tabPanel("Instructions", br(), 
                           h1("Instructions for the use of the Word Prediction Application"),
                           p("This application predicts a word based on a sequence of words provided in the input. 
                             The prediction is done based on N-Grams obtained from training texts.
                             The maximum NGram length to be used can be given. 
                             Any extra words in the input line will be ignored."),
                           h3("Use"),
                           p("Write part of a sentence in the text input box and press the Predict word -button"),
                           p("You can select the maximum NGram length used for prediction with the slider"),
                           p("The wordcloud shows a graphical representation of the scored output"),
                           p("The list of words shows the predicted word in descending order of likelihood.
                            For each word, it gives the score as well as the count in the training set and the NGram level used for prediction.")
                  )
        )
      # Show a plot of the generated distribution
#      mainPanel(
#        htmlOutput("textout")
      )
   )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
   
   output$textout <- renderUI({
     print(input$text)
     print(as.numeric(input$ngrams))
     ans <- predictwordn(input$text, as.numeric(input$ngrams))
     str <- list()
     n <- nrow(ans)
     if(n>10) n<- 10
     if(n>0)
     {
       for(i in 1:n)
       {
         if(!is.na(ans[i, "nextword"]))
         str <- append(str, paste(ans[i, "nextword"], "     (", substr(as.character(ans[i, "score"]), 1, 5), ", ", as.character(ans[i, "match"]+1), ", ", as.character(ans[i, "count"]), ")", sep=""))
       }
     }
#     print(n)
#     print(ans)
#     print(str)
     l <- lapply(str, tags$p)
     l <- append(list(h3("Predicted word (score, ngram, count)")), l)
     l
   })
   
   wordcloud_rep <- repeatable(wordcloud)
   
   output$myPlot <- renderPlot({
     ans <- predictwordn(input$text, as.numeric(input$ngrams))
     if(nrow(ans)>0)
      wordcloud_rep(words = ans$nextword, freq = ans$score*10000, min.freq = 1,
                   max.words=20, random.order=FALSE, rot.per=0.35, 
                   colors=brewer.pal(8, "Dark2"), scale=c(7,2))
   })
}

# Run the application 
shinyApp(ui = ui, server = server)

