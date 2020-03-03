library(tm)
# library(tokenizers)
library(readr)
library(bigmemory)
library(biganalytics)
library(bigtabulate)
library(ff)
# library(text2vec)
library(stringr)
library(gdata)
library(data.table)
library(rowr)
library(purrr)

#Clean a string and return one with no alphanumerics and ready to be used by glove
clean_string <- function(mystring){
  #Remove the /r character.
  mystring <- gsub('\r','',mystring)
  mystring <- gsub('\f','',mystring)
  #Make everything lowercase
  mystring<-tolower(mystring)
  #Remove all numbers
  mystring<-gsub('[[:digit:]]+', '', mystring)
  mystring <- removeNumbers(mystring)
  #Remove parantheses 
  mystring <- gsub('\\(','',mystring)
  mystring <- gsub('\\)','',mystring)
  mystring<-gsub('[[:punct:] ]+',' ',mystring)
  
  #Remove all non-alphanumeric characters
  mystring<-gsub('[^[:alnum:] ]+',' ',mystring)
  
  #Remove all characters that are only 1 length long from the string to avoid introductions of images figures etc.
  mystring <- gsub(" *\\b(?<!-)\\w{1}(?!-)\\b *", " ", mystring, perl=T)
  return(mystring)
}

# Read word groupings
word_units = read.xls("wordgroups.xlsx", sheet = 1, header = TRUE)
word_units = as.character(word_units[,1])

# Window size
window_size = 5L  # Window for context words
delimit_sentence = TRUE  # Whether we limit the window to be within a sentence 

# Read the corpus so far into 1 string
corpus_txt_name = "Yousem_StrokeXs_UpToDate_Rad2010_5_TRUE.txt"
myString <- read_file(corpus_txt_name)
separator <- paste(replicate(window_size, " ~ "), collapse = "")

wiki <- file("20pageviews.csv", open = "r")
i <- 0
while (length(oneLine <- readLines(wiki, n = 1)) > 0 && i < 100000) {
  myLine <- str_split_fixed(oneLine, pattern = ',"', n = 2)[2] # first col is wiki ID
  # Replace word groups
  for (i in 1:length(word_units)) {
    myLine <- gsub(word_units[i], paste(strsplit(word_units[i]," ")[[1]], collapse = ""),
                   clean_string(myLine))
  }
  if (delimit_sentence){
    sentences <- unlist(strsplit(myLine, "[.?!]\\s"))
    for (j in 1:length(sentences)) {
      sent <- gsub('[^[:alnum:] ]+', '', sentences[j])
      myString<- paste(myString, sent, sep=separator)
    }
  }else {
    myString<-paste(myString, gsub('[^[:alnum:] ]+', '', myLine), sep=separator)
  }
}
setwd("../")   # On cluster
corpus_txt_name <- paste0('Wiki_',corpus_txt_name)
close(wiki)

# Output to a total file
# setwd("../")  # On local
output <- file(corpus_txt_name)
writeLines(myString, output)
close(output)

