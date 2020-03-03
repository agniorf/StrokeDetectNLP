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
# library(splitstackshape)

# Clean a string and return one with letters and punct
clean_string <- function(mystring) {
  #Remove the /r character.
  mystring <- gsub('\r',' ',mystring)
  mystring <- gsub('\f',' ',mystring)
  #Make everything lowercase
  mystring<-tolower(mystring)
  #Remove all numbers
  mystring<-gsub('[[:digit:]]+', '', mystring)
  mystring <- removeNumbers(mystring)
  #Remove parantheses 
  #mystring <- gsub('\\(','',mystring)
  #mystring <- gsub('\\)','',mystring)
  
  #Remove all the non alphanumeric characters except punctuation
  mystring<-gsub('[^[:alnum:][:punct:] ]+','',mystring)
  
  #   #Remove all characters that are only 1 length long from the string 
  #   # to avoid introductions of images figures etc.
  #   mystring <- gsub(" *\\b(?<!-)\\w{1}(?!-)\\b *", " ", mystring, perl=T)
  
  # Replace more than one space or trailing space with one space
  mystring<- gsub("\\s+", " ", str_trim(mystring))
  return(mystring)
}

# Read word groupings
word_units = read.xls("wordgroups.xlsx", sheet = 1, header = TRUE)
word_units = as.character(word_units[,1])

# Window size
window_size = 5L  # Window for context words
delimit_sentence = TRUE  # Whether we limit the window to be within a sentence 

# Read the whole corpus into 1 string
myString<-" "
separator <- paste(replicate(window_size, " ~ "), collapse = "")

# Read each source: 
# sources <- c("Yousem", "StrokeXs", "UpToDate", "Rad2010")
sources <- c("StrokeXs")
corpus_txt_name = paste(paste(sources, collapse='_'), window_size,
                        paste0(delimit_sentence,".txt"), sep = "_")
if (length(sources) > 1){
  for (data_source in sources){
    singlesource <- paste(data_source, window_size,
                          paste0(delimit_sentence,".txt"), sep = "_")
    myString <- paste(myString, read_file(singlesource), sep=separator)
  }
} else{
  for (data_source in sources){
    datadir = paste("~/Dropbox (Partners HealthCare)/MVP/RPDR/GloVe/Training Radiology and Stroke Resources/", data_source, sep="")
    setwd(datadir)  
    # setwd(data_source)  # On cluster
    files <- list.files(pattern = ".txt$")
    print(paste("Reading", data_source, "..."))
    
    for (i in 1:length(files)) {
      filetext <- read_file(files[i])
      # Replace word groups
      for (i in 1:length(word_units)) {
        filetext <- gsub(word_units[i], paste(strsplit(word_units[i]," ")[[1]], collapse = ""), 
                         clean_string(filetext))
      }
      # if sentence delimiting
      if (delimit_sentence){
        sentences <- unlist(strsplit(filetext, "[.?!]\\s"))
        for (j in 1:length(sentences)) {
          sent <- gsub('[^[:alnum:] ]+', '', sentences[j])
          myString<- paste(myString, sent, sep=separator)
        }
      }
      else {
        myString<-paste(myString, gsub('[^[:alnum:] ]+', '', filetext), sep=separator)
      }
      # myString <- tm_map(myString, removeWords, c(stopwords("english")))
    }
    # setwd("../")  # On cluster
  }
}

# wikipedia
# setwd("Wikipedia")  # On cluster
# setwd("../Wikipedia")  # On local
# wiki <- file("20pageviews.csv", open = "r")
# while (length(oneLine <- readLines(wiki, n = 1)) > 0) {
#   myLine <- str_split_fixed(oneLine, pattern = ',"', n = 2)[2] # first col is wiki ID
#   # Replace word groups
#   for (i in 1:length(word_units)) {
#     myLine <- gsub(word_units[i], paste(strsplit(word_units[i]," ")[[1]], collapse = ""), 
#                    clean_string(myLine))
#   }
#   if (delimit_sentence){
#     sentences <- unlist(strsplit(myLine, "[.?!]\\s"))
#     for (j in 1:length(sentences)) {
#       sent <- gsub('[^[:alnum:] ]+', '', sentences[j])
#       myString<- paste(myString, sent, sep=separator)
#     }
#   }else {
#     myString<-paste(myString, gsub('[^[:alnum:] ]+', '', myLine), sep=separator)
#   }
# }
# # setwd("../")   # On cluster
# corpus_txt_name <- paste0(corpus_txt_name, '_Wiki')
# close(wiki)

# Output to a total file
setwd("../")  # On local
output <- file(corpus_txt_name)
writeLines(myString, output)
close(output)
