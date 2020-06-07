## Description:
## Convert the original file json.gz into gz file, which line is composed of 
## user_id, item_id, rating, time_stamp, words_num, reviewtext

import gzip, re

origin_file = '/content/drive/My Drive/datasets/amazon_reviews/reviews_Musical_Instruments_5.json.gz'
target_file = gzip.open('/content/Musical_Instruments.gz', 'w')

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()

g = gzip.open(origin_file, 'r')
for idx, line in enumerate(g):
    line = eval(line)
    reviewText = clean_str(line['reviewText'])
    
    data = ('%s %s %s %s %d %s\n' % (line['reviewerID'], line['asin'], line['overall'], line['unixReviewTime'], len(reviewText.split()), reviewText)).encode()
    target_file.write(data)
    #import sys; sys.exit()