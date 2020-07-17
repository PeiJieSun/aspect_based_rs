import numpy as np

data_name = 'amazon_musical_test'
origin_dir = '/content/drive/My Drive/task/aspect_based_rs/data/%s' % data_name
target_dir = '/content/'

org_train_data = open('%s/%s.train.data' % (origin_dir, data_name))
org_val_data = open('%s/%s.val.data' % (origin_dir, data_name))
org_test_data = open('%s/%s.test.data' % (origin_dir, data_name))

tar_train_data = open('%s/index_%s.train.dat' % (target_dir, data_name), 'w')
tar_val_data = open('%s/index_%s.val.dat' % (target_dir, data_name), 'w')
tar_test_data = open('%s/index_%s.test.dat' % (target_dir, data_name), 'w')

final_train_data = open('/content/drive/My Drive/task/A3NCF/python/data/%s.train.dat' % data_name, 'w')
final_val_data = open('/content/drive/My Drive/task/A3NCF/python/data/%s.val.dat' % data_name, 'w')
final_test_data = open('/content/drive/My Drive/task/A3NCF/python/data/%s.test.dat' % data_name, 'w')

abae_vocab_decoder_path = '/content/drive/My Drive/task/aspect_based_rs/data/%s/%s.abae_vocab_decoder.npy' % (data_name, data_name)
abae_vocab_decoder = np.load(abae_vocab_decoder_path, allow_pickle=True).item()

for line in org_train_data:
    line = eval(line)
    user, item, rating, abae_review, g_review =\
        line['user'], line['item'], line['rating'], line['abae_review'], line['g_review']

    write_sentence = ''
    for sentence in abae_review:
        for word_id in sentence:
            write_sentence += '%s ' % abae_vocab_decoder[word_id]
    
    tar_train_data.write('%d\t\t%d\t\t%d\t\t%s\n' % (user, item, rating, write_sentence))
    final_train_data.write('%d\t%d\t%d\n' % (user, item, rating))

for line in org_val_data:
    line = eval(line)
    user, item, rating, abae_review, g_review =\
        line['user'], line['item'], line['rating'], line['abae_review'], line['g_review']

    write_sentence = ''
    for sentence in abae_review:
        for word_id in sentence:
            write_sentence += '%s ' % abae_vocab_decoder[word_id]
    
    tar_val_data.write('%d\t\t%d\t\t%d\t\t%s\n' % (user, item, rating, write_sentence))
    final_val_data.write('%d\t%d\t%d\n' % (user, item, rating))

for line in org_test_data:
    line = eval(line)
    user, item, rating, abae_review, g_review =\
        line['user'], line['item'], line['rating'], line['abae_review'], line['g_review']

    write_sentence = ''
    for sentence in abae_review:
        for word_id in sentence:
            write_sentence += '%s ' % abae_vocab_decoder[word_id]
    
    tar_test_data.write('%d\t\t%d\t\t%d\t\t%s\n' % (user, item, rating, write_sentence))
    final_test_data.write('%d\t%d\t%d\n' % (user, item, rating))