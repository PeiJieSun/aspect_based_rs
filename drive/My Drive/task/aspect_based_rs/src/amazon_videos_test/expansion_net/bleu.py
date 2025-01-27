'''
Example:
hyp = [[1, 2, 3, 4, 5], [1, 6, 4, 5, 8], [8, 7, 5, 4, 5]]
ref = [[[1, 2, 3, 4, 5], [1, 6, 4, 5, 8], [8, 7, 5, 4, 5]]]
compute_bleu(hyp, [ref])
'''

"""
Borrowed from https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
"""

import collections
import math


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(translation_corpus, reference_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for order in range(1, max_order + 1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order - 1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  ratio = float(translation_length) / reference_length
  if ratio > 1.0:
    bp = 1.
  elif ratio == 0.:
    bp = 0.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu_scores = {}
  for order in range(1, max_order + 1):
    if min(precisions[:order]) > 0:
      p_log_sum = sum((1. / order) * math.log(p) for p in precisions[:order])
      geo_mean = math.exp(p_log_sum)
    else:
      geo_mean = 0
    bleu = geo_mean * bp
    bleu_scores[order] = bleu

  #print(bleu_scores)
  return bleu_scores