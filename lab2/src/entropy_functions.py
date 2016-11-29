
import itertools
import string
import nltk


def remove_characters(input, chars_to_remove):
    import sys
    if sys.version_info < (3,1) and isinstance(input, str):
        output = input.translate(None, chars_to_remove)
    else:
        # Create a dictionary using a comprehension - this maps every character from
        # string.punctuation to None. Initialize a translation object from it.
        translator = str.maketrans({key: None for key in chars_to_remove})
        output = input.translate(translator)
    return output


def clean_text(input_text, keep_punctuation=False, case_sensitive=False, keep_digits=False):
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # This funtion formats the raw text for extracting the words
    # - delete line breaks
    # - delete punctuation
    # - delete numbers
    # - capitalize
    # The parameter input text is the raw text
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # delete line breaks

    output_text = remove_characters(input_text, "\n")
    # delete punctuation?
    if not keep_punctuation:
        output_text = remove_characters(output_text, string.punctuation)
    # delete digits?
    if not keep_digits:
        output_text = remove_characters(output_text, string.digits)
    # translate from upper to lower case?
    if not case_sensitive:
        output_text = output_text.lower()

    return output_text


def get_tagged_words_from_file(inF):
    "get a list of pairs <word,POS> from a text file"
    with open(inF) as f:
        words = [tuple(word_tag.replace('\n', '').split('\t'))
                 for word_tag in f.readlines()]

    return words


def get_tagged_sents():
    # download universal_tagset model for tagged sentences
    # using nltk.download()
    return nltk.corpus.brown.tagged_sents(categories='news', tagset='universal')


def get_words_from_tagged_sents(tagged_sents):
    # chain returns a sequence of (word, tag)
    # zip(*) split in 2 sequences: words, tags
    words, tags = zip(*itertools.chain(*tagged_sents))
    return words


def clean_tagged_sent(t_sent, **kwargs_cleaning):
    # clean word and surround POS with <> to avoid ambiguity
    t_sent = map(lambda w_t: (clean_text(str(w_t[0]), **kwargs_cleaning), str('<%s>' % (w_t[1],))), t_sent)

    # remove pairs which were cleaned in the previous step
    t_sent = filter(lambda w_t: w_t[0], t_sent)

    return t_sent