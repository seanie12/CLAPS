import collections

from tqdm import tqdm
from transformers import BertTokenizer

from squad_utils import _check_is_max_context, _improve_answer_span


class HarvFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 attention_mask,
                 c_ids,
                 noq_start_position=None,
                 noq_end_position=None,
                 is_impossible=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.c_ids = c_ids

        self.noq_start_position = noq_start_position
        self.noq_end_position = noq_end_position


def convert_examples_to_harv_features(examples, tokenizer, max_seq_length,
                                      max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    features = []
    for example in tqdm(examples, total=len(examples)):
        query_tokens = bert_tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = bert_tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        if len(query_tokens) + len(all_doc_tokens) + 3 > max_seq_length:
            continue

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # for bert input
        noq_start_position = tok_start_position + 1
        noq_end_position = tok_end_position + 1

        context_tokens = []
        context_tokens.append("[CLS]")
        for token in all_doc_tokens:
            context_tokens.append(token)
        context_tokens.append("[SEP]")
        c_ids = bert_tokenizer.convert_tokens_to_ids(context_tokens)

        while len(c_ids) < max_seq_length:
            c_ids.append(0)

        answer_text = "answer: " + example.orig_answer_text
        answer_tokens = tokenizer.tokenize(answer_text)
        start_context = "context: "
        start_context_toks = tokenizer.tokenize(start_context)
        start_context_toks = answer_tokens + start_context_toks

        context = " ".join(example.doc_tokens)
        context_tokens = start_context_toks + tokenizer.tokenize(context)
        input_ids = tokenizer.convert_tokens_to_ids(context_tokens)[
            :max_seq_length]
        attention_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)

        features.append(
            HarvFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                c_ids=c_ids,
                noq_start_position=noq_start_position,
                noq_end_position=noq_end_position
            ))

    return features
