import re
import string
from typing import List, Dict
from functools import partial
from itertools import chain


def get_pattern_counter(values, patterns, print_others=False):
    custom_counter = {}
    cnt = 0
    for value in values:
        flag = False
        for pattern in patterns:
            if re.match(pattern, value):
                if pattern not in custom_counter:
                    custom_counter[pattern] = 0
                custom_counter[pattern] += 1
                flag = True
                break
        if not flag:
            if '其他' not in custom_counter:
                custom_counter['其他'] = 0
            if print_others and cnt < 10:
                print(value)
                cnt += 1
            custom_counter['其他'] += 1
    assert sum(custom_counter.values()) == len(values)
    return custom_counter


def full_width_to_half_width(text):
    """
    Convert full-width characters to half-width.
    """
    full_width_characters = list(range(0xFF01, 0xFF5E + 1))
    half_width_characters = list(range(0x0021, 0x007E + 1))
    full_half_map = dict(zip(full_width_characters, half_width_characters))
    return text.translate(full_half_map)


def half_width_to_full_width(text):
    """
    Convert half-width characters to full-width.
    """
    full_width_characters = list(range(0xFF01, 0xFF5E + 1))
    half_width_characters = list(range(0x0021, 0x007E + 1))
    full_half_map = dict(zip(half_width_characters, full_width_characters))
    return text.translate(full_half_map)


def text2ner_label_with_exactly_match(transcripts: List[str],
                                      exactly_entities_label: List[List[str]],
                                      markup_type: str = 'bio') -> List[List[str]]:
    """
    Convert transcripts to iob label using document level tagging match method,
        all transcripts will be concatenated as a sequences.
    Reference: https://github.com/wenwenyu/PICK-pytorch/blob/master/data_utils/documents.py#L229

    Args:
        transcripts: list of transcripts
        exactly_entities_label: dict of entities and their labels

    Returns:
        list of ner label

    Example:
        >>> transcripts = ['他是Dan', 'Mark', '刘德华是中国人']
        >>> exactly_entities_label = [['中国人', '刘德华'], ['外国人', 'DanMark']]
        >>> text2bio_label_with_exactly_match(transcripts, exactly_entities_label)
        [['O', 'O', 'B-外国人', 'I-外国人', 'I-外国人'], ['I-外国人', 'I-外国人', 'I-外国人', 'I-外国人'], ['B-中国人', 'I-中国人', 'I-中国人', 'O', 'O', 'O', 'O']]
        [('中国人', 9, 12), ('外国人', 2, 9)]
    """

    def preprocess_transcripts(transcripts: List[str]):
        """
        preprocess texts into separated word-level list, this is helpful to matching tagging label between source and target label,
        e.g. source: xxxx hello ! world xxxx  target: xxxx hello world xxxx,
        we want to match 'hello ! world' with 'hello world' to decrease the impact of ocr bad result.
        """
        seq, idx = [], []
        for index, x in enumerate(transcripts):
            if x not in string.punctuation and x not in string.whitespace:
                seq.append(x)
                idx.append(index)
        return seq, idx

    concatenated_sequences = []
    sequences_len = []
    for transcript in transcripts:
        concatenated_sequences.extend(list(transcript))
        sequences_len.append(len(transcript))

    result_tags = ['O'] * len(concatenated_sequences)
    matched_entities = []
    for entity_type, entity_value in exactly_entities_label:
        (src_seq, src_idx), (tgt_seq, _) = preprocess_transcripts(concatenated_sequences), preprocess_transcripts(entity_value)
        src_len, tgt_len = len(src_seq), len(tgt_seq)
        if tgt_len == 0:
            continue

        for i in range(src_len - tgt_len + 1):
            if src_seq[i:i + tgt_len] == tgt_seq:
                matched_entities.append((entity_type, i, i + tgt_len))
                if markup_type == 'bio':
                    tag = ['I-{}'.format(entity_type)] * (src_idx[i + tgt_len - 1] - src_idx[i] + 1)
                    tag[0] = 'B-{}'.format(entity_type)
                elif markup_type == 'bmes':
                    if tgt_len == 1:
                        tag = ['S-{}'.format(entity_type)]
                    else:
                        tag = ['M-{}'.format(entity_type)] * (src_idx[i + tgt_len - 1] - src_idx[i] + 1)
                        tag[0] = 'B-{}'.format(entity_type)
                        tag[-1] = 'E-{}'.format(entity_type)
                result_tags[src_idx[i]:src_idx[i + tgt_len - 1] + 1] = tag

    tagged_transcript = []
    start = 0
    for length in sequences_len:
        tagged_transcript.append(result_tags[start:start + length])
        start = start + length
        if start >= len(result_tags):
            break
    return tagged_transcript, matched_entities


text2bio_label_with_exactly_match = partial(text2ner_label_with_exactly_match, markup_type='bio')
text2bmes_label_with_exactly_match = partial(text2ner_label_with_exactly_match, markup_type='bmes')


def flatten_list(lst):
    return list(chain.from_iterable(lst))


def is_chinese(s):
    cnt = 0
    for _char in s:
        if '\u4e00' <= _char <= '\u9fa5':
            cnt += 1
    return cnt >= (len(s) / 2)