import re
import string
from typing import List, Dict
from functools import partial
from itertools import chain

FULL_WIDTH_CHARACTERS_A_Z = list(range(0xFF21, 0xFF3B))
FULL_WIDTH_CHARACTERS_a_z = list(range(0xFF41, 0xFF5B))
FULL_WIDTH_CHARACTERS_0_9 = list(range(0xFF10, 0xFF1A))
HALF_WIDTH_CHARACTERS_A_Z = list(range(0x0041, 0x005B))
HALF_WIDTH_CHARACTERS_a_z = list(range(0x0061, 0x007B))
HALF_WIDTH_CHARACTERS_0_9 = list(range(0x0030, 0x003A))
FULL_WIDTH_CHARACTERS_TOTAL = list(range(0xFF01, 0xFF5E + 1))
HALF_WIDTH_CHARACTERS_TOTAL = list(range(0x0021, 0x007E + 1))
FULL_WIDTH_CHARACTERS_PUNCTUATION = list(range(0xFF01, 0xFF0F + 1)) + \
                                    list(range(0xFF1A, 0xFF20 + 1)) + \
                                    list(range(0xFF3B, 0xFF40 + 1)) + \
                                    list(range(0xFF5B, 0xFF5E + 1))
HALF_WIDTH_CHARACTERS_PUNCTUATION = list(range(0x0021, 0x002F + 1)) + \
                                    list(range(0x003A, 0x0040 + 1)) + \
                                    list(range(0x005B, 0x0060 + 1)) + \
                                    list(range(0x007B, 0x007E + 1))


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


def get_full_or_width_characters(mode='full',
                                 include_a_z=True,
                                 include_A_Z=True,
                                 include_0_9=True,
                                 include_punctuation=True,
                                 index=False):
    index_list = []
    if include_a_z:
        index_list.extend(FULL_WIDTH_CHARACTERS_a_z if mode ==
                          'full' else HALF_WIDTH_CHARACTERS_a_z)
    if include_A_Z:
        index_list.extend(FULL_WIDTH_CHARACTERS_A_Z if mode ==
                          'full' else HALF_WIDTH_CHARACTERS_A_Z)
    if include_0_9:
        index_list.extend(FULL_WIDTH_CHARACTERS_0_9 if mode ==
                          'full' else HALF_WIDTH_CHARACTERS_0_9)
    if include_punctuation:
        index_list.extend(FULL_WIDTH_CHARACTERS_PUNCTUATION if mode ==
                          'full' else HALF_WIDTH_CHARACTERS_PUNCTUATION)
    if index:
        return index_list
    return [chr(i) for i in index_list]


get_full_width_characters = partial(get_full_or_width_characters, mode='full')
get_half_width_characters = partial(get_full_or_width_characters, mode='half')


def convert_full_and_half_width(
    text,
    mode="full2half",
    include_a_z=True,
    include_A_Z=True,
    include_0_9=True,
    include_punctuation=True,
):
    """
    Convert full-width characters to half-width.
    """
    full_width_characters = get_full_or_width_characters(mode='full',
                                                         include_a_z=include_a_z,
                                                         include_A_Z=include_A_Z,
                                                         include_0_9=include_0_9,
                                                         include_punctuation=include_punctuation,
                                                         index=True)
    half_width_characters = get_full_or_width_characters(mode='half',
                                                         include_a_z=include_a_z,
                                                         include_A_Z=include_A_Z,
                                                         include_0_9=include_0_9,
                                                         include_punctuation=include_punctuation,
                                                         index=True)
    assert len(full_width_characters) == len(half_width_characters), (len(full_width_characters),
                                                                      len(half_width_characters))
    if mode == "half2full":
        full_half_map = dict(zip(half_width_characters, full_width_characters))
    elif mode == "full2half":
        full_half_map = dict(zip(full_width_characters, half_width_characters))
    else:
        raise ValueError("mode must be 'half2full' or 'full2half'")
    return text.translate(full_half_map)


convert_full2half_width = partial(convert_full_and_half_width, mode='full2half')
convert_half2full_width = partial(convert_full_and_half_width, mode='half2full')


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
        (src_seq, src_idx), (tgt_seq, _) = preprocess_transcripts(
            concatenated_sequences), preprocess_transcripts(entity_value)
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
                        tag = ['M-{}'.format(entity_type)
                              ] * (src_idx[i + tgt_len - 1] - src_idx[i] + 1)
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