import re


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