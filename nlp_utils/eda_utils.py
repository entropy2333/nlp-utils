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