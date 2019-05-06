from __future__ import print_function, division
import re

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def IOB2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """

    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i-1] == 'O':
            tags[i] = 'B' + tag[1:]
        elif tag[1:] == tags[i-1][1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]
    return tags

# def IOBES(tags):
#     """
#     Check that tags have a valid IOB format.
#     Tags in IOB1 format are converted to IOBES.
#     """
#     for i, tag in enumerate(tags):
#         if tag == 'O':
#             continue
#         if tag.split('-')[0] == 'B':
#             tags[i] = 'S' + tag[1:]
#         elif i == 0:
#             if len(tags) > 1 and tags[1][0] == 'I':
#                 tags[i] = 'B' + tag[1:]
#             else:
#                 tags[i] = 'S' + tag[1:]
#         elif i+1 == len(tags):
#             if tags[i-1][0] != 'I':
#                 tags[i] = 'E' + tag[1:]
#             else:
#                 tags[i] = 'S' + tag[1:]
#         elif tags[i-1] == 'O':
#             tags[i] = 'B' + tag[1:]
#         elif tags[i+1] == 'O':
#             tags[i] = 'E' + tag[1:]
#         else:
#             continue
#     return tags

def IOBES(tags):
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        if tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                continue
            else:
                tags[i] = 'S' + tag[1:]
        elif i == 0 or tags[i - 1] == 'O':
            tags[i] = 'B' + tag[1:]
            if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                continue
            else:
                tags[i] = 'S' + tag[1:]
        elif tag[1:] == tags[i - 1][1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]
            if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                   continue
                else:
                    tags[i] = 'E' + tag[1:]
            else:
                tags[i] = 'S' + tag[1:]
    return tags

# def F1_score(tar_path, pre_paths, tag_dict):
#     right = 0.
#     found = 0.
#     origion = 0.
#     origion = len(tar_path)
#     found = len(pre_paths)
#     recall = 0. if origion == 0 else (right/origion)
#     precision = 0. if found == 0 else (right / found)
#     for pre in pre_paths:
#         if pre in tar_path:
#             right += 1
#     acc = right/found
#     F1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
#     print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(tag, recall, precision, F1))
#     return recall, percision, F1
