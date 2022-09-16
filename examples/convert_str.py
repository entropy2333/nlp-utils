from icecream import ic
from nlp_utils.data_utils import (convert_full2half_width,
                                  convert_half2full_width,
                                  get_full_width_characters,
                                  get_half_width_characters)

ic(convert_half2full_width("".join(get_half_width_characters())))
ic(convert_full2half_width("".join(get_full_width_characters())))
