"""
Reference: https://github.com/jiesutd/LatticeLSTM/blob/master/utils/data.py
"""
import numpy as np


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert embedd_dim + 1 == len(tokens)
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim


class CharEmbedding:
    def __init__(self, pretrain_path="gigaword_chn.all.a2b.uni.ite50.vec") -> None:
        self.pretrain_path = pretrain_path
        embed_dict, embed_dim = load_pretrain_emb(self.pretrain_path)
        self.embed_dict = embed_dict
        self.embed_dim = embed_dim
        self.scale = np.sqrt(3.0 / embed_dim)
        print(
            f"load pretrain embedding from {self.pretrain_path}, size: {len(self.embed_dict)}, dim: {self.embed_dim}"
        )

    def get_char_embedding(self, word, norm=True):
        if word in self.embed_dict:
            char_embed = self.embed_dict[word]
            if norm:
                char_embed = norm2one(char_embed)
        elif word.lower() in self.embed_dict:
            char_embed = self.embed_dict[word.lower()]
            if norm:
                char_embed = norm2one(char_embed)
        else:
            char_embed = np.random.uniform(-self.scale, self.scale, [1, self.embed_dim])
        return char_embed


if __name__ == "__main__":
    pretrain_char_emb_path = "./gigaword_chn.all.a2b.uni.ite50.vec"
    char_emb = CharEmbedding(pretrain_char_emb_path)
    print(char_emb.get_char_embedding("中"))
