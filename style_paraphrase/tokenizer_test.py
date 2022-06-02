import argparse
import sys
import torch

from inference_utils import GPT2Generator


parser = argparse.ArgumentParser()
para = "/root/workspace/style-transformer/style-transfer-paraphrase/style_paraphrase/saved_models/test_paraphrase/checkpoint-7305"
#shakespeare = "/root/workspace/style-transformer/style-transfer-paraphrase/style_paraphrase/saved_models/model_shakespeare_1/checkpoint-1377"
shakespeare = "/root/workspace/style-transformer/style-transfer-paraphrase/style_paraphrase/saved_models/model_300"
parser.add_argument('--model_dir0', default=para, type=str)
parser.add_argument('--model_dir1', default=shakespeare, type=str)

args = parser.parse_args()

print("Loading paraphraser...")
paraphraser = GPT2Generator(args.model_dir0, upper_length="same_5")
paraphraser_inverse = GPT2Generator(args.model_dir1, upper_length="same_5")

input_sentence = input("Enter your sentence, q to quit: ")

while input_sentence != "q" and input_sentence != "quit" and input_sentence != "exit":
    paraphraser.modify_p(top_p=0.0)
    decode_from_para = paraphraser.generate(input_sentence)
    print("\nf_para(X):\n{}\n".format(decode_from_para))
    print("tokenized version: {}\n".format(paraphraser.tokenizer.tokenize(input_sentence)))
    decode_from_inv = paraphraser_inverse.generate(input_sentence)
    print("\nf_inv(X):\n{}\n".format(decode_from_para))
    print("tokenized version: {}\n".format(paraphraser_inverse.tokenizer.tokenize(input_sentence)))
    print(paraphraser.generate(input_sentence))

    print("\nf_inv(f_para(X)):\n{}\n".format(paraphraser_inverse.generate(decode_from_para)))
    input_sentence = input("Enter your sentence, q to quit: ")
