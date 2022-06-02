import argparse
import logging
import sys
import torch

from inference_utils import GPT2Generator

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir0', default="../paraphraser_gpt2_large", type=str)
parser.add_argument('--model_dir1', default="../shakespeare_models/model_305", type=str)
# parser.add_argument('--model_dir0', default="saved_models/test_paraphrase/checkpoint-7305", type=str)
# parser.add_argument('--model_dir1', default="saved_models/model_shakespeare_1/checkpoint-1000", type=str)
parser.add_argument('--top_p_value', default=0.6, type=float)
args = parser.parse_args()

if not torch.cuda.is_available():
    print("Please check if a GPU is available or your Pytorch installation is correct.")
    sys.exit()

print("Loading paraphraser...")
paraphraser = GPT2Generator(args.model_dir0, upper_length="same_5")
parapharser_inverse = GPT2Generator(args.model_dir1, upper_length="same_5", data_dir="../datasets/shakespeare")

print("\n\nNOTE: Ignore the weight mismatch error, this is due to different huggingface/transformer versions + minor modifications I did myself, shouldn't affect the paraphrases.\n\n")

input_sentence = input("Enter your sentence, q to quit: ")

while input_sentence != "q" and input_sentence != "quit" and input_sentence != "exit":
    paraphraser.modify_p(top_p=0.0)
    greedy_decoding = paraphraser.generate(input_sentence)
    print("\nintermediate paraphase:\n{}\n".format(greedy_decoding))
    greedy_decoding = paraphraser.generate(greedy_decoding)
    print("\ngreedy sample:\n{}\n".format(greedy_decoding))
    paraphraser.modify_p(top_p=args.top_p_value)
    top_p_60_samples, _ = paraphraser.generate_batch([greedy_decoding, greedy_decoding, greedy_decoding])
    top_p_60_samples = "\n".join(top_p_60_samples)
    print("top_p = {:.2f} samples:\n{}\n".format(args.top_p_value, top_p_60_samples))
    input_sentence = input("Enter your sentence, q to quit: ")

print("Exiting...")
