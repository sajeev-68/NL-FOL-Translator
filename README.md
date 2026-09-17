# NL-to-FOL: Natural Language to First-Order Logic Translation

An encoder-decoder transformer built from scratch in pure PyTorch that translates natural language statements into first-order logic (FOL).

## Overview

This project explores whether a compact, from-scratch transformer can learn to map natural language sentences to their formal first-order logic representations. The architecture uses [nanoGPT](https://github.com/karpathy/nanoGPT) and "Attenion is all you need" as a reference foundation, implemented without high-level training frameworks.

## Approach

- **Architecture:** Encoder-decoder transformer, implemented from scratch in PyTorch
- **Tokenization:** Dual `bert-base-uncased` tokenizers — one for natural language input, one for FOL output — with FOL logical symbols mapped to reserved token slots to keep the target vocabulary compact and well-formed
- **Datasets:** [FOLIO](https://arxiv.org/abs/2209.00840), Willow, and MALLS, plus a custom-assembled version of P-FOLIO built from a raw Hugging Face export (cleaned merged-cell blanks, inconsistent truth labels, and dual derivation columns into ~1,436 usable proof blocks)

## Status

Actively in development. Results so far are qualitative; quantitative evaluation (accuracy, exact-match, BLEU) is planned but not yet complete.

## Notes

Along the way this involved debugging padding-mask issues in self- and cross-attention, resolving tokenizer/config mismatches across the dual vocabularies, and implementing a custom block-sparse MLP attention variant.

