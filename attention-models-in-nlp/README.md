# Natural Language Processing with Attention Models

# Week 1: Neural Machine Translation

## Outline

- Introduction to Neural Machine Translation
- Seq2Seq model and its shortcomings
- Solution for the information bottleneck

## Neural Machine Translation

In NMT we used an encoder and a decoder. For example to translate from English to French.

## Seq2Seq model

- Introduced by Google in 2014
- Maps variable-length sequences to fixed-length memory
- Inputs and outputs can have different lengths
- LSTMs and GRUs to avoid vanishing and exploding gradient problems

![Untitled](images/Untitled.png)

The encoder and decoder architectures are illustrated below

![Untitled](images/Untitled%201.png)

![Untitled](images/Untitled%202.png)

One of the problems with this architecture is the Information bottleneck. Because only a fixed amount of information goes from the encoder to the decoder.

![Untitled](images/Untitled%203.png)

The power of Se2Seq of variables length and fixed length memory become a bottleneck for large sentences. In other words, as sequence size increases, model performance decreases.

- One idea is to use all encoder hidden states.

The idea of using all hidden states can be achieve by using a better hidden layers between the encoder and decoder. This solution is ATTENTION. The model can focus on specific hidden states at every step.

![Untitled](images/Untitled%204.png)

## Attention

One of the original papers about Attention was introduce on **Neural Machine Translation by Jointly Learning to Align and Translate** *by Dzmitry bahdanau, KyungHyun Cho and yoshua bengio***.**

Paper → [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)

The below graph shows a huge difference between Traditional Seq2Seq and Seq2Seq with Attention

![Untitled](images/Untitled%205.png)

## How to use all the hidden states?

In order to use all hidden states from the encoder in an efficient way, we can set weights depending on the previous hidden state in the decoder. This is the core idea behind attention

![Untitled](images/Untitled%206.png)

## The attention layer in more depth

![Untitled](images/Untitled%207.png)

In order to find similar key for the given query, we use scaled dot-product attention

![Untitled](images/Untitled%208.png)

1. MatMul: Get Similarity Between Q an K 
2. Scale: Scale using the root of the key vector size
3. SoftMax: Weights for the weighted sum
4. MatMul: Weighted sum of values V

In the core, attention is the multiplication of two matrixes and a Softmax.

## Machine translation setup

- Use pre-trained vector embeddings
- Otherwise, initially represent words with one-hot vectors
- Keep track of index mappings with word2ind and ind2word dictionaries
- Add end of sequence tokens: <EOS>
- Pad the token vectors with zeros

![Untitled](images/Untitled%209.png)

![Untitled](images/Untitled%2010.png)

### Teacher Forcing

One problem with sequential predictions is that one error may be propagated on further steps, at the end of the prediction the meaning may be extremely different. To avoid this problem, we can use teacher forcing which means we can pass the right label in each state even if the model predicted wrong some cell. This helps to model learned better.

![Untitled](images/Untitled%2011.png)

## NMT Model with Attention

![Untitled](images/Untitled%2012.png)

In more details we have the following architecture for NMT

![Untitled](images/Untitled%2013.png)

### BLEU Score

BLUE stands for **B**i**L**ingual **E**valuation **U**nderstudy. It evaluates the quality of machine translate by compares candidate translations to reference (human)  translations. The closer to 1 the better

![Untitled](images/Untitled%2014.png)

BLUE Score also have some problems

- BLUE doesn’t consider semantic meaning
- BLUE doesn’t consider sentence structure: “Ate I was hungry because”

### ROUGE Score

**R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation

Compares candidates with reference (human) translations

![Untitled](images/Untitled%2015.png)

### ROUGE-N, BLUE and F1 score

![Untitled](images/Untitled%2016.png)

# Week 2: Text Summarization

One of the biggest problems with traditional Seq2Seq Architectures is that each step need the previous step. Hence, We have T sequential steps. In addition, there a loss of information and vanishing gradient problem.

![Untitled](images/Untitled%2017.png)

[https://www.notion.so](https://www.notion.so)

We saw before that Attention help with the problems of loss information and vanishing gradient. 

![Untitled](images/Untitled%2018.png)

However, we want to improve the parallels on the architecture. Hence we introduce Transformers because  **Attention is all you need**

## Scaled Dot-Product Attention

Transformer use Scaled Dot-Product Attention

![Untitled](images/Untitled%2019.png)

In general we use multi-head attention which can run in parallel =)

![Untitled](images/Untitled%2020.png)

The transformer Encoder

![Untitled](images/Untitled%2021.png)

The decoder

![Untitled](images/Untitled%2022.png)

## RNNs vs Transformer: Positional Encoding

![Untitled](images/Untitled%2023.png)

Notice how the Transformer add values into the embedding to priority relevant word respect to current one.

![Untitled](images/Untitled%2024.png)

## Transformer Applications

- Text Summarization
- Auto-Complete
- named Entity Recognition (NER)
- Question answering (Q&A)
- Translation
- Chat-bots
- Other NLP Tasks
    - Sentiment Analysis
    - Market Intelligence
    - Text Classification
    - Character Recognition
    - Spell Checking

## T5: Text-to-Text Transfer Transformer

With transformer we can train a single model and use that model for more specialize cases like Translation, Classification, Q&A.

[https://t5-trivia.glitch.me/](https://t5-trivia.glitch.me/)

![Untitled](images/Untitled%2025.png)

## Understanding Attention

- Queries: From one sentence
- Keys and Values from another

![Untitled](images/Untitled%2026.png)

![Untitled](images/Untitled%2027.png)

![Untitled](images/Untitled%2028.png)

![Untitled](images/Untitled%2029.png)

Looking at Query and the key we have

![Untitled](images/Untitled%2030.png)

Looking at the math behind attention

![Untitled](images/Untitled%2031.png)

![Untitled](images/Untitled%2032.png)

## Masked Self-Attention

In self-attention the queries, keys and values came from the same sentence. Hence, you get the attention within the sentence. 

![Untitled](images/Untitled%2033.png)

In Masked Self-Attention, queries, keys and values come from the same sentence. Queries do not attention to future positions. 

![Untitled](images/Untitled%2034.png)

Mathematically we mask the attention by adding a mask matrix inside the SoftMax.

![Untitled](images/Untitled%2035.png)

![Untitled](images/Untitled%2036.png)

## Multi-head Attention

![Untitled](images/Untitled%2037.png)

![Untitled](images/Untitled%2038.png)

With a little more details. Multi-Head Attention works as follows

![Untitled](images/Untitled%2039.png)

## Transformer Decoder

![Untitled](images/Untitled%2040.png)

### Explanation with a sentence

![Untitled](images/Untitled%2041.png)

![Untitled](images/Untitled%2042.png)

![Untitled](images/Untitled%2043.png)

![Untitled](images/Untitled%2044.png)

## Final Example a summarization

![Untitled](images/Untitled%2045.png)

Lets preprocess the data

![Untitled](images/Untitled%2046.png)

Then we check the Cross entropy Loss

![Untitled](images/Untitled%2047.png)

Now we can train our transformer summarizer

The inference will look like

![Untitled](images/Untitled%2048.png)

# Week 3: Question Answering & Hugging Face

- Context-Based Q&A
- Closed Book Q&A

## General Purpose Learning

Use a model pre- trained and use it to performance a task you want. Assume you used a pretrain model CBOW

![Untitled](images/Untitled%2049.png)

There are three main advantages to transfer learning:

- Reduce training time
- Improve predictions
- Allows you to use smaller datasets

![Untitled](images/Untitled%2050.png)

![Untitled](images/Untitled%2051.png)

## Examples of different architecture

![Untitled](images/Untitled%2052.png)

## Hugging Face 🤗

[https://huggingface.co/models](https://huggingface.co/models)

# References

## [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) (Raffel et al, 2019)

### [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) (Kitaev et al, 2020)

### [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al, 2017)

### [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf) (Peters et al, 2018)

### [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) (Alammar, 2018)

### [The Illustrated GPT-2 (Visualizing Transformer Language Models)](http://jalammar.github.io/illustrated-gpt2/) (Alammar, 2019)

### [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (Devlin et al, 2018)

### [How GPT3 Works - Visualizations and Animations](http://jalammar.github.io/how-gpt3-works-visualizations-animations/) (Alammar, 2020)