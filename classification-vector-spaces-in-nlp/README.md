# Natural Language Processing with Classification and Vector Spaces

# Week 1: Logistic Regression

## Supervised Learning & Logistic Regression

In supervised machine learning, you usually have an input X, which foes into y our prediction function to get your Ŷ**.** You can compare your prediction with the true value of Y. This gives you your cost which you use to update the parameters of theta.

![Untitled](images/Untitled.png)

For example: We can use supervised learning for sentiment analysis to analyze a sense and classify it as positive or negative. 

For logistic regression our Prediction Function is the sigmoid function.

Logistic regression makes use of the sigmoid function which outputs a probability between 0 and 1. The sigmoid function with some weight parameter *θ* and some input *x*(*i*) is defined as follows.

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/oL4Ox_JxTBi-DsfycUwYvw_d0582a0dddf7470486f0955c8b025dd6_Screen-Shot-2020-09-01-at-8.30.00-AM.png?expiry=1712620800000&hmac=luwhcAXIpKaA59PKzf5kTKFyY6eLn9W-ZuLmDvQkEPc](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/oL4Ox_JxTBi-DsfycUwYvw_d0582a0dddf7470486f0955c8b025dd6_Screen-Shot-2020-09-01-at-8.30.00-AM.png?expiry=1712620800000&hmac=luwhcAXIpKaA59PKzf5kTKFyY6eLn9W-ZuLmDvQkEPc)

Illustration of Gradient Decent

![Untitled](images/Untitled%201.png)

After so many iteration the cost should converges into an minimal cost.

![Untitled](images/Untitled%202.png)

![Untitled](images/Untitled%203.png)

After finding the optimal Cost and we finish improving our model. To compute accuracy, you solve the following equation:

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/1NW_9EVkS6yVv_RFZGusFA_90304c432911444eb5f981a4aaa97c47_Screen-Shot-2020-09-02-at-10.53.31-AM.png?expiry=1712620800000&hmac=KFjMVdLUh8LFu0rH2Ds3IrXJNCoU1DAGrwYPr5ZZZe8](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/1NW_9EVkS6yVv_RFZGusFA_90304c432911444eb5f981a4aaa97c47_Screen-Shot-2020-09-02-at-10.53.31-AM.png?expiry=1712620800000&hmac=KFjMVdLUh8LFu0rH2Ds3IrXJNCoU1DAGrwYPr5ZZZe8)

## Sparse representations:

Representations with a lot of zeros. In order words it is 0 almost everywhere. 

One of the main problems with sparse representations are

1. Large training time
2. Large prediction time

![Untitled](images/Untitled%204.png)

## Simple representations of tweets:

tweet = [1, sum(positive words), sum(negative words)]

## Preprocessing

### Elimination of words

**Stop words**: select a list of words consider stop words that will be eliminate  from the sentence.

We can do the Same with Handlers aka tags and  URLS since they usually don’t provide any value into the sentence. 

Similarly we can eliminate punctuation. However, sometime punctuation add value into a sente like exclamation marks !

### Stemming and lowercasing

We can to change the words into their basic form. For instance, tuning can be turn into tun. Similar with all variations of tune like tune, tuned, tunning.

lowercase is self explanatory and all it does is lower case the sentence.

# Week 2: Naive Bayes

Last week we classify tweets using linear regression. The objective of this we will be doing sentiment analysis using Naive Bayes.

To build a classifier, we will first start by creating conditional probabilities given the following table:

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/fwihrJk_RSGIoayZPxUhTQ_3cd35492d526492dbcae2e4baa02bdf4_Screen-Shot-2020-09-08-at-3.38.05-PM.png?expiry=1712620800000&hmac=jUp-lv3KvLasMOqiLwSA5rTOeOgngmCbPmeoQMw-yVE](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/fwihrJk_RSGIoayZPxUhTQ_3cd35492d526492dbcae2e4baa02bdf4_Screen-Shot-2020-09-08-at-3.38.05-PM.png?expiry=1712620800000&hmac=jUp-lv3KvLasMOqiLwSA5rTOeOgngmCbPmeoQMw-yVE)

This allows us compute the following table of probabilities:

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Iiek1X0pSX6npNV9KXl-NQ_2945fa3844184948b0a2590d0683b4f8_Screen-Shot-2020-09-08-at-3.41.46-PM.png?expiry=1712620800000&hmac=KyVRx_3Uvw22MZfO-MUchYVkmOG-jtA2oLfpvVPbqnY](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Iiek1X0pSX6npNV9KXl-NQ_2945fa3844184948b0a2590d0683b4f8_Screen-Shot-2020-09-08-at-3.41.46-PM.png?expiry=1712620800000&hmac=KyVRx_3Uvw22MZfO-MUchYVkmOG-jtA2oLfpvVPbqnY)

Once you have the probabilities, you can compute the likelihood score as follows

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/nc3XUyPHSgyN11Mjx7oMNQ_d2a00c93d29b497bba81a3944072feb9_Screen-Shot-2020-09-08-at-3.43.07-PM.png?expiry=1712620800000&hmac=pTycbVpcKIJhVVhho7J9SLle3BdT6Q_7MsZNXp84BK0](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/nc3XUyPHSgyN11Mjx7oMNQ_d2a00c93d29b497bba81a3944072feb9_Screen-Shot-2020-09-08-at-3.43.07-PM.png?expiry=1712620800000&hmac=pTycbVpcKIJhVVhho7J9SLle3BdT6Q_7MsZNXp84BK0)

We clearly see a problem in the case a word is 0. In other words, if we have a word that is present in one class but not in the other this will add extreme noise into the probability. Hence, we need to apply a smooth operator into the probability. In this case we will be applying the Laplacian Smoothing function.

![Untitled](images/Untitled%205.png)

![Untitled](images/Untitled%206.png)

There is a big problem of using the probabilities directly. Overflow, when multiplying numbers the  resulting number may be very big resulting on computer overflow. For this reason, we use the log function on the the ratios to smooth and lower our numbers.

![Untitled](images/Untitled%207.png)

## Training Naive Bayes

- Collect and annotate corpus
- Preprocess
- Word Count
- Compute P(w | class) using Laplace smoother
- Get Lambda score of each word aka log of previous values
- Get the log prior which is the log number of positive tweets over log of number of negative tweets

### Testing naive Bayes

For testing we compute the score of a tweet and then make a prediction on it

![Untitled](images/Untitled%208.png)

# Week 3: Vector Space Models

Vector spaces are very useful on NLP and representation theory. It help us understand the meaning of sentences and to differentiate when two sentences have similar meaning or different meanings. 

## Word by Word Design

One way to represent the meaning of the sentences as vector is by creating matrix where each row and column corresponds to a word in your vocabulary. Then you can iterate over a document and see the number of times each word shows up next each other word. We can think of *K* as the bandwidth that decides whether two words are next to each other or not. 

![Untitled](images/Untitled%209.png)

The data matrix above is called a co-concurrent matrix.

## Word by Document Design

Applying the  same concept as word by word design we can map words to documents. 

The rows could correspond to words and the columns to documents. The numbers in the matrix correspond to the number of times each word showed up in the document.

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/XXPXAWtOTAqz1wFrTqwK6w_741e4935c5ac435a96373dd745b124f9_Screen-Shot-2021-02-11-at-11.39.33-AM.png?expiry=1712707200000&hmac=-yaz76MpvrfbwOaMVgBcfSMoJgvIzEIlki0uYemfccM](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/XXPXAWtOTAqz1wFrTqwK6w_741e4935c5ac435a96373dd745b124f9_Screen-Shot-2021-02-11-at-11.39.33-AM.png?expiry=1712707200000&hmac=-yaz76MpvrfbwOaMVgBcfSMoJgvIzEIlki0uYemfccM)

You can represent the entertainment category, as a vector *v*=[500,7000]. You can then also compare categories as follows by doing a simple plot.

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/IkrSkusORvCK0pLrDpbwRQ_3ab7a81281fb491dbd0736f83b095b2a_Screen-Shot-2021-02-11-at-11.49.47-AM.png?expiry=1712707200000&hmac=NJk3J1KJyCBSmsQsbZ7kxSolQ4JynO-E1lMox5BAkaM](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/IkrSkusORvCK0pLrDpbwRQ_3ab7a81281fb491dbd0736f83b095b2a_Screen-Shot-2021-02-11-at-11.49.47-AM.png?expiry=1712707200000&hmac=NJk3J1KJyCBSmsQsbZ7kxSolQ4JynO-E1lMox5BAkaM)

## PCA

![Untitled](images/Untitled%2010.png)

![Untitled](images/Untitled%2011.png)

**Steps to Compute PCA:**

- Mean normalize your data
- Compute the covariance matrix
- Compute SVD on your covariance matrix. This returns [U S V] =*svd*(Σ). The three matrices U, S, V are drawn above. U is labelled with eigenvectors, and S is labelled with eigenvalues.
- You can then use the first n columns of vector *U*, to get your new data by multiplying *XU*[:,0:*n*].

# Week 4: Machine Translation

## Learning Objectives

Develop tools to perform Machine Translation and Document Search. Specifically we will cover

- Transform vector
- K nearest neighbors
- Hash tables
- Divide vector space into regions
- Locality sensitive hashing
- Approximated nearest neighbors

learn a mapping that will allow you to translate words by learning a "transformation matrix". Here is a visualization:

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/6_gRc_NRSza4EXPzUWs2IQ_4b3c0a4193444b2eb20dd73544755559_Screen-Shot-2021-02-22-at-9.01.23-AM.png?expiry=1712707200000&hmac=g0hdQftoNXwY0WmePAZTQKEk_SJ6_BT74E1luajpaXs](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/6_gRc_NRSza4EXPzUWs2IQ_4b3c0a4193444b2eb20dd73544755559_Screen-Shot-2021-02-22-at-9.01.23-AM.png?expiry=1712707200000&hmac=g0hdQftoNXwY0WmePAZTQKEk_SJ6_BT74E1luajpaXs)

Here is a visualization of that showing you the aligned vectors:

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/NWrTGJbMQzqq0xiWzDM6NA_9ed7e91d9206497e80cf1c4db61ae086_Screen-Shot-2021-02-22-at-9.05.57-AM.png?expiry=1712707200000&hmac=r2yeT_GZ_1EQZdC8hAJj5FAFwR6gBcRE8T1p1R_ZoF4](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/NWrTGJbMQzqq0xiWzDM6NA_9ed7e91d9206497e80cf1c4db61ae086_Screen-Shot-2021-02-22-at-9.05.57-AM.png?expiry=1712707200000&hmac=r2yeT_GZ_1EQZdC8hAJj5FAFwR6gBcRE8T1p1R_ZoF4)

After you have computed the output of *XR* you get a vector. You then need to find the most similar vectors to your output. Here is a visual example:

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/XaXETsw4RLulxE7MOFS7kg_b2bd6570f342403fb53dfe39f491dc0b_Screen-Shot-2021-02-22-at-9.54.21-AM.png?expiry=1712793600000&hmac=fB0T4R1LQyVxxjIsbSDaNvJ-vXl0GNVo0aRwSXyos14](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/XaXETsw4RLulxE7MOFS7kg_b2bd6570f342403fb53dfe39f491dc0b_Screen-Shot-2021-02-22-at-9.54.21-AM.png?expiry=1712793600000&hmac=fB0T4R1LQyVxxjIsbSDaNvJ-vXl0GNVo0aRwSXyos14)

## Locality Sensitive Hashing

Locality sensitive hashing is a technique that allows you to hash similar inputs into the same buckets with high probability.

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Os9-kXG0Q7GPfpFxtPOxOg_238c5102ee1e4d2a855a8b6b788b7ef5_Screen-Shot-2021-02-22-at-12.08.24-PM.png?expiry=1712793600000&hmac=1xmdupQ3tN-tWX8yQ7C6BITckeofuEgoUPivFhM35ic](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Os9-kXG0Q7GPfpFxtPOxOg_238c5102ee1e4d2a855a8b6b788b7ef5_Screen-Shot-2021-02-22-at-12.08.24-PM.png?expiry=1712793600000&hmac=1xmdupQ3tN-tWX8yQ7C6BITckeofuEgoUPivFhM35ic)

## Approximate Nearest Neighbors

Approximate nearest neighbors does not give you the full nearest neighbors but gives you an approximation of the nearest neighbors. It usually trades off accuracy for efficiency. Look at the following plot:

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/phiXuy1WQHaYl7stVmB2Jg_9033e2a9f3d0496887a9e7953e34105a_Screen-Shot-2021-02-22-at-1.35.41-PM.png?expiry=1712793600000&hmac=TC0k8AckFA66B_QD8mhWZdtqkmwN5hTfc5CM0tia8ec](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/phiXuy1WQHaYl7stVmB2Jg_9033e2a9f3d0496887a9e7953e34105a_Screen-Shot-2021-02-22-at-1.35.41-PM.png?expiry=1712793600000&hmac=TC0k8AckFA66B_QD8mhWZdtqkmwN5hTfc5CM0tia8ec)

## Searching Documents

A toy example of how you can actually represent a document as a vector.

![https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/620nGgp6TaetJxoKel2nwQ_40de3e1790574e8d920d1601422cf869_Screen-Shot-2021-02-22-at-1.43.43-PM.png?expiry=1712793600000&hmac=XWvClQqmNCXKO1Gq-KLzmTrVCQSO8VLT3rLv6eKfxFI](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/620nGgp6TaetJxoKel2nwQ_40de3e1790574e8d920d1601422cf869_Screen-Shot-2021-02-22-at-1.43.43-PM.png?expiry=1712793600000&hmac=XWvClQqmNCXKO1Gq-KLzmTrVCQSO8VLT3rLv6eKfxFI)