# U.S. Patent Phrase to Phrase Matching

The U.S. Patent and Trademark Office (USPTO) offers one of the largest repositories of scientific, technical, and commercial information in the world through its Open Data Portal. Patents are a form of intellectual property granted in exchange for the public disclosure of new and useful inventions. Because patents undergo an intensive vetting process prior to grant, and because the history of U.S. innovation spans over two centuries and 11 million patents, the U.S. patent archives stand as a rare combination of data volume, quality, and diversity.

In this competition, the task is to train machine learning models on a novel semantic similarity dataset to extract relevant information by matching key phrases in patent documents. Determining the semantic similarity between phrases is critically important during the patent search and examination process to determine if an invention has been described before. For example, if one invention claims "television set" and a prior publication describes "TV set", a model would ideally recognize these are the same and assist a patent attorney or examiner in retrieving relevant documents. This extends beyond paraphrase identification; if one invention claims a "strong material" and another uses "steel", that may also be a match. What counts as a "strong material" varies per domain (it may be steel in one domain and ripstop fabric in another, but you wouldn't want your parachute made of steel). We have included the Cooperative Patent Classification as the technical domain context as an additional feature to help you disambiguate these situations.

## Description

•	Applied multiple pretrained framework such as RoBERTa-large，Electra-large, and DeBERTa-v3 to compute and learn the semantic similarity between phrases across multiple U.S. Patent domain contexts 

•Refined downstream structures including both 1D CNN and multi-head attention model on contextualized embedding from pretrained models with increasing learning rates, and applied warming-up and multi-sample dropout for better generalization

•	Used adversarial weight perturbation, fast gradient sign method and pseudo label generation for overfitting reduction and aggerated final results with different loss using ensemble methods based on stratified k-fold cross validation, and ranked top 9% on Kaggle


## Getting Started

### Dependencies

* Kaggle Notebook

## Authors
Siquan Wang

sw3442@cumc.columbia.edu

## License

N/A

## Acknowledgments

competition website:
* https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching
