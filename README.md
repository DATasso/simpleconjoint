# simpleconjoint [Work in progress]

## What is this? 

simpleconjoint is a package to perform conjoint analysis in Python.

- Right now it only has functions to perform a choice-based conjoint, an example of this can be found in the [cbc jupyter notebook](notebooks/cbc.ipynb)

## Dependencies and installation

- PyStan
- Numpy
- Cython
- Pandas
- XlsxWriter

In addition, the PyStan version used (2.19+) needs a C++14 compatible compiler. For GCC 4.9.3+ and GCC 5+ versions are up-to-date. If you have any trouble it's recommended to follow their detailed instructions about this topic: [Setting up C++ compiler](https://pystan2.readthedocs.io/en/latest/installation_beginner.html#setting-up-c-compiler)

You can get the latest released version using the Python Package Index (PyPI)
```sh
pip install simpleconjoint
```


## What is Conjoint Analysis

It is a multivariate technique that allows evaluating to what extent the people surveyed value the characteristics that make up a product or service.

This is done by asking a sample of the population to indicate their preferences regarding a series of possible combinations of characteristics, on a specific product.

For example, a cell phone, some attributes are evaluated such as: brand, storage capacity, battery, resolution , etc., so that the customer emulates his decision as he would in a real situation, that is, not because of their separate characteristics, but all of them in a single product.

Analyzing the results obtained on their preferences is what allows us to observe how customers value each of the possible characteristics that make up the product, seeking to determine the relative importance of various product attributes and utilities assigned to different levels of said attributes.

## Basic Glossary

**Attribute:** This word is used to refer to a characteristic of a product, for example, the flavor of an ice cream, the color of an object, the brand of a computer, etc.

<hr>

**Level:** Each attribute can have a series of varieties, this is what we mean by levels, for example, the "color" attribute of a certain product can be blue, green or red.

<hr>

**Alternative:** Also called a profile, it is the set of combinations of attributes of a product, that is, the final product itself, for example, a cellphone with "X" brand, a 4000 mAh battery and a 32MP camera.

<hr>

**Task:** Also known sometimes as set or scenario. When answering a survey on conjoint analysis, respondents are repeatedly asked to indicate to show their preferences regarding a fixed number of alternatives that are shown to them. We refer to each question with the term task, for example, a survey can ask the user 10 times their preferences about the alternative that are shown, that is, the user must perform this task 10 times, comparing a fixed number of alternatives each time and different scenarios each time (some of the profiles could be repeated but not the group of alternatives).

<hr>

**Utility or Part-worth:** As a result of applying a conjoint analysis, the count of numerical values that represent the degree of preference for each attribute level is obtained, measuring how much each feature influences the customer's decision to select an alternative given a set, these are the so-called utilities, whose sum for each attribute must be 0 (meaning the utilities are zero centered).

## Types of Conjoint Analysis

### Traditional conjoint (CVA or Conjoint Value Analysis)

It was the first of these techniques, developed in the 70s. This type of conjoint analysis is simple and currently little used, in which the user is shown an option and is asked to select a value of a rating scale for such option, that is, quantify each alternative or profile. Part of the hypothesis that the valuation assigned by the respondent is directly the utility he or she perceives from the product. The utility of the parts that make up the product is calculated using a multiple linear regression.

<hr>

### Adaptive Conjoint Analysis (ACA)

It was released by Sawtooth in 1985. The ACA model was designed as a computer-based card sorting tool. It allowed the researchers to measure more attributes than they could with the CVA, making ACA a popular choice due to ease of use and more powerful analysis.

<hr>

### Choice Based Conjoint Analysis (CBC)

It was released in the 1990s and quickly became the most popular conjoint analysis. Instead of making ratings, it shows profiles of "products" among which respondents are asked to choose the one they prefer, sometimes being able to include the option “none of the above”. It is still the most popular model for joint analysis today. To perform this type of analysis, discrete choice models are needed, such as the Multinomial Logistic Regression and the Hierarchical Bayes Model, which are the most used models for this type of analysis at the aggregate level.

<hr>

### Adaptive Choice Based Conjoint (ACBC)

> is a newer methodology that was introduced around 2010. ACBC combines elements of CBC (Choice-Based Conjoint), artificial intelligence, and (optionally) dynamic list-building. It tends to probe more deeply into each respondent’s decision structure than traditional CBC. ACBC “combines the best aspects of adaptive interviewing with the realism and accuracy of choice data” (Orme, Brian. Getting Started with Conjoint Analysis).
> -- <cite>https://martecgroup.com/conjoint-analysis/ </cite> 

> An Adaptive Choice interview is an interactive experience, customized to the preferences and opinions of each individual. It tends to probe more deeply into each respondent’s decision structure than a traditional Choice-Based Conjoint , but the survey is often twice to three times as long. Fortunately, respondents find the adaptive nature of the survey more engaging than CBC, so they usually perceive the questionnaire to be more enjoyable and to last about as long as the shorter CBC.
> -- <cite>https://sawtoothsoftware.com/conjoint-analysis/acbc</cite>
