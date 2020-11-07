# Singaporean Speech Classification with Neural Networks - Capstone Project

## Problem Statement

Speech recognition programmes have the ability to convert voice to text, mostly by building a Speech-To-Text (STT) model using Machine Learning. This type of STT programmes are commonly seen in smart phones or other websites which have speech-enabled inputs. However, **many of these "off-the-shelves" speech recognition programmes have difficulty recognising Singaporean accented English as they are not trained with Singaporean speeches. Furthermore, Singaporeans do not speak with one universal accent, making it even more challenging for speech recognition models.**

I would like to find out if it is possible for machines to understand Singaporean accent. In data science terms, I wish to find out **how accurately can machine learning algorithms classify Singaporean-accented English** and the potential scaling it up as a business solution..

To investigate this, a **multi-classification model** will be built, with a total of **5 classes**. The machine learning algorithm used will be mainly neural networks, consisting of a regular **Feedforward Neural Network, a Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN).**

The audio features which will be explored are their Mel-Frequency Cepstral Coefficients (MFCCs) and Mel-Spectrograms. The training data will be preprocessed differently based on these features and subsequently be fed into the Neural Networks.

**Measurement of Success:**

The main metric for evaluation is the average weighted **accuracy**. The higher the accuracy, the better the model is at classifying the singaporean-accented words into the 5 classes. Additional metrics for consideration includes **Precision** and **Recall** to identify if the machine learning algorithm would mistake words that sounds similar.

Computational time as well

**Relevance for Stakeholders:**

A simple speech classification model can be scaled up to recognise more words or even commands to build a speech recognition engine. There will be businesses which could benefit from it. For example, call centres could build an Automated Attendant which is trained with Singaporean-accented commands, allowing callers to navigate the menu system without pressing physical buttons. Such navigation systems can also be implemented for placing orders at a restaurants.

This can also be scaled up to build a complex speech recognition models based on phonetics to comprehend long sentences, allowing for real-time transcribing. For example, providing subtitles for live performances or live TV-shows. This will enhance the experience of audiences and increase watch-rate.

*Secondary Audience*

Furthermore, this will improve accessibility for some visually impaired or physically challenged people, who have difficulty interacting with physical touch menus. 


**Data source:**

Singapore IMDA National Speech Corpus (NSC)

https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus
