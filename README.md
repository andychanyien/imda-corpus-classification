# Singaporean Speech Classification with Neural Networks - Capstone Project

## Table of Content

* [Full Problem Statement](#Problem-Statement)
* [Repo Structure](#Repo-Structure)
* [Libraries and System Specification](#Libraries-and-System-Specifications)
* [Data Dictionary](#Data-Dictionary)
* [Executive Summary](#Executive-Summary)
* [Conclusions and Recommendations](#Conclusions-and-Recommendations)
* [Sources and Other Useful Resources](#Sources-and-Other-Useful-Resources)


## Problem Statement

Speech recognition programmes have the ability to convert voice to text, mostly by building a Speech-To-Text (STT) model using Machine Learning. This type of STT programmes are commonly seen in smart phones or other websites which have speech-enabled inputs. However, **many of these "off-the-shelves" speech recognition programmes have difficulty recognising Singaporean accented English as they are not trained with Singaporean speeches. Furthermore, Singaporeans do not speak with one universal accent, making it even more challenging for speech recognition models.**

I would like to find out if it is possible for machines to understand Singaporean accent. In data science terms, I wish to find out **how accurately can machine learning algorithms classify Singaporean-accented English** and the potential scaling it up as a business solution..

To investigate this, a **multi-classification model** will be built, with a total of **5 classes**. The machine learning algorithm used will be mainly neural networks, consisting of a regular **Feedforward Neural Network, a Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN).**

The audio features which will be explored are their Mel-Frequency Cepstral Coefficients (MFCCs) and Mel-Spectrograms. The training data will be preprocessed differently based on these features and subsequently be fed into the Neural Networks.

**Metrics of Evaluation:**

The main metric for evaluation is the average weighted **Accuracy** on both validation set and unseen data. The higher the accuracy, the better the model is at classifying the singaporean-accented words into the 5 classes. Additional metrics for consideration includes **Precision** and **Recall** to identify if the machine learning algorithm would mistake words that sounds similar.

As speed is also an important factor for real-time transcription or Speech-to-text translation, **Computational Time** will also be used to evaluate the model, based on pre-processing time and training time.

**Relevance for Stakeholders:**

A simple speech classification model can be scaled up to recognise more words or even commands to build a speech recognition engine. There will be businesses which could benefit from it. For example, call centres could build an Automated Attendant which is trained with Singaporean-accented commands, allowing callers to navigate the menu system without pressing physical buttons. Such navigation systems can also be implemented for placing orders at a restaurants.

This can also be scaled up to build a complex speech recognition models based on phonetics to comprehend long sentences, allowing for real-time transcribing. For example, providing subtitles for live performances or live TV-shows. This will enhance the experience of audiences and increase watch-rate.

*Secondary Audience*

Furthermore, this will improve accessibility for some visually impaired or physically challenged people, who have difficulty interacting with physical touch menus. 

**Data source:**

Singapore IMDA National Speech Corpus (NSC)

https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus

## Repo Structure
```
root:.
|   1 Data Collection & Scoping.ipynb
|   2 Audio EDA.ipynb
|   3 Preprocessing and Modelling.ipynb
|   git command mass extraction.xlsx
|   README.md
|       
+---assets
|   |   ABOUT.txt
|   |   
|   +---audio_test
|   |   101 audio files (in .WAV) 
|   |         
|   +---audio_train
|   |   1049 audio files (in .WAV) 
|   |         
|   +---images
|   |  
|   +---spec_test
|   |   101 mel-spectrograms (in .png)
|   |   
|   +---spec_train
|   |   1049 mel-spectrograms (in .png)
|   |  
|   \---transcript
|       2034 transcripts (in .txt)
|           
\---datasets
    |   test.csv
    |   train.csv
    \---transcripts_session_0.csv
```

## Libraries and System Specifications

**System Specifications**

The results in these notebooks were achieved in an Anaconda virtual environment with the following configurations.

It is highly recommended that users create a virtual environment for this repo, especially if users intend to use GPU for tensorflow.

Anaconda Client Version: 1.7.2
Python Version: 3.6.12
GPU: GeForce GTX 1660 Super

|**Library**|Description|Version No.|
|---|---|---|
|**Librosa**|For processing audio files and extracting audio features|0.8.0|
|**Keras-gpu**|For tensorflow neural networks, note that this is the GPU version|2.3.1|
|**Tensorflow-gpu**|For tensorflow neural networks, note that this is the GPU version|2.1.0|
|**iPython**|Mainly for `ipython.display` to enable audio playback. This is not mandatory for modelling|7.16.1|
|**Pillow**|For processing image data for the CNN Model|8.0.0|
|**NLTK**|For processing transcript, for stopwords feature, one can use other stopwords libraries|3.5|
|**Numpy**|Data Science essential, for creating numpy arrays and for manipulating matrices|1.19.1|
|**Pandas**|To view the data in a dataframe and for analysing the datasets|1.1.3|
|**Scikit-learn**|Used quite extensively to build the model, process the features and even for vectorising the transcript|0.23.2|
|**Matplotlib**|For plotting graphs during EDA and for creating Mel-Spectrograms for the CNN model, note that this is also a dependency for Librosa's waveplot method|3.3.1|

Users who do not use GPU can simply use the CPU library with the same version number.

## Data Dictionary

Dataset: transcript_0.csv

Description: This is the transcript of recordings from session 0, which is the first session.

|**Feature**|Type|Description|
|---|---|---|
|**id**|int|The given ID of each audio data, containing information about the which recording session, speaker id and transcript id|
|**text**|str|The actual transcription of their recording, provided by IMDA|
|**speaker**|int|Speaker identification number, ranging from 1 to beyond 1000|
|**session**|int|The recording session number, 0 being the first recording and 1 being the second recording session and so on|
|**line**|int|The transcript ID|
|**wordcount**|int|The number of words based on the transcription|

Dataset: train.csv & test.csv

Description: training dataset generated from the audio files, containing metadata about each audio files. The test set contains unseen data which will be used to evaluate the model.

|**Feature**|Type|Description|
|---|---|---|
|**id**|int|The given ID of each audio data, containing information about the which recording session, speaker id and transcript id|
|**filepath**|str|The relative file path of the audio files in this repo|
|**duration**|float|The duration of each audio clip in seconds (0.5 means 0.5 seconds)|
|**class_label**|str|Label of the audio clip ("apples" means it is a recording of someone pronouncing "apple")|
|**mfccs_mean**|np.array, float|The average of 40 Mel-Frequency Cepstral Coefficients across time (Sum of each MFCC per timestep, divided by timesteps)|
|**mfccs_std**|np.array, float|The standard deviation of 40 Mel-Frequency Cepstral Coefficients across time (Standard deviation of each MFCC across the whole duration)|
|**mfccs_delta_mean**|np.array, float|The difference in magnitutde between a specific timestep and the previous timestep|
|**mfccs_delta_std**|np.array, float|The difference between the delta at a specific timestep and the previous timestep|
|**combined_mfccs**|np.array, float|An array combining the 4 features above|
|**mfcc_pad**|np.array, int|The Mel-Frequency Cepstral Coefficients across timesteps but post-padded to the length of the audio flip with the longest duration|
|**mfcc_pad_combined**|np.array, int|An array combining the Mel-Frequency Cepstral Coefficients, its delta and delat-delta, post-padded to the length of the audio flip with the longest duration|
|**mel**|np.array, float|The plotting data of the Mel-Frequency Spectrogram|

## Executive Summary

**Goal**

- Build a Multiclassification Model
- Determine how accurate can machine learning algorithm classify Singaporean-accented English
- Explore the possibilities of scaling it up to a speech recognition system which can potentially be implemented in businesses

**Metrics**

- Accuracy on validation dataset and unseen dataset
- Computational time

**Data Used**

- Audio files extracted from [National Speech Corpus provided by the Singapore Infocomm Media Development Authority (IMDA)](https://www.imda.gov.sg/programme-listing/digital-services-lab/national-speech-corpus)

**Findings**

- It is difficult to classify speeches accurately based on raw sound waves alone due to the complexity of audio signals
- Decomposing sound waves into MFCCs allows the neural network to understand the unique characteristics of each label.
- Classification of audio with images is also very possible as patterns among each class is quite evident when comparing their Mel-Spectrogram.

**Features for Model**

- Mel-Frequency Cepstral Coefficients
    - Mean, Standard Deviations, Delta and Delta-Delta
    - Padded Sequences
- Mel-Spectrograms
    - Images

**Models Developed** 

7 multiclassification models were developed and tested
- 3 regular Feedforward Neural Network (FNN)
- 2 Convolutional Neural Network (CNN)
- 2 Recurrent Neural Network (RNN)

**Model Evaluation and Selection**

Top 3 Models:
1. RNN with Gated Recurrent United (RNN - GRU), with padded sequence (98% val. accuracy, 97% unseen accuracy, 4mins training time)
2. CNN model, with Mel-Spectrogram images (99% val. accuracy, 96% unseen accuracy, 6mins training time)
3. RNN with Long-Short Term Memory (LSTM), with padded sequence (97% val. accuracy, 93% unseen accuracy, 3mins training time)

**Conclusion**

- In general, neural networks are able to classify Singporean-accented English with very high accuracy
- RNN models are most efficient with sequential data, best for real-time transcription.
- CNN models are also highly accurate, could be used if the objective is only classification

*Limitations*

- audio recordings are in controlled environment, result might differ if there is background noises
- speakers are all proficient English speakers, to consider having a more inclusive dataset with less proficient speakers


## Conclusions and Recommendations

Based on the results, Neural Networks can classify Singaporean-accented English with at least 90% accuracy. The best model for the multiclassification is the Recurrent Neural Network with GRU layers (Model 7). It has successfully classified the speeches with an average accuracy of 98% on validation dataset and generalised to unseen data with an accuracy of 97%, with the highest efficiency. Image classification with CNN uses more preprocessing and processing time, even though it also returned high accuracy.

The RNN (GRU) Model definitely has the potential to be scaled up and be implemented as a business solution. It can be trained to recognise more words or commands and with larger dataset. This also means that building a more complex speech recognition model which can accurately and efficiently recognise Singaporean-accented speeches is very possible.

**Recommendation for Implementation**

- MFCC sequential data should be used on sequential models (such as Recurrent Neural Network) for real-time transcription or classification as it has proved to be the most efficient way to process them.

- Mel-spectrogram images should be used for when accuracy is more important but computational time is not important. For example, trying to sort audio recordings or classifying speech patterns.

**Limitations**

- The recordings that were used in this project were recorded in a controlled environment, with a good quality microphone in a quiet room (most probably in a studio). However, we cannot expect the audio input environment to always be quiet enough to isolate the speaker's speech. For practical application, the dataset could perhaps be expanded to include speeches recorded in a noisier environment. Perhaps, another model that can cancel background noises could be built to filter noises out of the recordings first before running through the RNN.

- All of the recordings are also scripted, and are not based on spontaneous conversations, hence, they are almost free from mispronunciation. The audio data was collected from people with good english proficiency as they were able to read from the script. This way, the classification models or speech recognition models could be biased towards proficient English speakers. One possibility would be for data collect of people with lower english proficiency as they may pronounce the same words slightly differently.

**Suggestion for Further Research**

Data Scientists interested in audio classification and speech classification of Singaporean-accented speech can explore the following to build a more inclusive and accurate model.

- Explore further preprocessing techniques for MFCC. The MFCC turns out to be a great determinant in classifying the speeches. 
- Increase the training dataset. After doubling the training data from 400 to 800, it increased the accuracy of RNN model by a few percentage points.
- Expand the training set to include less proficient English speakers
- Build a model that can classify speeches in noisy environments.

## Sources and Other Useful Resources

This project would not have been possible without the resources from the data science community. I have done lots of research and below are what I find the most helpful among everything I had consumed. The challenge was not the lack of resources. Rather, I found it to be like an echo-chamber where people mostly just copy and paste codes without understanding why they do certain things. For example, I had difficulty finding out why certain features were used or why were the audio signals processed in a certain way. To save you time, I have compiled what I feel is the most essential materials below.

I have organised them into two sections. *Main Reference Materials* are the most important materials which have formed the core of this project. They are important for understanding the intricacies of handling audio data. *Other Reference Materials* are materials which are good to widen your knowledge and could be use to enhance your own project.

**Main Reference Materials**

- [The Sound of AI - Valerio Velardo Youtube Channel](#https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ)
	- I had spent days and weeks listening to his interesting lectures. They are pretty long but very worth going through them. I think this is the only place where you can find such a comprehensive and clear explanation behind processing audio signals. It is a little bit technical but not too technical for beginners of data science.

- [How to apply machine learning and deep learning methods to audio analysis](#https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc)
	- It has a very detailed step-by-step guide on entire process. It helps you to understand the data science process flow of audio classification with machine learning. The author also had a bonus lesson on Comet. Comet is a machine learning cloud-based platform that can help you track, monitor, analyse and optimise your machine learning model. I did experimented with Comet but did not use it.

- [Voice Classification Blog on Github by Jurgenarias](#https://github.com/jurgenarias/Portfolio/tree/master/Voice%20Classification)
	- This has helped me understand the process of transforming audio signals into image files for CNN

- [Multi-class Metrics made Simple](#https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1)
	- On how to evaluate a multiclassification model

- [How to use Learning Curves to Diagnose Machine Learning Model Performance](#https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
	- This is important for understanding how to tweak your model to prevent overfitting or underfitting, as neural networks tend to overfit.

**Other Reference Materials**

- [Github Blog on Audio](#https://github.com/scarecrow1123/blog/issues/9)
	- this gives a general overview the handling of audio files, things to take note etc

- [Urban Sound Classification with Librosa Nuanced Cross Validation](#https://towardsdatascience.com/urban-sound-classification-with-librosa-nuanced-cross-validation-5b5eb3d9ee30)
	- A popular dataset on audio classification, but it has complications with doing cross validation for audio chunks belonging to the same original sequence. This blog goes through how to conduct the complicated cross validation.

- [Learn to Build Your First Speech to Text Model](#https://www.analyticsvidhya.com/blog/2019/07/learn-build-first-speech-to-text-model-python/)
	- A clear guided walkthrough on speech to text model with python. 

- [Audio Data Analysis Using Deep Learning with Python](#https://www.analyticsvidhya.com/blog/2019/07/learn-build-first-speech-to-text-model-python/)
	- A walkthrough on how to preprocess audio data with codes and some explanation

