Download Link: https://assignmentchef.com/product/solved-dt2119-lab1-feature-extraction
<br>
The objective is to experiment with different features commonly used for speech analysis and recognition. The lab is designed in Python, but the same functions can be obtained in Matlab/Octave or using the Hidden Markov Toolkit (HTK). In Appendix A, a reference table is given indicating the correspondence between different systems.

<h1>1         Task</h1>

<ul>

 <li>compute MFCC features step-by-step</li>

 <li>examine features</li>

 <li>evaluate correlation between feature</li>

 <li>compare utterances with Dynamic Time Warping</li>

 <li>illustrate the discriminative power of the features with respect to words</li>

 <li>perform hierarchical clustering of utterances</li>

 <li>train and analyze a Gaussian Mixture Model of the feature vectors.</li>

</ul>

In order to pass the lab, you will need to follow the steps described in this document, and present your results to a teaching assistant. Use Canvas to book a time slot for the presentation. Remember that the goal is not to show your code, but rather to show that you have understood all the steps.

<h1>2         Data</h1>

The files lab1_data.npz and lab1_example.npz contain the data to be used for this exercise. The files contains two arrays: data and example<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>.

3.1      example

The array example can be used for debugging because it contains calculations of all the steps in Section 4 for one utterance. It can be loaded with:

import numpy as np

example = np.load(‘lab1_example.npz’)[‘example’].item()

The element example is a dictionary with the following keys:

samples:     speech samples for one utterance samplingrate:     sampling rate

frames:       speech samples organized in overlapping frames preemph:      pre-emphasized speech samples windowed:        hamming windowed speech samples spec:           squared absolute value of Fast Fourier Transform mspec:         natural log of spec multiplied by Mel filterbank mfcc:      Mel Frequency Cepstrum Coefficients lmfcc:               Liftered Mel Frequency Cepstrum Coefficients Figure 1 shows the content of the elements in example.

3.2      data

The array data contains a small subset of the TIDIGITS database (<a href="https://catalog.ldc.upenn.edu/LDC93S10">https://catalog.ldc. </a><a href="https://catalog.ldc.upenn.edu/LDC93S10">upenn.edu/LDC93S10</a><a href="https://catalog.ldc.upenn.edu/LDC93S10">)</a> consisting of a total of 44 spoken utterances from one male and one female speaker<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>. The file was generated with the script lab1_data.py<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>. For each speaker, 22 speech files are included containing two repetitions of isolated digits (eleven words: “oh”, “zero”, “one”, “two”, “three”, “four”, “five”, “six”, “seven”, “eight”, “nine”). You can read the file from Python with:

data = np.load(‘lab1_data.npz’)[‘data’]

The variable data is an array of dictionaries. Each element contains the following keys:

filename:                  filename of the wave file in the database

samplingrate:           sampling rate of the speech signal (20kHz in all examples) gender: gender of the speaker for the current utterance (man, woman) speaker:   speaker ID for the current utterance (ae, ac) digit:     digit contained in the current utterance (o, z, 1, …, 9) repetition:     whether this was the first (a) or second (b) repetition samples:    array of speech samples

<h1>3         Mel Frequency Cepstrum Coefficients step-by-step</h1>

Follow the steps below to computer MFCCs. Use the example array to double check that your calculations are right.

You need to implement the functions specified by the headers in proto.py. Once you have done this, you can use the function mfcc in tools.py to compute MFCC coefficients in one go.

4.1       Enframe

Implement the enframe function in proto.py. This will take as input speech samples, the frame length in samples and the number of samples overlap between consecutive frames and outputs a

Figure 1. Evaluation of MFCCs step-by-step

two dimensional array where each row is a frame of samples. Consider only the frames that fit into the original signal disregarding extra samples. Apply the enframe function to the utterance example[‘samples’] with window length of 20 milliseconds and shift of 10 ms (figure out the length and shift in samples from the sampling rate, and write it in the lab report). Use the pcolormesh function from matplotlib.pyplot to plot the resulting array. Verify that your result corresponds to the array in example[‘frames’].

4.2       Pre-emphasis

Implement the preemp function in proto.py. To do this, define a pre-emphasis filter with preemphasis coefficient 0.97 using the lfilter function from scipy.signal. Explain how you defined the filter coefficients. Apply the filter to each frame in the output from the enframe function. This should correspond to the example[‘preemph’] array.

4.3         Hamming Window

Implement the windowing function in proto.py. To do this, define a hamming window of the correct size using the hamming function from scipy.signal with extra option sym=False<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a>. Plot the window shape and explain why this windowing should be applied to the frames of speech signal. Apply hamming window to the pre-emphasized frames of the previous step. This should correspond to the example[‘windowed’] array.

4.4          Fast Fourier Transform

Implement the powerSpectrum function in proto.py. To do this, compute the Fast Fourier Transform (FFT) of the input from scipy.fftpack and then the squared modulus of the result. Apply your function to the windowed speech frames, with FFT length of 512 samples. Plot the resulting power spectrogram with pcolormesh. Beware of the fact that the FFT bins correspond to frequencies that go from 0 to <em>f</em>max and back to 0. What is <em>f</em>max in this case according to the Sampling Theorem? The array should correspond to example[‘spec’].

4.5          Mel filterbank log spectrum

Implement the logMelSpectrum function in proto.py. Use the trfbank function, provided in the tools.py file, to create a bank of triangular filters linearly spaced in the Mel frequency scale. Plot the filters in linear frequency scale. Describe the distribution of the filters along the frequency axis. Apply the filters to the output of the power spectrum from the previous step for each frame and take the natural log of the result. Plot the resulting filterbank outputs with pcolormesh. This should correspond to the example[‘mspec’] array.

4.6           Cosine Transofrm and Liftering

Implement the cepstrum function in proto.py. To do this, apply the Discrete Cosine Transform (dct function from scipy.fftpack.realtransforms) to the outputs of the filterbank. Use coefficients from 0 to 12 (13 coefficients). Then apply liftering using the function lifter in tools.py. This last step is used to correct the range of the coefficients. Plot the resulting coefficients with pcolormesh. These should correspond to example[‘mfcc’] and example[‘lmfcc’] respectively.

Once you are sure all the above steps are correct, use the mfcc function (tools.py) to compute the liftered MFCCs for all the utterances in the data array. Observe differences for different utterances.

<h1>4         Feature Correlation</h1>

Concatenate all the MFCC frames from all utterances in the data array into a big feature [<em>N </em>×<em>M</em>] array where <em>N </em>is the total number of frames in the data set and <em>M </em>is the number of coefficients. Then compute the correlation coefficients between features and display the result with pcolormesh. Are features correlated? Is the assumption of diagonal covariance matrices for Gaussian modelling justified? Compare the results you obtain for the MFCC features with those obtained with the Mel filterbank features (‘mspec’ features).

<h1>5         Comparing Utterances</h1>

Given two utterances of length <em>N </em>and <em>M </em>respectively, compute an [<em>N </em>×<em>M</em>] matrix of local Euclidean distances between each MFCC vector in the first utterance and each MFCC vector in the second utterance.

Write a function called dtw (proto.py) that takes as input this matrix of local distances and outputs the result of the Dynamic Time Warping algorithm. The main output is the global distance between the two sequences (utterances), but you may want to output also the best path for debugging reasons.

For each pair of utterances in the data array:

<ol>

 <li>compute the local Euclidean distances between MFCC vectors in the first and secondutterance</li>

 <li>compute the global distance between utterances with the dtw function you have written</li>

</ol>

Store the global pairwise distances in a matrix <em>D </em>(44×44). Display the matrix with pcolormesh. Compare distances within the same digit and across different digits. Does the distance separate digits well even between different speakers?

Run hierarchical clustering on the distance matrix <em>D </em>using the linkage function from scipy.cluster.hierarchy. Use the ”complete” linkage method. Display the results with the function dendrogram from the same library, and comment them. Use the tidigit2labels function (tools.py) to create labels to add to the dendrogram to simplify the interpretation of the results.

<h1>6         Explore Speech Segments with Clustering</h1>

Train a Gaussian mixture model with sklearn.mixture.GMM. Vary the number of components for example: 4, 8, 16, 32. Consider utterances containing the same words and observe the evolution of the GMM posteriors. Can you say something about the classes discovered by the unsupervised learning method? Do the classes roughly correspond to the phonemes you expect to compose each word? Are those classes a stable representation of the word if you compare utterances from different speakers. As an example, plot and discuss the GMM posteriors for the model with 32 components for the four occurrences of the word “seven” (utterances 16, 17, 38, and 39).

<h1>A             Alternative Software Implementations</h1>

Although this lab has been designed for being carried out in Python, several implementations of speech related functions are available.

A.1           Matlab/Octave Instructions

The Matlab signal processing toolbox is one of the most complete signal processing piece of software available. Many speech related functions are however implemented in third party toolboxes. The most complete are the Voicebox<a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a> which is more oriented towards speech technology and the Auditory Toolbox<sup>6 </sup>that is more focused on human auditory models.

If you use Octave instead of Matlab, make sure you have the following extra packages (in parentheses are the names of the corresponding apt-get packages for Debian based GNU Linux distributions, all packages are already installed on CSC Ubuntu machines):

<ul>

 <li>signal (octave-signal)</li>

</ul>

A.2               Hidden Markov Models Toolkit (HTK)

HTK is a powerful toolkit developed by Cambridge University for performing HMM-based speech recognition experiments. The HTK package is available at all CSC Ubuntu stations, or can be download for free at <a href="http://htk.eng.cam.ac.uk/">http://htk.eng.cam.ac.uk/</a> after registration to the site. Its manual, the HTK Book, can be downloaded separately. In spite of being open source and free of charge, HTK, is unfortunately not free software in the Free Software Foundation sense because neither its original form nor its modifications can be freely distributed. Please refer to the license agreement for more information.

The HTK commands that are relevant to this exercise are the following:

HCopy: feature extraction tool. Can read audio files or feature files in HTK format and outputs HTK format files

HList: terminal based visualization of features. Reads HTK format feature files and displays information about them General options are:

<ul>

 <li>-C config: reads configuration file conf</li>

 <li>-S filelist: reads list of files to process from filelist for a complete list of options and usage information, run the commands without arguments.</li>

</ul>

Hint: HList -r …: the -r option in HList will output the feature data in raw (ascii) format. This will make it easy to import the features in other programs such as python, Matlab or R.

Table 2 lists a number of possible spectral features and the corresponding HTK codes to be used in HCopy or HList.

<table width="577">

 <tbody>

  <tr>

   <td width="192">Feature name</td>

   <td width="121">Matlab</td>

   <td width="264">Python</td>

  </tr>

  <tr>

   <td width="192">Linear filter</td>

   <td width="121">filter</td>

   <td width="264">scipy.signal.lfilter</td>

  </tr>

  <tr>

   <td width="192">Hamming window</td>

   <td width="121">hamming</td>

   <td width="264">scipy.signal.hamming</td>

  </tr>

  <tr>

   <td width="192">Fast Fourier Transform</td>

   <td width="121">fft</td>

   <td width="264">scipy.fftpack.fft</td>

  </tr>

  <tr>

   <td width="192">Discrete Cosine Transform</td>

   <td width="121">dct</td>

   <td width="264">scipy.fftpack.realtransforms.dct</td>

  </tr>

  <tr>

   <td width="192">Gaussian Mixture Model</td>

   <td width="121">gmdistribution</td>

   <td width="264">sklearn.mixture.GMM</td>

  </tr>

  <tr>

   <td width="192">Hierarchical clustering</td>

   <td width="121">linkage</td>

   <td width="264">scipy.cluster.hierarchy.linkage</td>

  </tr>

  <tr>

   <td width="192">Dendrogram</td>

   <td width="121">dendrogram</td>

   <td width="264">scipy.cluster.hierarchy.dendrogram</td>

  </tr>

  <tr>

   <td width="192">Plot lines</td>

   <td width="121">plot</td>

   <td width="264">matplotlib.pyplot.plot</td>

  </tr>

  <tr>

   <td width="192">Plot arrays</td>

   <td width="121">image, imagesc</td>

   <td width="264">matplotlib.pyplot.pcolormesh</td>

  </tr>

 </tbody>

</table>

Table 1. Mapping between Matlab and Python functions used in this exercise

<table width="314">

 <tbody>

  <tr>

   <td width="240">Feature name</td>

   <td width="74">KTH code</td>

  </tr>

  <tr>

   <td width="240">linear filer-bank parameters</td>

   <td width="74">MELSPEC</td>

  </tr>

  <tr>

   <td width="240">log filter-bank parameters</td>

   <td width="74">FBANK</td>

  </tr>

  <tr>

   <td width="240">Mel-frequency cepstral coefficients</td>

   <td width="74">MFCC</td>

  </tr>

  <tr>

   <td width="240">linear prediction coefficients</td>

   <td width="74">LPC</td>

  </tr>

 </tbody>

</table>

Table 2. Feature extraction in HTK. The HCopy executable can be used to generate features from wave file to feature file. HList can be used to output the features in text format to stdout, for easy import in other systems

<a href="#_ftnref1" name="_ftn1"></a>