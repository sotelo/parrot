# Speech synthesis using recurrent neural networks.

This repo has the code for our ICLR submission:

Jose Sotelo, Soroush Mehri, Kundan Kumar, João Felipe Santos, Kyle Kastner, Aaron Courville, Yoshua Bengio. Char2Wav: End-to-End Speech Synthesis.

The website is [here](http://www.josesotelo.com/speechsynthesis).

NOTE: The code is currently being refactored/cleaned/documented. We wanted to make it available as soon as possible but we know well that the current version is not ready for replication. Therefore, if you're interested in this, please come back later. Alternatively, you can send me an email and I will let you know when it's ready.

NOTE(2): The code for the neural vocoder is based on [sampleRNN](https://github.com/soroushmehr/sampleRNN_ICLR2017).

Updates:
* SampleRNN modules added
* End-to-end model can generate more than 4s of audio in 1s on p6000 gpu. (This is achieved by generating 200 samples of length 10s in a single batch. Total time taken for this is 448 seconds.)
