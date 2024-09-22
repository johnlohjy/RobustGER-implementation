# RobustGER Plan

[RobustGER GitHub](https://github.com/YUCHEN005/RobustGER)

[LipGER GitHub](https://github.com/Sreyan88/LipGER)

## Part 1: Find Dataset

[HuggingFace Dataset Blog](https://huggingface.co/blog/audio-datasets)

[Selecting a Dataset](https://huggingface.co/learn/audio-course/en/chapter5/choosing_dataset)

[HuggingFace Hub for ASR Datasets](https://huggingface.co/datasets?task_categories=task_categories:automatic-speech-recognition&sort=downloads)

**Considerations:**

- Probably easier to have clean speech datasets because we need . Noise can just be injected like in LipGER?
  - Noisy speech is simulated by
    - Adding room reverberations by convolving clean speech with impulse responses from MIT Impulse Response Survey
    - Adding random audio samples from VoxCeleb2 dataset to simulate interfering speakers
    - Adding random audio samples from Audioset to simulate background noise. The signal-to-noise ratio is adjusted randomly between 0 and 40db wrt clean signal

- Number of hours and diversity. If we want a model that generalises well, we want a diverse dataset with lots of different speakers, domains and speaking styles.

- More on Domain: where the data was sourced from. We need to match our domain to the conditions we anticipate at inference time.
  - Each domain has a different distribution of data. For example, audiobooks are recorded in high-quality studio conditions (with no background noise) and text that is taken from written literature. Whereas for YouTube, the audio likely contains more background noise and a more informal style of speech.

- Speaking Style, 2 categories: Narrated and Spontaneous i.e. unscripted/conversational
  - Narrated speech: Spoken articulately and without any errors
  - Spontaneous speech: More colloquial style of speech, with the inclusion of repetitions, hesitations and false-starts

**Our Constraints, from the Considerations: Need to discuss**

- Clean speech
- Diverse dataset (not in terms of domain yet, more on different types of speakers and speaking styles???)
- Spontaeneous speech

**Potential Datasets**
- [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech): English corpus with 10,000 hours of high quality labeled audio. Transcribed audio is from audiobooks, podcasts, Youtube. Covers both read and spontaneous speaking styles and a variety of topics

- [MultiLingual LibriSpeech (English only)](https://huggingface.co/datasets/parler-tts/mls_eng): 44.5k hours of English. Derived from read audiobooks

- Common Voice Datasets
    - https://huggingface.co/datasets/fsicoli/common_voice_19_0
    - https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
    - https://huggingface.co/datasets/fsicoli/common_voice_18_0
    - https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0
    - https://huggingface.co/datasets/fsicoli/common_voice_17_0
    - https://huggingface.co/datasets/mozilla-foundation/common_voice_9_0
    - https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0
    - https://huggingface.co/datasets/mozilla-foundation/common_voice_6_1

- [Fleurs Dataset](https://huggingface.co/datasets/google/fleurs)

- [TED-LIUM Dataset](https://huggingface.co/datasets/LIUM/tedlium): Corpus of English-language TED talks, with transcriptions

- [Emilia Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset): Contains diverse speech data with various speaking styles from diverse video platforms and podcasts on the internet. 46k hours of english

- [AMI Meeting Dataset](https://huggingface.co/datasets/edinburghcstr/ami): 100 Hours of meeting-setting recordings. close-talking and far-field microphones etc.

- [CSTR VCTK Dataset](https://huggingface.co/datasets/CSTR-Edinburgh/vctk): 44-hours of speech data uttered by 110 English speakers with various accents. Each speaker reads out about 400 sentences, which were selected from a newspaper, the rainbow passage and an elicitation paragraph used for the speech accent archive.

- [GLOBE Dataset](https://huggingface.co/datasets/MushanW/GLOBE): High-quality English corpus with worldwide accents. Compared to commonly used English corpora, such as LibriTTS and VCTK, GLOBE is unique in its inclusion of utterances from 23,519 speakers and covers 164 accents worldwide, along with detailed metadata for these speakers. Compared to its original corpus, i.e., Common Voice, GLOBE significantly improves the quality of the speech data through rigorous filtering and enhancement processes, while also populating all missing speaker metadata.

- [Facebook Multilingual Librespeech](https://huggingface.co/datasets/facebook/multilingual_librispeech): Dataset is derived from read audiobooks from LibriVox. 44.5K hours of English

- [People's speech](https://huggingface.co/datasets/MLCommons/peoples_speech): 30,000+ hours of transcribed speech in English languages with a diverse set of speakers. Configurations for the dataset include cc-by-clean, cc-by-dirty, cc-by-sa-clean, cc-by-sa-dirty?

- [Librispeech](https://huggingface.co/datasets/openslr/librispeech_asr): Corpus of approximately 1000 hours of 16kHz read English speech. Derived from read audiobooks from the LibriVox project

- [Facebook Voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli): Raw data is collected from 2009-2020 European Parliament event recordings.

- [Google Xtreme-S](https://huggingface.co/datasets/google/xtreme_s): XTREME-S covers speech recognition with Fleurs, Multilingual LibriSpeech (MLS) and VoxPopuli, speech translation with CoVoST-2, speech classification with LangID (Fleurs) and intent classification (MInds-14) and finally speech(-text) retrieval with Fleurs. XTREME-S aims for task, domain and language diversity. Tasks should be diverse and cover several domains to provide a reliable evaluation of model generalization and robustness to noisy naturally-occurring speech in different environments.

## Part 2: Use Pre-trained ASR to output N-best Hypotheses

## Part 3: Extract unrefined language-space noise-embedding from N-best Hypotheses

## Part 4: Refine unrefined language noise-embedding

## Part 5: Parameter-efficient Fine-Tune LLM with language-space noise-embedding

![alt text](<images/RobustGER Architecture.jpg>)