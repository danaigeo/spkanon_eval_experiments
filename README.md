# Robustness and Validity of state-of-the-art ASV systems

 The task of these experiments is to find a configuration of the ASV system that reliably measures the privacy gained through the anonymization, providing results that generalize to other datasets and use cases. To this end, we experiment with different numbers of speakers and samples per speaker, assessing the roles these two factors play when measuring privacy.

## SpkAnon Eval

The following experimentation is based on the evaluation framework found in the `spkanon_eval` respository: 
https://github.com/carlosfranzreb/spkanon_eval used for privacy evaluation of speaker anonymization models

## Evaluated anonymization models

The anonymization models are part of a separate repository, as well as the build scripts required for them. They are:

- **StarGANv2-VC**: voice conversion model trained with 20 target speakers of VCTK.
- **kNN-VC**: any-to-any voice conversion model trained on LibriSpeech train-other-100 and 40 targets provided by the dev-clean dataset.

The components and build instructions are to be found in the `spkanon_models` repository: <https://github.com/carlosfranzreb/spkanon_models>.

## Datasets:

Experimentation was conducted on the LibriSpeech datasets (https://www.openslr.org/12) test-clean, test-other and train-other-500 that can be found under the `data` folder.

## Experimentation Strategy


The experimentation is performed on subsets of the evaluation datasets by considering two data reduction strategies:
1. selecting different amounts of speakers to be evaluated, and
2. selecting different amounts of utterances per speaker by
    1. selecting different amount of enrollment utterances with a fixed number of trials per speaker, and 
    2. selecting different amount of trial utterances, while keeping the number of enrollments per speaker consistent.

## Scripts

The experimentation scripts for each evaluation strategy are to be found in the `scripts` folder. The scripts:
1. select the pre-trained LDA and PLDA models (under `logs/stargan/train` and `logs/knnvc/train`)
2. select the trial and enrollment folders, which contain LibriSpeech files (may be anonymized, but same name).
3. perform privacy evaluation (computation of SpkId vectors, PLDA mapping and EER computation ) with different speaker population sizes and sample amounts per speaker. 