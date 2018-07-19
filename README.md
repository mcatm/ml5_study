# ml5.js

## ml5.jsとは

> Friendly Machine Learning for the Web.

WEBに最適な機械学習ライブラリ

https://ml5js.org/

> ml5.js aims to make machine learning approachable for a broad audience of artists, creative coders, and students. The library provides access to machine learning algorithms and models in the browser, building on top of TensorFlow.js with no other external dependencies.
>
> The library is supported by code examples, tutorials, and sample datasets with an emphasis on ethical computing. Bias in data, stereotypical harms, and responsible crowdsourcing are part of the documentation around data collection and usage.

------

## できること

### `imageClassifier()`

> You can use neural networks to recognize the content of images. ml5.imageClassifier() is a method to create an object that classifies an image using a pre-trained model.
>
> It should be noted that the pre-trained model provided by the example below was trained on a database of approximately 15 million images (ImageNet). The ml5 library accesses this model from the cloud. What the algorithm labels an image is entirely dependent on that training data -- what is included, excluded, and how those images are labeled (or mislabeled).

画像の内容を把握するためにニューラルネットワークを使うことが出来ます。`ml5.imageClassifier()`は、事前にトレーニングされたモデルを使用して、画像を分類するオブジェクトを生成します。

- DEMO
  - [Image Classification · ml5js](https://ml5js.org/docs/image-classification-example)
  - [Video Classification · ml5js](https://ml5js.org/docs/video-classification-example)
- [imageClassifier() · ml5js](https://ml5js.org/docs/ImageClassifier)

### `featureExtractor()`

> You can use neural networks to recognize the content of images. Most of the times you will be using a model trained on a large dataset for this. But you can also use part of a pre-trained model that has already learned some features about the dataset and 'retrain' or 'reuse' it for a new custom task. This is known as Transfer Learning.
>
> This class allows you to extract features from pre-trained models and retrain them with new types of data.

おおよそ、巨大なデータセットの中のトレーニングされたモデルを使うことになると思うのですが、それは一部使用しつつ、「再トレーニング」「再利用」することも可能になります。これは「Transfer Learning」として知られています。

このクラスは、事前トレーニングされたモデルからデータを抜粋して、新しいデータにトレーニングし直すことが出来ます。

- DEMO
  - [Classifier with Feature Extractor · ml5js](https://ml5js.org/docs/custom-classifier)
  - [Regression with Feature Extractor · ml5js](https://ml5js.org/docs/custom-regression)
- [featureExtractor() · ml5js](https://ml5js.org/docs/FeatureExtractor)

### `LSTMGenerator()`

> LSTMs (Long Short Term Memory networks) are a type of Neural Network architecture useful for working with sequential data (like characters in text or the musical notes of a song) where the order of the that sequence matters. This class allows you run a model pre-trained on a body of text to generate new text.
>
> You can train your own models using this tutorial or use this set of pretrained models. More on this soon!

- DEMO
  - [Text Generation with LSTM · ml5js](https://ml5js.org/docs/lstm-example)
  - [Interactive Text Generation LSTM · ml5js](https://ml5js.org/docs/lstm-interactive-example)
- [LSTMGenerator() · ml5js](https://ml5js.org/docs/LSTMGenerator)

### `pitchDetection()`

> A pitch detection algorithm is a way of estimating the pitch or fundamental frequency of an audio signal. This method allows to use a pre-trained machine learning pitch detection model to estimate the pitch of sound file.
>
> Right now ml5.js only support the CREPE model. This model is a direct port of github.com/marl/crepe and only support direct input from the browser microphone.

- [pitchDetection() · ml5js](https://ml5js.org/docs/PitchDetection)

### `poseNet()`

> PoseNet is a machine learning model that allows for Real-time Human Pose Estimation.
>
> PoseNet can be used to estimate either a single pose or multiple poses, meaning there is a version of the algorithm that can detect only one person in an image/video and one version that can detect multiple persons in an image/video.

- DEMO
  - [PoseNet with Webcam · ml5js](https://ml5js.org/docs/posenet-webcam)
- [poseNet() · ml5js](https://ml5js.org/docs/PoseNet)

### `styleTransfer()`

> Style Transfer is a machine learning technique that allows to transfer the style of one image into another one. This is a two step process, first you need to train a model on one particular style and then you can apply this style to another image.
>
> You can train your own images following this tutorial. More on this soon!

- DEMO
  - [Style Transfer · ml5js](https://ml5js.org/docs/style-transfer-image-example)
  - [Style Transfer with Webcam · ml5js](https://ml5js.org/docs/style-transfer-webcam-example)
- [styleTransfer() · ml5js](https://ml5js.org/docs/StyleTransfer)

### `word2vec()`

> Word2vec is a group of related models that are used to produce word embeddings. This method allows you to perform vector operations on a given set of input vectors.
>
> You can use the word models we provide, trained on a corpus of english words (watch out for bias data!), or you can train your own vector models following this tutorial. More of this soon!

- DEMO
  - [Word2Vec · ml5js](https://ml5js.org/docs/word2vec-example)
- [word2vec() · ml5js](https://ml5js.org/docs/Word2vec)

### `YOLO()`

> Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections. We use a totally different approach. We apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

- DEMO
  - [YOLO with Webcam · ml5js](https://ml5js.org/docs/yolo-webcam)
- [YOLO() · ml5js](https://ml5js.org/docs/YOLO)

------

## 用語

- Ml5.js
- MobileNet
  - Pre-trainedなデータを提供
  - [Google、TensorFlow向けモバイルファーストコンピュータ画像モデルをオープンソース化 - THE BRIDGE（ザ・ブリッジ）