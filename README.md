
# Image Caption Generation



Image Caption Generation is a Python project that uses deep learning techniques to generate descriptive captions for images. By providing an image path to the program, users can obtain meaningful textual descriptions for the content of the image.




## Description
The objective of the project is to predict the captions for the input image. The dataset consists of 8k images and 5 captions for each image. The features are extracted from both the image and the text captions for input. The features will be concatenated to predict the next word of the caption. CNN is used for image and LSTM is used for text.

[flickr8k dataset link](https://www.kaggle.com/adityajn105/flickr8k)


## Libraries

1. numpy
2. matplotlib
3. keras
4. tensorflow
5. os
6. pickle
7. tqdm
## Neural Network

1. VGG16 Network
2. CNN-LSTM Network
## Folder
Place files and folder at one place.

    -->your project Folder
        -->icg (1).ipynb        (program)
        -->Images               (kaggle data set)
        -->captions.txt         (kaggle data set)
        -->features (1).pkl     (During Execution)
        -->max_length.pkl       (During Execution)
        -->modeltrain1.h5       (During Execution)(our own train Model)
        -->tokenizer.pkl        (During Execution)

## Contributing

Contributing
Contributions are welcome! If you find bugs or have ideas for improvements, please open an issue. For code contributions, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and submit a pull request.


## Acknowledgements

The image caption generation model is based on the work of
 - [ashwintechguy](https://github.com/aswintechguy/Deep-Learning-Projects/tree/main/Image%20Caption%20Generator%20-%20Flickr%20Dataset)



## Authors

- [@kunalSMehra](https://www.github.com/kunalSMehra)

