# FontoGen

Generate your very own font with FontoGen. Read more about the project in
my [blog article](https://serce.me/posts/02-10-2023-hey-computer-make-me-a-font).

![screenshot](./img/fontogen.png)

## Installation

```bash
pipenv install
pipenv shell
```

### Training

Please only train the model on open source fonts.
`./train_example.sh` contains an example of the training process.

```bash
# the input fonts
ls ./example/dataset/
# prepare the dataset and start training
./train_example.sh
```

### Inference

The model needs to be re-trained on a large dataset of OFL fonts. If anyone would like to contribute and re-train the model, please reach out and I'll be happy to help you set up the environment.
