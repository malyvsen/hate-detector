# AI-powered hate speech detection
A deployment-ready toxicity detection system.

## Setup
Download `weights.h5` and `config.pkl` from [the Kaggle notebook used to train the model](https://www.kaggle.com/malyvsen/toxicity-detection) and place them in `hate/model`.
To run inference locally, just `import hate` and use `hate.classify()`, passing a list of texts as the first parameter.
To run inference on a server, first use `convert_model.ipynb` to convert the model weights to a servable format, then serve the model with `sudo server/serve.sh`. To query the server, use `hate.classify()` with `server='server_address:port'`.

Alternatively, you can use the provided `cli.py` - run `python cli.py -h` to see the available options.

## Pitfalls
The model does have unintended bias. For example, it seems to consider any sentence including the word "gay" to be offensive. (Oops! This reflects how people communicate online.)