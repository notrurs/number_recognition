# Number Recognition
Simple program, which can predict a number on the image

## Getting Started

### Installing
The game is alpha state, but if you want to play, you need to do these steps below.

Step 1. Clone this repo

Step 2. Install requirements:
```
pip install -r requirements.txt
```

Step 3. Move to `number_recognition` dir:
```
cd number_recognition
```

### Usage

You can draw 28x28 image with number on a background like MNIST.

Be sure to give a name to the file `pred_data.png` 

Then start the program:
```
python main.py
```
If you want your own trained model, delete `model.model` and program will train a new model by itself

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.