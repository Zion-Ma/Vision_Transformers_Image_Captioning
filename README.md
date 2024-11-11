# Vision_Transformers_Image_Captioning
Using Tranformers and RNN to generate image captions

# Transformer and RNN Model Project

## Overview

This project provides a collection of Python and JavaScript files for building, training, and visualizing transformer and RNN models. Key components include data preprocessing, model implementation, and visualization. The project is structured to support various machine learning tasks such as natural language processing and sequence modeling.

## Project Structure

- **transformer.py**: Contains the core implementation of a transformer model.
- **model.py**: General model functions and utility methods shared across different models.
- **preprocessing.py**: Data preprocessing pipeline to prepare input data for training.
- **decoder.py**: Decoder component for sequence generation tasks in the transformer model.
- **rnn.sh**: Shell script for training an RNN model, including an option to increase the number of epochs for enhanced training performance&#8203;:contentReference[oaicite:0]{index=0}.
- **head_view.js**: JavaScript file for D3-based visualization of the attention mechanism in transformers. It dynamically renders visualizations with configurable settings for layers, heads, and attention filtering&#8203;:contentReference[oaicite:1]{index=1}.

## Requirements

- **Python 3.6+**
- **D3.js**: For visualizations in `head_view.js`.
- **jQuery**: Required for visualization dependencies in `head_view.js`.

## Setup

1. **Install Dependencies**:
   - Install required Python packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure D3.js and jQuery are included in the project’s HTML file for the visualizations to render correctly.

2. **Running the Transformer Model**:
   - Configure model parameters in `transformer.py`.
   - Run the model training:
     ```bash
     python transformer.py
     ```

3. **Training the RNN Model**:
   - Modify `rnn.sh` to specify data path and model checkpoint path.
   - Run the training script:
     ```bash
     bash rnn.sh
     ```

4. **Visualization**:
   - Use `head_view.js` to visualize the attention mechanism in the transformer model. This script should be integrated into a Jupyter notebook or an HTML file with configured parameters.

## TODO

- Increase epochs for RNN training when experimenting with larger datasets&#8203;:contentReference[oaicite:2]{index=2}.
- Enhance the transformer model to support additional layers and heads.
- Improve attention visualization to include additional filtering options.

## License

This project is licensed under the MIT License.
