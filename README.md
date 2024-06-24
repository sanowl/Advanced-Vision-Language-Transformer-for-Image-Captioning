# Advanced Vision-Language Transformer for Image Captioning

This project implements an advanced Vision-Language Transformer model for generating image captions. It utilizes state-of-the-art techniques in computer vision and natural language processing to create high-quality image descriptions.

## Features

- Advanced multimodal transformer architecture
- ResNet-152 image encoder
- BERT tokenizer for text processing
- Positional encoding for improved sequence learning
- Supervised Contrastive Learning for better alignment of image and text representations
- Mixed precision training for improved performance
- Cosine Annealing learning rate scheduler
- Comprehensive data preprocessing and augmentation

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- transformers
- Pillow
- numpy
- tqdm

You can install the required packages using:

```
pip install torch torchvision transformers Pillow numpy tqdm
```

## Usage

### Training

1. Prepare your dataset:
   - Organize your images in a directory
   - Create JSON files for training and validation captions in the format:
     ```json
     {
       "image1.jpg": ["caption1", "caption2", ...],
       "image2.jpg": ["caption1", "caption2", ...],
       ...
     }
     ```

2. Run the training script:

```bash
python vision_language_transformer.py \
    --image_dir /path/to/images \
    --train_captions /path/to/train_captions.json \
    --val_captions /path/to/val_captions.json \
    --model_path /path/to/save/model.pth \
    --batch_size 32 \
    --num_workers 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --epochs 30
```

### Inference

To generate captions for new images, you can use the following script:

```python
import torch
from torchvision import transforms
from PIL import Image
from vision_language_transformer import VisionLanguageTransformer, BertTokenizer

def generate_caption(model, image_path, tokenizer, max_length=50):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(model.device)
    
    with torch.no_grad():
        output = torch.zeros(1, 1).long().to(model.device)
        for _ in range(max_length):
            predictions = model(image, output)
            predicted_id = predictions[:, -1:, :].argmax(dim=-1)
            output = torch.cat([output, predicted_id], dim=-1)
            
            if predicted_id.item() == tokenizer.sep_token_id:
                break
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionLanguageTransformer(vocab_size=30522).to(device)  # BERT vocab size
model.load_state_dict(torch.load("path/to/trained/model.pth"))

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Generate caption for a new image
image_path = "path/to/your/image.jpg"
caption = generate_caption(model, image_path, tokenizer)
print(f"Generated caption: {caption}")
```

## Model Architecture

The Vision-Language Transformer consists of the following components:

1. Image Encoder: A pre-trained ResNet-152 model (excluding the final classification layer)
2. Image Feature Projection: Linear projection of image features to match the transformer dimension
3. Text Embedding: BERT tokenizer and embedding layer
4. Positional Encoding: Sinusoidal positional encoding for sequence information
5. Transformer: A stack of transformer encoder and decoder layers
6. Output Layer: Linear layer for generating caption tokens

The model incorporates a supervised contrastive learning mechanism to improve the alignment between image and text representations.

## Training Process

The training process includes the following key steps:

1. Data loading and preprocessing using custom `ImageCaptioningDataset`
2. Mixed precision training with `GradScaler` for improved performance
3. Combination of cross-entropy loss for caption generation and supervised contrastive loss for image-text alignment
4. Cosine annealing learning rate scheduling
5. Model checkpointing based on validation loss

## Customization

You can customize various aspects of the model and training process by modifying the command-line arguments or the model architecture in the code. Some key areas for customization include:

- Transformer architecture (number of layers, heads, dimension)
- Learning rate and optimization parameters
- Loss function weights
- Data augmentation techniques

## License

This project is licensed under the MIT License.

## Acknowledgments

- The BERT implementation is based on the Hugging Face Transformers library.
- The ResNet implementation is from torchvision.
- The supervised contrastive loss is inspired by the paper "Supervised Contrastive Learning" by Khosla et al.