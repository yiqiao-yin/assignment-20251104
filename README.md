# Claude Chatbot with MNIST Tools

A powerful Gradio-based chatbot that integrates with Claude AI and includes machine learning tools for MNIST digit classification.

## Features

- **🤖 Claude AI Integration**: Chat with Claude using the Anthropic API
- **🛠️ Tool Calling**: Claude can automatically call tools based on your requests
- **📊 MNIST Logistic Regression**: Train PyTorch models on the MNIST dataset
- **🔒 Secure API Key Management**: Enter your API key securely in the sidebar
- **💬 Natural Conversation**: Ask for ML tasks in natural language

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Get an Anthropic API Key**:
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create an account and generate an API key

3. **Run the Application**:
   ```bash
   python gradio_claude_app.py
   ```

4. **Access the App**:
   - Open your browser to `http://localhost:7860`
   - Enter your API key in the sidebar
   - Start chatting!

## Usage Examples

### Basic Chat
```
User: "Hello! How are you today?"
Claude: "Hello! I'm doing well, thank you for asking..."
```

### Train MNIST Model
```
User: "Can you train a logistic regression model on MNIST with 5 epochs?"
Claude: "I'll train a logistic regression model on MNIST for you..."
[Tool execution happens automatically]
Claude: "I've successfully trained the model! Here are the results..."
```

### Get Model Information
```
User: "Tell me about the MNIST dataset and model architecture"
Claude: "Let me get the technical details for you..."
[Tool execution happens automatically]
Claude: "Here's the information about the MNIST dataset and model..."
```

## Available Tools

### 1. MNIST Logistic Regression Training
- **Function**: `mnist_logistic_regression`
- **Parameters**:
  - `epochs` (int): Number of training epochs (default: 10)
  - `learning_rate` (float): Learning rate (default: 0.01)
  - `batch_size` (int): Batch size (default: 64)

### 2. Model Information
- **Function**: `get_mnist_model_info`
- **Parameters**: None
- **Returns**: Dataset info, model architecture, and performance expectations

## File Structure

```
├── gradio_claude_app.py          # Main Gradio application
├── mnist_logistic_regression.py  # MNIST ML tool implementation
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── data/                        # MNIST dataset (auto-downloaded)
```

## Technical Details

- **Framework**: Gradio for UI, Anthropic for AI, PyTorch for ML
- **Model**: Simple logistic regression (784 → 10 linear layer)
- **Dataset**: MNIST handwritten digits (60k train, 10k test)
- **Expected Accuracy**: 85-92%
- **Training Time**: 1-5 minutes on CPU

## Troubleshooting

### Common Issues

1. **API Key Error**:
   - Make sure your Anthropic API key is valid
   - Check that you have sufficient credits

2. **PyTorch Installation**:
   - Install appropriate PyTorch version for your system
   - Visit [PyTorch.org](https://pytorch.org/) for installation instructions

3. **MNIST Download Issues**:
   - Ensure internet connection for first run
   - Dataset will be cached in `./data/` folder

### Getting Help

- Check the console for detailed error messages
- Ensure all dependencies are installed
- Try running `python mnist_logistic_regression.py` standalone to test ML functionality

## Example Conversations

**Training a Model**:
```
User: "I want to train a quick MNIST model with just 3 epochs"
Claude: "I'll train a logistic regression model on MNIST with 3 epochs for you."

🔧 Using tool: mnist_logistic_regression
Parameters: {"epochs": 3}

📊 Tool Result:
{
  "status": "success",
  "training_results": {
    "final_test_accuracy": 89.45
  }
}

💬 Claude's Analysis:
Great! I've successfully trained a logistic regression model on the MNIST dataset. 
The model achieved 89.45% accuracy on the test set after just 3 epochs, which is 
quite good for such a simple model...
```

**Getting Technical Info**:
```
User: "What can you tell me about the model architecture?"
Claude: "Let me get the detailed information about the MNIST model for you."

[Technical details displayed]

The model uses a simple logistic regression architecture with 784 input features 
(28x28 flattened images) connected directly to 10 output classes...
```

## License

This project is open source. Feel free to modify and extend it for your needs!
