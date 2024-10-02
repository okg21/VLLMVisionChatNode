# VLLMVisionChatNode
A custom node for ComfyUI that enables interaction with vision-language models hosted on a vLLM server.

## Installation

1. Ensure you have ComfyUI installed and set up.
2. Copy the `vllm_vision_chat_node.py` file to your ComfyUI custom nodes directory.
3. Install the required dependencies:

   ```
   pip install torch Pillow openai numpy
   ```

4. Restart ComfyUI or reload custom nodes.

## Usage

After installation, the "vLLM Vision Chat" node will be available in the ComfyUI interface under the "AI/ML" category.

### Inputs

* **Required**:
   * `prompt` (STRING): The text prompt to send to the model.
   * `model_url` (STRING): The URL of the vLLM server (default: "http://localhost:8000/v1").
   * `model_name` (STRING): The name of the model to use (default: "gpt-4-vision-preview").

* **Optional**:
   * `image` (IMAGE): An image input (can be connected from other ComfyUI nodes).
   * `temperature` (FLOAT): Controls randomness in generation (default: 0.7, range: 0.0 to 2.0).
   * `max_tokens` (INT): Maximum number of tokens to generate (default: 256, range: 1 to 4096).

### Output

* `STRING`: The generated response from the model.
