import torch
from PIL import Image
import base64
import io
from openai import OpenAI
import numpy as np
from io import BytesIO

class VLLMVisionChatNode:
    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model_name": ("STRING", {"default": "gpt-4-vision-preview"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "chat_with_vllm"
    CATEGORY = "AI/ML"

    def chat_with_vllm(self, prompt, model_url, model_name, image=None, temperature=0.7, max_tokens=256):
        if self.client is None:
            self.client = OpenAI(api_key="EMPTY", base_url=model_url)

        content = [{"type": "text", "text": prompt}]

        if image is not None:
            try:
                # Debug: Print image shape and type
                print(f"Input image shape: {image.shape}, dtype: {image.dtype}")

                # Handle the image format from ComfyUI's "Load Image" node
                if len(image.shape) == 4 and image.shape[0] == 1:
                    # Remove the batch dimension
                    image = image.squeeze(0)

                # Ensure the image is in the format [height, width, channels]
                if image.shape[2] != 3:
                    image = image.permute(1, 2, 0)

                # Convert to numpy and ensure it's in the range 0-255
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)

                # Debug: Print numpy array shape and type
                print(f"Processed numpy array shape: {image_np.shape}, dtype: {image_np.dtype}")

                # Convert to PIL Image
                pil_image = Image.fromarray(image_np)

                # Debug: Print PIL Image size and mode
                print(f"PIL Image size: {pil_image.size}, mode: {pil_image.mode}")

                # Convert the image to base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()

                content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    })
            except Exception as e:
                return (f"Error processing image: {str(e)}",)

        messages = [{"role": "user", "content": content}]

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return (response.choices[0].message.content,)
        except Exception as e:
            return (f"Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "VLLMVisionChatNode": VLLMVisionChatNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLLMVisionChatNode": "vLLM Vision Chat"
}