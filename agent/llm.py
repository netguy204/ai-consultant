"""provide completions via vertex ai"""
# Initialize Vertex AI
import vertexai


import base64
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
import sys


PROJECT_ID = "expeng-k8s-prototype"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


async def agenerate(prompt):
    """yield chunks that complete the prompt"""
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
    responses = model.generate_content(
      [prompt],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=True,
    )

    for response in responses:
        yield response.text


async def main():
    """main function"""
    import sys
    async for chunk in agenerate("tell 3 a dad jokes"):
        sys.stdout.write(chunk)
        sys.stdout.flush()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())