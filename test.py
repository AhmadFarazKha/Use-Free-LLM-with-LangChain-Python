from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from decouple import config

# Get API token from environment
HUGGINGFACEHUB_API_TOKEN = config('HUGGINGFACEHUB_API_TOKEN')

# Define the model and endpoint
repo_id = "mistralai/Mistral-7B-v0.1"
endpoint_url = f"https://api-inference.huggingface.co/models/{repo_id}"

# Create prompt template
prompt_template = "<s>[INST]Write short answer of {question}[/INST]</s>"
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

# Format the prompt
formatted_prompt = prompt.format(question="What is the Chairman Pakistan tehreek e insaf?")

# Initialize the HuggingFace endpoint with correct parameter structure
llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    task="text-generation",
    temperature=0.7,
    max_new_tokens=200,
    repetition_penalty=1.1
)

# Generate response
try:
    llm_output = llm.invoke(formatted_prompt)
    print("Generated Response:", llm_output)
except Exception as e:
    print(f"An error occurred: {str(e)}")