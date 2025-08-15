from src.main import boto3
from src.main import Config
from src.main import ClientError

config = Config(
        retries = dict(

            max_attempts = 10,
            total_max_attempts = 25,
        )
    )

bedrock_client = boto3.client("bedrock-runtime", config=config)

def convert_to_dbl_qt(input: str) -> str:
  return input.replace("'", '"')

### Select models
model_id = [ "us.amazon.nova-pro-v1:0"]

model_id_c35 = "anthropic.claude-3-5-sonnet-20240620-v1:0" # Due to model access restriction #'anthropic.claude-3-5-sonnet-20240620-v1:0'
model_id_mistral_large = 'mistral.mistral-large-2402-v1:0'
model_id_novapro = "us.amazon.nova-pro-v1:0"
model_id_novalite = "us.amazon.nova-lite-v1:0"


# Choose multiple models for different purpose to deversify and avoid potential bias
llm = get_llm(model_id)
llm_claude35 = get_llm(model_id_c35)
llm_mistral = get_llm(model_id_mistral_large)
llm_novapro = get_llm(model_id_novapro)
llm_novalite = get_llm(model_id_novalite)