import json
from concurrent.futures import ThreadPoolExecutor
import tqdm

from bedrock.client import BedrockClient
from botocore.exceptions import ClientError

from superduper import Model, logging
from superduper.components.vector_index import sqlvector

from dataclasses import dataclass

@dataclass
class BedrockCohereEnglishEmbeddings(Model):
    """ A class to generate text embeddings using the Cohere Embed English model. """
    # References:
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
    # https://docs.cohere.com/reference/embed

    signature: str = 'singleton'
    foundation_model: str = None
    batch_size: int = 100
    aws_region: str = None

    def __post_init__(self, db, artifacts, example):
        if not self.aws_region or not self.foundation_model:
            raise ValueError("aws_region and foundation_model must be provided and cannot be None.")
        self.bedrock_client = BedrockClient(
            region_name=self.aws_region
        )._get_bedrock_client()
        return super().__post_init__(db, artifacts, example)

    def generate_text_embeddings(self, body):
        """
        Generate text embedding by using the Cohere Embed model.
        Args:
            model_id (str): The model ID to use.
            body (str) : The request body to use.
        Returns:
            dict: The response from the model.
        """

        accept = '*/*'
        content_type = 'application/json'

        response = self.bedrock_client.invoke_model(
            body=body,
            modelId=self.foundation_model,
            accept=accept,
            contentType=content_type
        )

        return response

    def predict(self, text: str):
        """ Predict text embeddings based on the input text. 

        Args:
            text (str): The input text to generate embeddings for.

        Returns:
            list: The text embeddings generated by the model.
        """

        input_type = "search_document"
        embedding_types = ["float"]

        try:
            body = json.dumps({
                "texts": [text],
                "input_type": input_type,
                "embedding_types": embedding_types}
            )
            response = self.generate_text_embeddings(body=body)
            # Extract the response embeddings
            response_embeddings = json.loads(response.get('body').read())[
                "embeddings"]["float"][0]

            return response_embeddings
        except ClientError as err:
            message = err.response["Error"]["Message"]
            logging.error("A client error occurred: %s", message)

    def predict_batches(self, texts: list, num_threads=10):
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm.tqdm(executor.map(
                self.predict, texts), total=len(texts)))
        return results
