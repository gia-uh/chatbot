from openai import OpenAI

class Embedding:
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model: str,
                 #embedding_dimension: int,
    ) -> None:
        
        # Load configuration data
        self.model = model
        #self.embedding_dimension = embedding_dimension
        # LLM Tools
        self.client = OpenAI(
                             base_url=base_url,
                             api_key=api_key,
                            )
        

    def create(self,documents: str | list[str]):
        """Generate embeddings for the given documents using OpenAI's API.

        Args:
            documents (List[str]): The list of documents to get the embeddings

        Returns:
            ndarray: The embeddings of the documents
        """
        # Getting the embeddings
        
        try:      
            response =  self.client.embeddings.create(
                        input=documents,
                        model=self.model,
                        #dimensions=self.embedding_dimension,
                        )
            return [v.embedding for v in response.data] 
                
        except Exception as e:
            print(f"Error in get_embedding: {e}")
            return None
        
