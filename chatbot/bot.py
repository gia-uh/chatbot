from textwrap import dedent
from lingo import Lingo, LLM, Context, Engine, Message
from lingo.core import Conversation
from .config import load
from .utils import get_db, embed
#from pydantic import BaseModel

# class DocumentChunk(BaseModel):
#     reasoning: str
#     content: str
#     relevant: bool



def build(username: str, conversation: Conversation) -> Lingo:
    config = load()

    # Instantiate our chatbot
    llm=LLM(**config.llm.model_dump())
    
    chatbot = Lingo(
        # Change name and description as desired to
        # fit in the system prompt
        llm=llm,
        # You can also modify the system prompt
        # to completely replace the chatbot personality.
        system_prompt=config.prompts.system.format(username=username, botname="Bot"),
        # We pass the conversation wrapper here
        conversation=conversation,
    )

    # Add skills for the chatbot here
    # Check out Lingo's documentation
    # to learn how to write custom skills:
    # <https://github.com/gia-uh/lingo>

    @chatbot.skill
    async def chat(ctx: Context, engine: Engine):
        """Basic chat skill, just replies normally."""
        print("Entrando a chat skill")
        # Compute reply directly from LLM
        msg = await engine.reply(ctx)

        # Add it to the context (otherwise the bot won't remember its own response)
        ctx.append(msg)

        print(ctx.messages)

    @chatbot.skill
    async def qa(ctx: Context, engine: Engine):
        """Question answering skill, uses context retrieved from documents."""

        # Compute reply from LLM with context documents
        print("Se llamo a QA")
        tool = await engine.equip(ctx)
        print(tool.name)
        result = await engine.invoke(ctx, tool)
        print(f"ToolResult: {result}")
        # Add it to the context (otherwise the bot won't remember its own response)
        response = await engine.reply(
        ctx,
        Message.system(result),
        Message.system("Respond to the user query using the information in query ."),
    )
        ctx.append(response)
        print(ctx.messages)
    
    # @chatbot.tool
    # async def build_context(query: str) -> list[str]:
    #     """Retrieve relevant documents from the database based on the query."""
        
    #     db = get_db()
    #     query_embedding = embed(query)
        
    #     # Perform a similarity search in the database
    #     docs = db.collection("documents").search(query_embedding,top_k=config.conversation.history)
    #     chunks = []
    #     for doc, _ in docs:
    #         chunk = await llm.create(model=DocumentChunk, messages=[
    #             Message.system(dedent("""
    #                 Dado el siguiente fragmento de documento,
    #                 determina si es relevante para la
    #                 siguiente pregunta de usuario, genera
    #                 un breve razonamiento sobre su relevancia (o no)
    #                 y extrae el fragmento de contenido que sea relevant.

    #                 Query: {query}
    #                 """).format(query=query)),
    #             Message.user(doc.body),
    #         ])

    #         print(chunk)
    #         if chunk.relevant:
    #             print("Fragmento relevante encontrado:")
    #             chunks.append(chunk)
    #     print("Chunks relevantes encontrados:", len(chunks))
    #     return chunks
    
    @chatbot.tool
    async def build_context(query: str) -> str:
        """Retrieve relevant documents from the database based on the query."""
        
        db = get_db()
        query_embedding = embed(query)
        # Perform a similarity search in the database
        docs = db.collection("documents").search(query_embedding,top_k=config.conversation.history)
        context = "\n".join([str(doc.body) for doc,_ in docs])
        
        response = await llm.chat(messages=[
            Message.system(dedent("""
                Dado el siguiente fragmento de documento,
                extrae solo la informacion que sea relevante 
                para responder la query siguiente
                Query: {query}
                """).format(query=query)),
            Message.user(context),
        ])

        print(response.content)
        return response.content
        
    # Return the newly created chatbot instance
    return chatbot
