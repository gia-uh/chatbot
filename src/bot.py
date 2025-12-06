from lingo import Lingo, LLM, Context, Engine
from lingo.core import Conversation
from config import load


def build(username: str, conversation: Conversation) -> Lingo:
    config = load()

    # Instantiate our chatbot

    chatbot = Lingo(
        # Change name and description as desired to
        # fit in the system prompt
        llm=LLM(**config.llm.model_dump()),
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

        # Compute reply directly from LLM
        msg = await engine.reply(ctx)

        # Add it to the context (otherwise the bot won't remember its own response)
        ctx.append(msg)

        print(ctx.messages)

    # ... Add your extra skills and tools here
    # ...
    # ...

    # Return the newly created chatbot instance
    return chatbot
