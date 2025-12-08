import logging
from typing import Iterator, cast
from lingo import Message
from pydantic import BaseModel
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

from purely import ensure
from beaver import BeaverDB
from lingo.core import Conversation

from .config import load
from .bot import build

config = load()
db = BeaverDB(config.db)

# Setup logging (Crucial for debugging async apps)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class UserConfig(BaseModel):
    last_message: int = 0


class ConversationHandler(Conversation):
    def __init__(self, db: BeaverDB, user_id: int):
        self.db = db
        self.user_id = user_id
        self.list = db.list(f"conversation:{user_id}", model=Message)
        self.config = db.dict(f"configs", model=UserConfig)

    def append(self, message: Message, /):
        self.list.push(message)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.get_messages()) # type: ignore

    def get_messages(self):
        config = self.configuration

        for msg in self.list[config.last_message:]:
            yield msg

    def __getitem__(self, index: int, /) -> Message:
        return cast(Message, self.list[index])

    @property
    def configuration(self) -> UserConfig:
        return self.config.get(str(self.user_id), UserConfig())

    def clear(self):
        config = self.configuration
        config.last_message = len(self.list)
        self.update(config)

    def update(self, config: UserConfig):
        self.config[str(self.user_id)] = config

    def __len__(self) -> int:
        return len(self.list) - self.configuration.last_message


# --- CORE HANDLERS (The "Upstream" Logic) ---


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Standard welcome message.
    """
    user_id = ensure(update.effective_user).id
    conversation = ConversationHandler(db, user_id)
    conversation.clear()

    username = ensure(update.effective_user).first_name

    await ensure(update.message).reply_text(config.start.format(username=username))


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Standard welcome message.
    """
    username = ensure(update.effective_user).first_name
    user_id = ensure(update.effective_user).id

    conversation = ConversationHandler(db, user_id)
    chatbot = build(username, conversation)

    msg = ensure(update.effective_message).text

    if msg:
        reply = await chatbot.chat(msg)
        await ensure(update.effective_message).reply_text(reply.content)


# --- THE ENGINE STARTUP ---


def start_bot():
    """
    Main entry point. Loads config and starts the polling loop.
    """
    print("🚀 Building Application...")
    application = ApplicationBuilder().token(config.telegram.token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT, chat))

    print("✅ Bot is running! Press Ctrl+C to stop.")
    # run_polling() handles the async event loop automatically
    application.run_polling()


if __name__ == "__main__":
    start_bot()
