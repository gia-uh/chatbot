# AI Chatbot Template

A batteries-included, modular starter kit for building intelligent Telegram chatbots. This project integrates modern Python tooling to provide a robust foundation for LLM-powered applications with persistent memory and easy extensibility.

## 🚀 How to Use

Follow these steps to get your bot up and running:

### 1. Fork and Clone
Fork this repository to your GitHub account to create your own copy, then clone it locally.

### 2. Install Dependencies

This project uses **uv** for fast package management.

1.  **Install uv** (if you haven't already):
    ```bash
    pip install uv
    ```
2.  **Sync dependencies**:
    ```bash
    uv sync
    ```

### 3. Configuration

The project uses a secure configuration system that separates secrets from logic.

1.  **Create the config file**:
    ```bash
    cp config.example.yaml config.yaml
    ```
2.  **Set up environment variables**:
    Create a `.env` file in the root directory to store your API keys (this file is ignored by git).

    ```env
    TELEGRAM_TOKEN=your_telegram_bot_token
    OPENROUTER_TOKEN=your_llm_provider_key
    EMBEDDING_TOKEN=your_embedding_provider_key
    ```
3.  **Review `config.yaml`**:
    Ensure the `token` and `api_key` fields reference your environment variables (e.g., `${TELEGRAM_TOKEN}`). You can also adjust the LLM model and system prompt here.

### 4. Run the Bot

You can start the application using the included Makefile or directly via uv:

```bash
# Option A: Using Make
make run

# Option B: Direct
uv run src/app.py
```

### 5. Customization

To add custom logic or new capabilities to your bot:

  * **Edit `src/bot.py`**: This is where the AI logic lives. You can define new "skills" using the `@chatbot.skill` decorator provided by `lingo-ai`.
  * **Edit `src/config.py`**: Update the Pydantic models if you need to add new configuration sections.

## 🏗 Architecture Overview

This project is built on a modular stack designed for maintainability and scale:

  * **🤖 Python Telegram Bot**:
    Handles the networking layer. It receives updates from Telegram and manages the polling loop in `src/app.py`.

  * **🧠 Lingo AI**:
    Found in `src/bot.py`, this library manages the Agentic logic. It handles the System Prompt, wraps the LLM connection, and defines "Skills" (functions the bot can execute).

  * **🗄️ Beaver DB**:
    A lightweight, local database used to store conversation history.

      * When a user chats, `src/app.py` retrieves their history from BeaverDB using a unique key (`conversation:{user_id}`).
      * This allows the bot to remember context across different sessions.

  * **⚙️ Purely & Pydantic**:

      * **Purely** provides utility functions for validation (like `ensure`).
      * **Pydantic** (`src/config.py`) ensures strictly typed configuration, preventing runtime errors due to missing or malformed settings.

## 📄 License

This project is licensed under the **MIT License** - Copyright (c) 2025 Grupo de Inteligencia Artificial (GIA-UH).
