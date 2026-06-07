import os
from dotenv import load_dotenv

#https security
try:
    import truststore
    truststore.inject_into_ssl()
except ModuleNotFoundError:
    pass

# Read .env once
load_dotenv()
DEFAULT_PROVIDER = os.getenv("PROVIDER", "groq").lower()


DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",   
    "fm": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
}


def get_chat_model(provider: str | None = None, model: str | None = None,
                   temperature: float = 0.0, **kwargs):
    """Return a ready-to-use LangChain chat model.

    Args:
        provider: "groq" | "fm". Defaults to PROVIDER in .env (groq).
        model:    model id. Defaults to a good pick for the chosen provider.
        temperature: 0.0
        **kwargs: passed straight through to the underlying LangChain class.
    """
    provider = (provider or DEFAULT_PROVIDER).lower()
    model = model or DEFAULT_MODELS.get(provider)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, **kwargs)

    if provider == "fm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("FM_API_KEY"),
            base_url=os.getenv("FM_BASE_URL", "https://fmgateway.proxem.dsone.3ds.com/v1"),
            **kwargs,
        )

    raise ValueError(
        f"Unknown provider {provider!r}. Use one of: {', '.join(DEFAULT_MODELS)}."
    )



if __name__ == "__main__":
    print(f"Default provider: {DEFAULT_PROVIDER}")
    print(f"Default model:    {DEFAULT_MODELS.get(DEFAULT_PROVIDER)}")
    model = get_chat_model()
    reply = model.invoke("Say hello in exactly five words.")
    print("Model replied:   ", reply.content)
