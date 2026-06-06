"""
providers.py — one switchboard for every LLM you talk to in this course.

Why this file exists
--------------------
Every script in this repo gets its model the SAME way:

    from providers import get_chat_model
    model = get_chat_model()          # uses the default provider (Groq)

That means you can swap Groq -> Gemini -> FM Gateway by changing ONE line in
your .env (PROVIDER=...), without editing a single example. The rest of the
course never has to care which company is behind the model.

The three providers
-------------------
  groq    LangChain's native ChatGroq            (default: free + fast)
  gemini  LangChain's native ChatGoogleGenerativeAI
  fm      3DS FM Gateway, which speaks the OpenAI API. We therefore use the
          OpenAI client and just point its base_url at the gateway. This is the
          standard trick for ANY "OpenAI-compatible" endpoint.

You only need a key for the one you actually use.
"""

import os
from dotenv import load_dotenv

# --- Corporate TLS / self-signed certificate support -----------------------
# On many corporate / managed machines, outbound HTTPS is intercepted by a
# proxy that presents the company's OWN root CA. Python's default certificate
# bundle (certifi) doesn't know that CA, so providers like Gemini and the FM
# Gateway fail with:
#     [SSL: CERTIFICATE_VERIFY_FAILED] self signed certificate in certificate chain
# `truststore` makes Python verify against the OS trust store, where your IT
# department already installed the corporate CA — fixing every provider at once,
# WITHOUT disabling certificate verification. It's optional: if it isn't
# installed we just continue (e.g. Groq alone usually works without it).
#   To enable:  pip install truststore
try:
    import truststore
    truststore.inject_into_ssl()
except ModuleNotFoundError:
    pass

# Read .env once, when this module is first imported.
load_dotenv()

# If you don't pass a provider explicitly, this is the one we use.
DEFAULT_PROVIDER = os.getenv("PROVIDER", "groq").lower()

# Sensible default model per provider. Override per-call with get_chat_model(model="...").
# Groq's current line-up (check https://console.groq.com/docs/models) includes
# llama-3.3-70b-versatile, llama-3.1-8b-instant, qwen/qwen3-32b, openai/gpt-oss-120b...
# We default to llama-3.3-70b-versatile: dependable tool-calling, great for demos.
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",   # alt: "qwen/qwen3-32b" (mirrors FM's Qwen)
    "gemini": "gemini-2.0-flash",
    "fm": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}


def get_chat_model(provider: str | None = None, model: str | None = None,
                   temperature: float = 0.0, **kwargs):
    """Return a ready-to-use LangChain chat model.

    Args:
        provider: "groq" | "gemini" | "fm". Defaults to PROVIDER in .env (groq).
        model:    model id. Defaults to a good pick for the chosen provider.
        temperature: 0.0 = as deterministic as the model allows (what you want
                     for tool-using agents and reproducible demos).
        **kwargs: passed straight through to the underlying LangChain class.

    Returns:
        A chat model exposing the same interface regardless of provider:
        .invoke(messages), .stream(...), .bind_tools([...]),
        .with_structured_output(Schema).
    """
    provider = (provider or DEFAULT_PROVIDER).lower()
    model = model or DEFAULT_MODELS.get(provider)

    if provider == "groq":
        # pip install langchain-groq  ;  needs GROQ_API_KEY
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, **kwargs)

    if provider == "gemini":
        # pip install langchain-google-genai  ;  needs GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)

    if provider == "fm":
        # FM Gateway is OpenAI-compatible: use the OpenAI client + a base_url.
        # pip install langchain-openai  ;  needs FM_API_KEY (+ optional FM_BASE_URL)
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


# Run `python providers.py` to confirm your key + provider work end to end.
if __name__ == "__main__":
    print(f"Default provider: {DEFAULT_PROVIDER}")
    print(f"Default model:    {DEFAULT_MODELS.get(DEFAULT_PROVIDER)}")
    model = get_chat_model()
    reply = model.invoke("Say hello in exactly five words.")
    print("Model replied:   ", reply.content)
