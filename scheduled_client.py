import asyncio
import os
import random
import itertools
from openai import RateLimitError, APIStatusError
from langchain_openai import ChatOpenAI
from api_gateway import global_gateway
from langsmith import traceable

# Key rotator for multi-key load balancing and failover
_key_pool = []
_key_cycle = None

def _get_key_pool():
    global _key_pool, _key_cycle
    if not _key_pool:
        keys = []
        for env_var in ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY"]:
            k = os.environ.get(env_var, "").strip()
            if k and k not in keys:
                keys.append(k)
        _key_pool = keys or [""]
        _key_cycle = itertools.cycle(_key_pool)
    return _key_pool

def get_next_groq_api_key() -> str:
    pool = _get_key_pool()
    if not pool:
        return ""
    global _key_cycle
    if _key_cycle is None:
        _key_cycle = itertools.cycle(pool)
    return next(_key_cycle)

def is_rate_or_capacity_error(exc: Exception) -> bool:
    """Detects HTTP 429, 413, or Groq TPM / token rate-limit rejections."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        if getattr(exc, "status_code", None) in (413, 429):
            return True
    msg = str(exc).lower()
    return any(phrase in msg for phrase in ["rate_limit_exceeded", "tokens per minute", "tpm", "request too large", "429", "413", "service tier"])


class ScheduledChatOpenAI(ChatOpenAI):
    """
    LangChain ChatOpenAI client with APIGateway admission control,
    multi-key Groq rotation, and automatic seamless fallback to local Ollama.
    """
    agent_priority: int = 5
    provider_mode: str = "auto"
    _fallback_ollama_client: ChatOpenAI = None

    def __init__(self, *args, **kwargs):
        provider = kwargs.pop("provider_mode", "auto")
        super().__init__(*args, **kwargs)
        self.provider_mode = provider

        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
        self._fallback_ollama_client = ChatOpenAI(
            model=ollama_model,
            api_key="ollama",
            base_url=ollama_url,
            temperature=self.temperature,
            timeout=240,
        )

    def _rotate_underlying_key(self, next_key: str):
        self.openai_api_key = next_key
        if hasattr(self, "root_async_client") and self.root_async_client is not None:
            self.root_async_client.api_key = next_key
        if hasattr(self, "root_client") and self.root_client is not None:
            self.root_client.api_key = next_key

    async def _execute_ollama_fallback(self, func_name: str, *args, **kwargs):
        print(f" 🦙 [OLLAMA FALLBACK] Engaging local Ollama ({self._fallback_ollama_client.model}) — Zero rate limits, 100% reliable...")
        fn = getattr(self._fallback_ollama_client, func_name)
        return await fn(*args, **kwargs)

    @traceable(name="Scheduled Model Generate", run_type="llm")
    async def _agenerate(self, *args, **kwargs):
        # Direct local Ollama mode
        if self.provider_mode == "ollama":
            return await self._fallback_ollama_client._agenerate(*args, **kwargs)

        max_retries = 3
        pool = _get_key_pool()
        last_error = None

        for attempt in range(max_retries):
            await global_gateway.wait_for_admission(priority=self.agent_priority)
            async with global_gateway.concurrency_semaphore:
                try:
                    return await super()._agenerate(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if not is_rate_or_capacity_error(e):
                        raise e

                    # Rotate key on first attempt if multiple keys exist
                    if attempt == 0 and len(pool) > 1:
                        next_key = get_next_groq_api_key()
                        self._rotate_underlying_key(next_key)
                        print(f" [CLIENT] Groq capacity hit ({type(e).__name__}). Swapping to secondary Groq key (attempt {attempt+1})...")
                        await asyncio.sleep(1.0)
                        continue

                    # If rate limits persist, seamlessly switch to Ollama
                    enable_fallback = os.environ.get("ENABLE_OLLAMA_FALLBACK", "true").lower() in ("true", "1", "yes")
                    if enable_fallback and self.provider_mode != "groq":
                        try:
                            return await self._execute_ollama_fallback("_agenerate", *args, **kwargs)
                        except Exception as fallback_err:
                            print(f" [FALLBACK ERROR]: Ollama fallback failed: {fallback_err}")
                            raise last_error

                    base_delay = 2 ** (attempt + 1)
                    jitter = random.uniform(1.0, 2.0)
                    total_delay = base_delay + jitter
                    print(f" [CLIENT] Groq rate limit hit. Backing off for {total_delay:.2f}s...")
                    await asyncio.sleep(total_delay)

        # Final safety check: Fall back to Ollama
        enable_fallback = os.environ.get("ENABLE_OLLAMA_FALLBACK", "true").lower() in ("true", "1", "yes")
        if enable_fallback and self.provider_mode != "groq":
            try:
                return await self._execute_ollama_fallback("_agenerate", *args, **kwargs)
            except Exception as fallback_err:
                print(f" [FALLBACK ERROR]: Ollama fallback failed: {fallback_err}")

        if last_error:
            raise last_error

    @traceable(name="Scheduled Model Stream", run_type="llm")
    async def _astream(self, *args, **kwargs):
        if self.provider_mode == "ollama":
            async for chunk in self._fallback_ollama_client._astream(*args, **kwargs):
                yield chunk
            return

        max_retries = 3
        pool = _get_key_pool()
        last_error = None

        for attempt in range(max_retries):
            await global_gateway.wait_for_admission(priority=self.agent_priority)
            async with global_gateway.concurrency_semaphore:
                try:
                    async for chunk in super()._astream(*args, **kwargs):
                        yield chunk
                    return
                except Exception as e:
                    last_error = e
                    if not is_rate_or_capacity_error(e):
                        raise e

                    if attempt == 0 and len(pool) > 1:
                        next_key = get_next_groq_api_key()
                        self._rotate_underlying_key(next_key)
                        print(f" [CLIENT] Groq stream capacity limit hit. Swapping to secondary key (attempt {attempt+1})...")
                        await asyncio.sleep(1.0)
                        continue

                    enable_fallback = os.environ.get("ENABLE_OLLAMA_FALLBACK", "true").lower() in ("true", "1", "yes")
                    if enable_fallback and self.provider_mode != "groq":
                        try:
                            print(f" 🦙 [OLLAMA FALLBACK STREAM] Engaging local Ollama ({self._fallback_ollama_client.model})...")
                            async for chunk in self._fallback_ollama_client._astream(*args, **kwargs):
                                yield chunk
                            return
                        except Exception as fallback_err:
                            print(f" [FALLBACK ERROR]: Ollama stream failed: {fallback_err}")
                            raise last_error

                    base_delay = 2 ** (attempt + 1)
                    jitter = random.uniform(1.0, 2.0)
                    total_delay = base_delay + jitter
                    print(f" [CLIENT] Groq stream rate limit. Backing off for {total_delay:.2f}s...")
                    await asyncio.sleep(total_delay)

        enable_fallback = os.environ.get("ENABLE_OLLAMA_FALLBACK", "true").lower() in ("true", "1", "yes")
        if enable_fallback and self.provider_mode != "groq":
            try:
                print(f" 🦙 [OLLAMA FALLBACK STREAM] Engaging local Ollama ({self._fallback_ollama_client.model})...")
                async for chunk in self._fallback_ollama_client._astream(*args, **kwargs):
                    yield chunk
                return
            except Exception as fallback_err:
                print(f" [FALLBACK ERROR]: Ollama stream failed: {fallback_err}")

        if last_error:
            raise last_error


def get_scheduled_chat_client(
    priority: int = 5,
    temperature: float = 0.0,
    provider: str = None
) -> ScheduledChatOpenAI:
    provider_mode = (provider or os.environ.get("PRIMARY_PROVIDER", "auto")).lower()
    
    if provider_mode == "ollama":
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
        client = ScheduledChatOpenAI(
            model=ollama_model,
            api_key="ollama",
            base_url=ollama_url,
            temperature=temperature,
            timeout=240,
            provider_mode="ollama"
        )
        client.agent_priority = priority
        return client

    api_key = get_next_groq_api_key()
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    max_tokens = int(os.environ.get("GROQ_MAX_TOKENS", "1600"))

    client = ScheduledChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
        timeout=120,
        provider_mode=provider_mode
    )
    client.agent_priority = priority
    return client
