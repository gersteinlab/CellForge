import os
import json
import requests
from typing import Dict, Any, Optional, List, Union
try:
    import openai
except ImportError:
    openai = None
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
import logging
import ssl
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LLMInterface:
    def __init__(self):

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.llama_api_key = os.getenv("LLAMA_API_KEY")
        self.qwen_api_key = os.getenv("QWEN_API_KEY")
        self.custom_api_url = os.getenv("CUSTOM_API_URL")
        self.custom_api_key = os.getenv("CUSTOM_API_KEY")


        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))


        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        self.deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.llama_base_url = os.getenv("LLAMA_BASE_URL", "https://api.llama-api.com")
        self.qwen_base_url = os.getenv("QWEN_BASE_URL", "https://api.qwen.ai")


        if self.anthropic_api_key and Anthropic is not None:
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)


        self.session = self._create_session()


        if not hasattr(LLMInterface, '_environment_validated'):
            self._validate_environment()
            LLMInterface._environment_validated = True

    def _create_session(self):
        """Create requests session with retry strategy and SSL configuration"""
        session = requests.Session()


        retry_strategy = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )


        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)


        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE


        session.timeout = (30, 60)


        self.request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

        return session

    def _validate_environment(self):
        """Validate environment variable configuration"""
        configured_providers = []

        if self.openai_api_key:
            configured_providers.append("OpenAI")
        if self.anthropic_api_key:
            configured_providers.append("Anthropic")
        if self.deepseek_api_key:
            configured_providers.append("DeepSeek")
        if self.llama_api_key:
            configured_providers.append("Llama")
        if self.qwen_api_key:
            configured_providers.append("Qwen")
        if self.custom_api_url and self.custom_api_key:
            configured_providers.append("Custom")

        if not configured_providers:
            logger.warning("No LLM API keys configured. Please set at least one API key.")
        else:
            logger.info(f"Configured LLM providers: {', '.join(configured_providers)}")

    def get_config_status(self) -> Dict[str, Any]:
        """Get configuration status"""
        return {
            "openai_configured": bool(self.openai_api_key),
            "anthropic_configured": bool(self.anthropic_api_key),
            "deepseek_configured": bool(self.deepseek_api_key),
            "llama_configured": bool(self.llama_api_key),
            "qwen_configured": bool(self.qwen_api_key),
            "custom_configured": bool(self.custom_api_url and self.custom_api_key),
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    def generate(self,
                prompt: str,
                system_prompt: Optional[str] = None,
                model: Optional[str] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Unified LLM interface supporting multiple providers

        Args:
            prompt: user prompt
            system_prompt: system prompt (optional)
            model: model name (optional)
            temperature: temperature parameter (optional)
            max_tokens: maximum token number (optional)

        Returns:
            generated response content
        """

        model = model or self.model_name
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens

        try:

            if model.startswith("gpt") and self.openai_api_key:
                return self._generate_openai(prompt, system_prompt, model, temperature, max_tokens)
            elif model.startswith("claude") and self.anthropic_api_key:
                return self._generate_anthropic(prompt, system_prompt, model, temperature, max_tokens)
            elif model.startswith("deepseek") and self.deepseek_api_key:
                return self._generate_deepseek(prompt, system_prompt, model, temperature, max_tokens)
            elif model.startswith("llama") and self.llama_api_key:
                return self._generate_llama(prompt, system_prompt, model, temperature, max_tokens)
            elif model.startswith("qwen") and self.qwen_api_key:
                return self._generate_qwen(prompt, system_prompt, model, temperature, max_tokens)
            elif self.custom_api_url and self.custom_api_key:
                return self._generate_custom(prompt, system_prompt, model, temperature, max_tokens)
            else:

                if self.openai_api_key:
                    return self._generate_openai(prompt, system_prompt, model, temperature, max_tokens)
                elif self.anthropic_api_key:
                    return self._generate_anthropic(prompt, system_prompt, model, temperature, max_tokens)
                elif self.deepseek_api_key:
                    return self._generate_deepseek(prompt, system_prompt, model, temperature, max_tokens)
                elif self.llama_api_key:
                    return self._generate_llama(prompt, system_prompt, model, temperature, max_tokens)
                elif self.qwen_api_key:
                    return self._generate_qwen(prompt, system_prompt, model, temperature, max_tokens)
                elif self.custom_api_url and self.custom_api_key:
                    return self._generate_custom(prompt, system_prompt, model, temperature, max_tokens)
                else:
                    raise ValueError(f"Unsupported model: {model} and no fallback provider available")

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise Exception(f"LLM generation failed: {str(e)}")

    def _generate_openai(self, prompt: str, system_prompt: str, model: str,
                        temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        if openai is None:
            raise RuntimeError("OpenAI provider selected but the openai package is not installed")
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})


            client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                timeout=self.request_timeout,
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return {
                "content": response.choices[0].message.content,
                "model": model,
                "provider": "openai"
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _generate_anthropic(self, prompt: str, system_prompt: str, model: str,
                          temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate response using Anthropic API"""
        if Anthropic is None:
            raise RuntimeError(
                "Anthropic provider selected but the anthropic package is not installed"
            )
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return {
                "content": response.content[0].text,
                "model": model,
                "provider": "anthropic"
            }
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _generate_deepseek(self, prompt: str, system_prompt: str, model: str,
                          temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate response using DeepSeek API"""
        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API key not configured")

        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }


        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }


        endpoints = [
            f"{self.deepseek_base_url}/v1/chat/completions",
            f"{self.deepseek_base_url}/chat/completions"
        ]

        for endpoint in endpoints:
            try:
                logger.info(f"Trying DeepSeek endpoint: {endpoint}")
                logger.info(f"Request data: {json.dumps(data, indent=2)}")


                logger.info("Sending request to DeepSeek API...")
                response = self.session.post(endpoint, headers=headers, json=data, verify=False, timeout=self.request_timeout)
                logger.info(f"Received response with status code: {response.status_code}")

                response.raise_for_status()
                result = response.json()
                logger.info("Successfully parsed response from DeepSeek API")

                return {
                    "content": result["choices"][0]["message"]["content"],
                    "model": model,
                    "provider": "deepseek"
                }
            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTP error with endpoint {endpoint}: Status {e.response.status_code}")
                logger.warning(f"Response text: {e.response.text}")

                if e.response.status_code == 422:

                    logger.warning(f"DeepSeek API 422 error for model '{model}': {e.response.text}")

                    fallback_models = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner", "deepseek-r1"]
                    for fallback_model in fallback_models:
                        if fallback_model != model:
                            try:
                                logger.info(f"Trying fallback model: {fallback_model}")
                                data["model"] = fallback_model
                                response = self.session.post(endpoint, headers=headers, json=data, verify=False, timeout=self.request_timeout)
                                response.raise_for_status()
                                result = response.json()
                                logger.info(f"Successfully used fallback model: {fallback_model}")
                                return {
                                    "content": result["choices"][0]["message"]["content"],
                                    "model": fallback_model,
                                    "provider": "deepseek"
                                }
                            except Exception as fallback_error:
                                logger.warning(f"Fallback model {fallback_model} failed: {fallback_error}")
                                continue


                    continue
                else:
                    continue
            except requests.exceptions.SSLError as e:
                logger.warning(f"SSL error with endpoint {endpoint}: {e}")
                logger.warning("This might be due to SSL certificate issues or network configuration")
                continue
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error with endpoint {endpoint}: {e}")
                logger.warning("This might be due to network connectivity issues or firewall blocking")
                continue
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout error with endpoint {endpoint}: {e}")
                logger.warning("Request timed out - this might be due to slow network or API server issues")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error with endpoint {endpoint}: {e}")
                logger.warning(f"Error type: {type(e).__name__}")
                continue


        raise Exception("All DeepSeek API endpoints failed")

    def _generate_llama(self, prompt: str, system_prompt: str, model: str,
                       temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate response using Llama API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.llama_api_key}",
                "Content-Type": "application/json"
            }

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = self.session.post(
                f"{self.llama_base_url}/v1/chat/completions",
                headers=headers,
                json=data,
                verify=False
            )
            response.raise_for_status()
            result = response.json()
            return {
                "content": result["choices"][0]["message"]["content"],
                "model": model,
                "provider": "llama"
            }
        except Exception as e:
            logger.error(f"Llama API error: {e}")
            raise

    def _generate_qwen(self, prompt: str, system_prompt: str, model: str,
                      temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate response using Qwen API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.qwen_api_key}",
                "Content-Type": "application/json"
            }

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = self.session.post(
                f"{self.qwen_base_url}/v1/chat/completions",
                headers=headers,
                json=data,
                verify=False
            )
            response.raise_for_status()
            result = response.json()
            return {
                "content": result["choices"][0]["message"]["content"],
                "model": model,
                "provider": "qwen"
            }
        except Exception as e:
            logger.error(f"Qwen API error: {e}")
            raise

    def _generate_custom(self, prompt: str, system_prompt: str, model: str,
                        temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate response using custom API endpoint"""
        try:
            headers = {
                "Authorization": f"Bearer {self.custom_api_key}",
                "Content-Type": "application/json"
            }

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = self.session.post(
                self.custom_api_url,
                headers=headers,
                json=data,
                verify=False,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            return {
                "content": result["choices"][0]["message"]["content"],
                "model": model,
                "provider": "custom"
            }
        except Exception as e:
            logger.error(f"Custom API error: {e}")
            raise

    def chat_completion(self, messages: List[Dict[str, str]],
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       model: Optional[str] = None) -> str:
        """
        Chat completion interface for compatibility with existing code

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            model: Model name

        Returns:
            Generated response content
        """

        prompt = ""
        system_prompt = None

        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] == "user":
                prompt = message["content"]

        if not prompt:
            raise ValueError("No user message found in messages")

        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.get("content", "")

    def parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse JSON formatted response

        Args:
            response: LLM generated response

        Returns:
            parsed JSON object
        """
        try:
            content = response["content"]

            return json.loads(content)
        except json.JSONDecodeError:

            import re
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                raise ValueError("Response does not contain valid JSON")

    def test_connection(self, provider: str = None) -> Dict[str, Any]:
        """
        Test connection to LLM providers

        Args:
            provider: Specific provider to test (optional)

        Returns:
            Test results
        """
        test_prompt = "Hello, this is a connection test. Please respond with 'Connection successful'."
        results = {}

        providers_to_test = []
        if provider:
            providers_to_test = [provider]
        else:
            if self.openai_api_key:
                providers_to_test.append("openai")
            if self.anthropic_api_key:
                providers_to_test.append("anthropic")
            if self.deepseek_api_key:
                providers_to_test.append("deepseek")
            if self.llama_api_key:
                providers_to_test.append("llama")
            if self.qwen_api_key:
                providers_to_test.append("qwen")
            if self.custom_api_url and self.custom_api_key:
                providers_to_test.append("custom")

        for p in providers_to_test:
            try:
                if p == "openai":
                    response = self._generate_openai(test_prompt, None, "gpt-3.5-turbo", 0.1, 50)
                elif p == "anthropic":
                    response = self._generate_anthropic(test_prompt, None, "claude-3-haiku-20240307", 0.1, 50)
                elif p == "deepseek":
                    response = self._generate_deepseek(test_prompt, None, "deepseek-chat", 0.1, 50)
                elif p == "llama":
                    response = self._generate_llama(test_prompt, None, "llama-2-7b-chat", 0.1, 50)
                elif p == "qwen":
                    response = self._generate_qwen(test_prompt, None, "qwen-turbo", 0.1, 50)
                elif p == "custom":
                    response = self._generate_custom(test_prompt, None, "custom-model", 0.1, 50)

                results[p] = {
                    "status": "success",
                    "response": response["content"][:100] + "..." if len(response["content"]) > 100 else response["content"]
                }
            except Exception as e:
                results[p] = {
                    "status": "failed",
                    "error": str(e)
                }

        return results
