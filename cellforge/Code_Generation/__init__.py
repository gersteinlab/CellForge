"""
Code Generation Module for CellForge
Automatically generates code based on research plans using OpenHands
"""

import json
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import os
import shutil

logger = logging.getLogger(__name__)

class OpenHandsCodeGenerator:
    """Code generator using OpenHands Docker container"""
    
    def __init__(self, openhands_path: Optional[str] = None):
        """
        Initialize OpenHands code generator
        
        Args:
            openhands_path: Path to OpenHands directory (default: cellforge/Code_Generation/OpenHands)
        """
        if openhands_path is None:
            # Default to the OpenHands directory in this module
            openhands_path = Path(__file__).parent / "OpenHands"
        self.openhands_path = Path(openhands_path)
        
        # Check if OpenHands directory exists
        if not self.openhands_path.exists():
            raise FileNotFoundError(f"OpenHands directory not found: {self.openhands_path}")
        
        # Check if docker-compose.yml exists
        docker_compose_file = self.openhands_path / "docker-compose.yml"
        if not docker_compose_file.exists():
            raise FileNotFoundError(f"docker-compose.yml not found in: {self.openhands_path}")
        
        # Load environment variables from global .env file
        self._load_env_config()
        
        # Fixed English prompt for code generation
        self.code_generation_prompt = """You are a computational biologist and expert Python developer. Your task is to write correct, efficient, and well-documented code based on the provided research plan for single-cell perturbation prediction.

IMPORTANT REQUIREMENTS:
1. Write all code in a single file named 'result.py'
2. Follow the research plan specifications exactly
3. Implement proper data processing, model architecture, and training pipeline
4. Use appropriate libraries (scanpy, torch, numpy, pandas, etc.)
5. Include comprehensive error handling and validation
6. Add detailed comments explaining biological and computational logic

CRITICAL CHECKS TO PERFORM:
- Basic syntax errors and Python best practices
- Variable and class naming consistency (no mismatched names)
- Parameter passing and function signatures
- Model architecture logic and implementation
- Data processing pipeline correctness
- Training loop and optimization logic
- Biological constraint integration
- Memory efficiency and scalability

CODE STRUCTURE REQUIREMENTS:
1. Data loading and preprocessing functions
2. Model architecture definition
3. Training functions with proper validation
4. Evaluation and prediction functions
5. Main execution block with example usage
6. Comprehensive error handling

BIOLOGICAL CONSIDERATIONS:
- Ensure proper handling of single-cell data formats
- Implement appropriate normalization and quality control
- Consider batch effects and technical noise
- Validate biological assumptions in the model
- Include pathway and network analysis capabilities

WAIT FOR COMPLETE INPUT:
- If the research plan JSON is being streamed or is very long, wait for the complete input before starting code generation
- Do not start generating code until you have received the full research plan
- Ensure all JSON content is properly parsed and understood before proceeding

Please generate the complete implementation based on the research plan provided. If you identify any issues or inconsistencies, please fix them and provide explanations for your changes."""

    def _load_env_config(self):
        """Load environment configuration from global .env file"""
        try:
            # Load environment variables from .env file
            from dotenv import load_dotenv
            
            # Try to find .env file in project root
            project_root = Path(__file__).parent.parent.parent  # scAgents root
            env_path = project_root / ".env"
            
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment from: {env_path}")
            else:
                logger.warning(f".env file not found at: {env_path}")
            
            # Get API configuration
            self.api_config = self._get_api_config()
            logger.info(f"Using API configuration: {self.api_config['provider']}")
            
        except ImportError:
            logger.warning("python-dotenv not installed, using system environment variables")
            self.api_config = self._get_api_config()
        except Exception as e:
            logger.error(f"Error loading environment config: {e}")
            self.api_config = self._get_api_config()

    def _get_api_config(self) -> Dict[str, Any]:
        """Get API configuration from environment variables"""
        # Check for different API providers in order of preference
        if os.getenv("OPENAI_API_KEY"):
            return {
                "provider": "openai",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "model": os.getenv("MODEL_NAME", "gpt-4o-mini")
            }
        elif os.getenv("ANTHROPIC_API_KEY"):
            return {
                "provider": "anthropic",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                "model": os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
            }
        elif os.getenv("DEEPSEEK_API_KEY"):
            return {
                "provider": "deepseek",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "model": os.getenv("MODEL_NAME", "deepseek-chat")
            }
        elif os.getenv("LLAMA_API_KEY"):
            return {
                "provider": "llama",
                "api_key": os.getenv("LLAMA_API_KEY"),
                "base_url": os.getenv("LLAMA_BASE_URL", "https://api.llama-api.com"),
                "model": os.getenv("MODEL_NAME", "llama-2-70b-chat")
            }
        elif os.getenv("QWEN_API_KEY"):
            return {
                "provider": "qwen",
                "api_key": os.getenv("QWEN_API_KEY"),
                "base_url": os.getenv("QWEN_BASE_URL", "https://api.qwen.ai"),
                "model": os.getenv("MODEL_NAME", "qwen-turbo")
            }
        else:
            # Fallback to default configuration
            logger.warning("No API key found in environment, using default configuration")
            return {
                "provider": "anthropic",
                "api_key": None,
                "base_url": "https://api.anthropic.com",
                "model": "claude-3-5-sonnet-20241022"
            }

    def check_docker_available(self) -> bool:
        """Check if Docker is available and running"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Docker available: {result.stdout.strip()}")
                return True
            else:
                logger.error("Docker not available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("Docker not found or not accessible")
            return False

    def check_docker_compose_available(self) -> bool:
        """Check if docker-compose is available"""
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Docker Compose available: {result.stdout.strip()}")
                return True
            else:
                logger.error("Docker Compose not available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("Docker Compose not found or not accessible")
            return False

    def start_openhands_docker(self) -> bool:
        """Start OpenHands Docker container"""
        try:
            logger.info("Starting OpenHands Docker container...")
            
            # Check prerequisites
            if not self.check_docker_available():
                logger.error("Docker is not available")
                return False
            
            if not self.check_docker_compose_available():
                logger.error("Docker Compose is not available")
                return False
            
            # Change to OpenHands directory
            original_cwd = Path.cwd()
            os.chdir(self.openhands_path)
            
            # Stop any existing containers first
            logger.info("Stopping any existing OpenHands containers...")
            subprocess.run(
                ["docker-compose", "down"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Set environment variables for OpenHands from .env
            env_vars = self._prepare_environment_variables()
            
            # Start Docker container using docker-compose with environment variables
            logger.info("Building and starting OpenHands container...")
            cmd = [
                "docker-compose", "up", "-d", "--build"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout for build
                env=env_vars
            )
            
            if result.returncode == 0:
                logger.info("OpenHands Docker container started successfully")
                logger.info(f"Container output: {result.stdout}")
                # Wait for container to be ready
                time.sleep(15)
                return True
            else:
                logger.error(f"Failed to start OpenHands: {result.stderr}")
                logger.error(f"Command output: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout starting OpenHands Docker container")
            return False
        except Exception as e:
            logger.error(f"Error starting OpenHands: {e}")
            return False
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def stop_openhands_docker(self) -> bool:
        """Stop OpenHands Docker container"""
        try:
            logger.info("Stopping OpenHands Docker container...")
            
            # Change to OpenHands directory
            original_cwd = Path.cwd()
            os.chdir(self.openhands_path)
            
            # Stop Docker container
            cmd = ["docker-compose", "down"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("OpenHands Docker container stopped successfully")
                return True
            else:
                logger.error(f"Failed to stop OpenHands: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout stopping OpenHands Docker container")
            return False
        except Exception as e:
            logger.error(f"Error stopping OpenHands: {e}")
            return False
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def wait_for_openhands_ready(self, timeout: int = 180) -> bool:
        """Wait for OpenHands to be ready"""
        logger.info("Waiting for OpenHands to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if OpenHands is responding
                response = requests.get("http://localhost:3000", timeout=10)
                if response.status_code == 200:
                    logger.info("OpenHands is ready!")
                    return True
                else:
                    logger.info(f"OpenHands responding with status: {response.status_code}")
            except requests.RequestException as e:
                logger.info(f"OpenHands not ready yet: {e}")
            
            time.sleep(5)
        
        logger.error("OpenHands did not become ready within timeout")
        return False

    def generate_code(self, research_plan: Dict[str, Any], output_dir: str = "cellforge/data/results") -> Optional[str]:
        """
        Generate code using OpenHands based on research plan
        
        Args:
            research_plan: Research plan dictionary
            output_dir: Directory to save generated code
            
        Returns:
            Path to generated code file or None if failed
        """
        try:
            logger.info("=== Starting Code Generation with OpenHands ===")
            
            # Start OpenHands Docker
            if not self.start_openhands_docker():
                logger.error("Failed to start OpenHands Docker")
                return None
            
            # Wait for OpenHands to be ready
            if not self.wait_for_openhands_ready():
                logger.error("OpenHands did not become ready")
                return None
            
            # Prepare the full prompt with research plan
            research_plan_json = json.dumps(research_plan, indent=2, ensure_ascii=False)
            full_prompt = f"{self.code_generation_prompt}\n\nRESEARCH PLAN:\n{research_plan_json}"
            
            # Check if research plan is large and wait for complete input if needed
            if len(research_plan_json) > 5000:  # More than 5KB
                logger.info(f"Large research plan detected ({len(research_plan_json)} characters), ensuring complete input...")
                self._wait_for_complete_json_input(full_prompt)
            
            # Send request to OpenHands API
            code_file_path = self._send_to_openhands(full_prompt, output_dir)
            
            if code_file_path:
                logger.info(f"Code generated successfully: {code_file_path}")
                return code_file_path
            else:
                logger.error("Failed to generate code")
                return None
                
        except Exception as e:
            logger.error(f"Error in code generation: {e}")
            return None
        finally:
            # Stop OpenHands Docker
            self.stop_openhands_docker()

    def _send_to_openhands(self, prompt: str, output_dir: str) -> Optional[str]:
        """Send prompt to OpenHands API and save generated code"""
        try:
            logger.info("Sending request to OpenHands API...")
            
            # Check if prompt contains JSON and wait for complete input if needed
            if self._contains_large_json(prompt):
                logger.info("Detected large JSON content, ensuring complete input...")
                self._wait_for_complete_json_input(prompt)
            
            # Prepare API request
            api_url = "http://localhost:3000/api/chat"
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": self.api_config["model"],
                "temperature": float(os.getenv("TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("MAX_TOKENS", "8000")),
                "api_key": self.api_config["api_key"]
            }
            
            # Add headers if needed for specific providers
            headers = {}
            if self.api_config["provider"] == "anthropic":
                headers["x-api-key"] = self.api_config["api_key"]
                headers["anthropic-version"] = "2023-06-01"
            elif self.api_config["provider"] == "openai":
                headers["Authorization"] = f"Bearer {self.api_config['api_key']}"
            
            logger.info(f"Sending request to: {api_url}")
            logger.info(f"Using model: {self.api_config['model']}")
            
            # Send request
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=600  # 10 minutes timeout
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_code = result.get("content", "")
                
                logger.info(f"Generated content length: {len(generated_code)} characters")
                
                # Extract code from response (look for code blocks)
                if "```python" in generated_code:
                    # Extract code between ```python and ```
                    start_idx = generated_code.find("```python") + 9
                    end_idx = generated_code.find("```", start_idx)
                    if end_idx != -1:
                        code_content = generated_code[start_idx:end_idx].strip()
                    else:
                        code_content = generated_code[start_idx:].strip()
                else:
                    code_content = generated_code
                
                logger.info(f"Extracted code length: {len(code_content)} characters")
                
                # Save code to file
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                code_file_path = output_path / "result.py"
                with open(code_file_path, 'w', encoding='utf-8') as f:
                    f.write(code_content)
                
                logger.info(f"Code saved to: {code_file_path}")
                return str(code_file_path)
            else:
                logger.error(f"OpenHands API error: {response.status_code} - {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error saving generated code: {e}")
            return None

    def _contains_large_json(self, prompt: str) -> bool:
        """Check if prompt contains large JSON content that might need waiting"""
        try:
            # Look for JSON content in the prompt
            if "RESEARCH PLAN:" in prompt:
                json_start = prompt.find("RESEARCH PLAN:") + len("RESEARCH PLAN:")
                json_content = prompt[json_start:].strip()
                
                # Check if it's valid JSON and large enough to be concerning
                if len(json_content) > 1000:  # More than 1KB
                    # Try to parse as JSON to ensure it's complete
                    json.loads(json_content)
                    return True
            return False
        except json.JSONDecodeError:
            # If JSON is incomplete, we need to wait
            return True
        except Exception:
            return False

    def _wait_for_complete_json_input(self, prompt: str, max_wait_time: int = 30) -> None:
        """Wait for complete JSON input before proceeding with code generation"""
        try:
            logger.info("Waiting for complete JSON input...")
            
            start_time = time.time()
            while time.time() - start_time < max_wait_time:
                # Check if JSON is complete by trying to parse it
                if "RESEARCH PLAN:" in prompt:
                    json_start = prompt.find("RESEARCH PLAN:") + len("RESEARCH PLAN:")
                    json_content = prompt[json_start:].strip()
                    
                    try:
                        # Try to parse the JSON
                        json.loads(json_content)
                        logger.info("JSON content is complete and valid")
                        return
                    except json.JSONDecodeError:
                        # JSON is still incomplete, wait a bit more
                        logger.info("JSON content incomplete, waiting...")
                        time.sleep(2)
                        continue
                
                # If no JSON found, proceed
                return
                
            logger.warning(f"Timeout waiting for complete JSON input after {max_wait_time} seconds")
            
        except Exception as e:
            logger.error(f"Error waiting for complete JSON input: {e}")
            # Continue anyway to avoid blocking the process

    def _prepare_environment_variables(self) -> Dict[str, str]:
        """Prepare environment variables for OpenHands from .env file"""
        try:
            # Get current environment
            env_vars = os.environ.copy()
            
            # Add OpenHands specific environment variables from .env
            if self.api_config["provider"] == "openai":
                env_vars["OPENAI_API_KEY"] = self.api_config["api_key"]
                env_vars["OPENAI_BASE_URL"] = self.api_config["base_url"]
                env_vars["MODEL_NAME"] = self.api_config["model"]
            elif self.api_config["provider"] == "anthropic":
                env_vars["ANTHROPIC_API_KEY"] = self.api_config["api_key"]
                env_vars["ANTHROPIC_BASE_URL"] = self.api_config["base_url"]
                env_vars["MODEL_NAME"] = self.api_config["model"]
            elif self.api_config["provider"] == "deepseek":
                env_vars["DEEPSEEK_API_KEY"] = self.api_config["api_key"]
                env_vars["DEEPSEEK_BASE_URL"] = self.api_config["base_url"]
                env_vars["MODEL_NAME"] = self.api_config["model"]
            elif self.api_config["provider"] == "llama":
                env_vars["LLAMA_API_KEY"] = self.api_config["api_key"]
                env_vars["LLAMA_BASE_URL"] = self.api_config["base_url"]
                env_vars["MODEL_NAME"] = self.api_config["model"]
            elif self.api_config["provider"] == "qwen":
                env_vars["QWEN_API_KEY"] = self.api_config["api_key"]
                env_vars["QWEN_BASE_URL"] = self.api_config["base_url"]
                env_vars["MODEL_NAME"] = self.api_config["model"]
            
            # Add other common environment variables
            env_vars["TEMPERATURE"] = os.getenv("TEMPERATURE", "0.1")
            env_vars["MAX_TOKENS"] = os.getenv("MAX_TOKENS", "8000")
            
            # Add OpenHands configuration path
            config_path = self.openhands_path / "config.toml"
            if config_path.exists():
                env_vars["OPENHANDS_CONFIG_PATH"] = str(config_path)
                logger.info(f"Using OpenHands config: {config_path}")
            
            logger.info(f"Prepared environment variables for {self.api_config['provider']}")
            return env_vars
            
        except Exception as e:
            logger.error(f"Error preparing environment variables: {e}")
            return os.environ.copy()

def generate_code_from_plan(research_plan: Dict[str, Any], output_dir: str = "cellforge/data/results") -> Optional[str]:
    """
    Generate code from research plan using OpenHands
    
    Args:
        research_plan: Research plan dictionary
        output_dir: Directory to save generated code
        
    Returns:
        Path to generated code file or None if failed
    """
    try:
        generator = OpenHandsCodeGenerator()
        return generator.generate_code(research_plan, output_dir)
    except Exception as e:
        logger.error(f"Error creating OpenHands generator: {e}")
        return None
