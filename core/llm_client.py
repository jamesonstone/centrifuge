"""
LLM client for Centrifuge.
Integrates directly with LiteLLM library for canonical value mapping.
"""

import os
import json
import logging
import asyncio
import litellm
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM interactions using direct LiteLLM library integration."""

    def __init__(self,
                 base_url: str = None,  # Kept for backward compatibility but unused
                 api_key: str = None,
                 model: str = "gpt-4o",  # Updated to gpt-4o for JSON mode support
                 temperature: float = 0.0,  # Set to 0 for deterministic output
                 max_retries: int = 3):
        """
        Initialize LLM client.

        Args:
            base_url: Deprecated - kept for compatibility (direct library doesn't need proxy URL)
            api_key: OpenAI API key (default from env OPENAI_API_KEY)
            model: Model to use
            temperature: Temperature for responses
            max_retries: Maximum retry attempts
        """
        # Note: base_url is ignored in direct library mode
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        # Allow environment variable to override model
        self.model = os.getenv('LLM_MODEL_ID', model)
        # Ensure temperature is 0 for deterministic output
        self.temperature = float(os.getenv('LLM_TEMPERATURE', str(temperature)))
        self.max_retries = max_retries
        
        # Set seed for deterministic output
        self.seed = int(os.getenv('LLM_SEED', '42'))

        # configure litellm
        if self.api_key:
            litellm.api_key = self.api_key
            
        # Set litellm to drop unsupported params gracefully
        litellm.drop_params = True

        # check for api key in production
        if not self.api_key and not os.getenv('USE_MOCK_LLM', 'false').lower() == 'true':
            logger.warning("No OPENAI_API_KEY found, LLM features will be limited")

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a canonicalization request.

        Args:
            request: Request with column, values, and canonical_values

        Returns:
            Response with mappings
        """
        column = request.get('column')
        values = request.get('values', [])
        canonical_values = request.get('canonical_values', [])
        
        logger.debug(f"LLM Request: column='{column}', values_count={len(values)}")

        if not values:
            logger.debug("No values to process, returning empty mappings")
            return {'success': True, 'mappings': {}}

        # check if we should use mock
        use_mock = os.getenv('USE_MOCK_LLM', 'false').lower() == 'true'
        if use_mock or not self.api_key:
            logger.info(f"Using {'MOCK' if use_mock else 'MOCK (no API key)'} LLM for column '{column}'")
            return self._generate_mock_response(column, values, canonical_values)

        # prepare prompt
        system_prompt = self._get_system_prompt(column)
        user_prompt = self._get_user_prompt(column, values, canonical_values)

        # call llm with retry
        logger.info(f"Calling real LLM API for column '{column}' (model: {self.model})")
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"  Attempt {attempt + 1}/{self.max_retries}")
                response = await self._call_llm(system_prompt, user_prompt)

                if response:
                    # parse and validate response
                    mappings = self._parse_llm_response(response, values, canonical_values)
                    
                    logger.info(f"  LLM returned {len(mappings)} mappings for column '{column}'")
                    non_null = {k: v for k, v in mappings.items() if v is not None}
                    logger.debug(f"  Non-null mappings: {len(non_null)}")

                    return {
                        'success': True,
                        'mappings': mappings,
                        'confidence': 0.85,
                        'model': self.model
                    }

            except Exception as e:
                logger.warning(f"  LLM call attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    # exponential backoff
                    await asyncio.sleep(2 ** attempt)

        # all retries failed
        logger.error(f"All LLM attempts failed for column '{column}'")
        return {
            'success': False,
            'error': 'LLM processing failed after retries',
            'mappings': {}
        }

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Make actual LLM API call using direct LiteLLM library.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt

        Returns:
            LLM response text or None
        """
        try:
            # prepare messages
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            
            # Build kwargs based on model capabilities
            kwargs = {
                'model': self.model,
                'messages': messages,
                'temperature': self.temperature,
                'api_key': self.api_key,
                'seed': self.seed  # For deterministic output
            }
            
            # Only add response_format for models that support it
            if 'gpt-4o' in self.model or 'gpt-4-turbo' in self.model or 'gpt-3.5-turbo' in self.model:
                kwargs['response_format'] = {'type': 'json_object'}
            else:
                # For older models, we'll request JSON in the prompt itself
                logger.info(f"Model {self.model} may not support response_format, relying on prompt instructions")

            # make direct library call
            logger.debug(f"    Calling litellm with model={kwargs['model']}, temp={kwargs['temperature']}")
            response = await litellm.acompletion(**kwargs)
            
            content = response.choices[0].message.content
            logger.debug(f"    Response received: {len(content)} chars")
            return content

        except Exception as e:
            logger.debug(f"    LLM API error: {e}")

            # check for specific errors
            error_str = str(e).lower()
            if 'api key' in error_str or 'unauthorized' in error_str:
                raise ValueError("Invalid API key")
            elif 'rate limit' in error_str or 'quota' in error_str:
                raise ValueError("Rate limit exceeded or quota exhausted")
            elif 'response_format' in error_str:
                # Log and retry without response_format
                logger.warning(f"Model {self.model} doesn't support response_format, retrying without it")
                raise ValueError("Model doesn't support response_format")
            else:
                raise ValueError(f"API error: {e}")

    def _get_system_prompt(self, column: str) -> str:
        """Get system prompt for column."""
        if column == 'Department':
            return """You are a data standardization expert specializing in department names.
Your task is to map variant department names to canonical values.

Canonical departments: Sales, Operations, Admin, IT, Finance, Marketing, HR, Legal, Engineering, Support

For each value, determine if it can be mapped to a canonical department.
If unclear or ambiguous, set mapped to null.

Always respond with valid JSON only. Return a JSON object mapping each input value to its canonical value or null.
Do not include any text before or after the JSON object."""

        elif column == 'Account Name':
            return """You are an accounting expert specializing in chart of accounts standardization.
Your task is to map variant account names to canonical values.

Canonical accounts: Cash, Accounts Receivable, Accounts Payable, Sales Revenue,
Cost of Goods Sold, Operating Expenses, Equipment, Inventory, Retained Earnings

For each value, determine if it can be mapped to a canonical account.
If unclear or requires investigation, set mapped to null.

Always respond with valid JSON only. Return a JSON object mapping each input value to its canonical value or null.
Do not include any text before or after the JSON object."""

        else:
            return f"""You are a data standardization expert.
Map the provided values to canonical values for the column '{column}'.
Always respond with valid JSON only. Return a JSON object mapping each input value to its canonical value or null.
Do not include any text before or after the JSON object."""

    def _get_user_prompt(self, column: str, values: List[str], canonical_values: List[str]) -> str:
        """Get user prompt."""
        values_str = json.dumps(values, indent=2)
        canonical_str = json.dumps(canonical_values, indent=2) if canonical_values else "[]"

        return f"""Map these {column} values to canonical values.

Input values:
{values_str}

Canonical values:
{canonical_str}

Return ONLY a valid JSON object where keys are input values and values are canonical mappings or null.
Example: {{"Tech Support": "Support", "Unknown Dept": null}}

Remember: Output only the JSON object, no additional text."""

    def _parse_llm_response(self, response: str, values: List[str], canonical_values: List[str]) -> Dict[str, str]:
        """
        Parse and validate LLM response.

        Args:
            response: Raw LLM response
            values: Original values
            canonical_values: Valid canonical values

        Returns:
            Mapping dictionary
        """
        try:
            # clean response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            # parse json
            mappings = json.loads(cleaned)

            if not isinstance(mappings, dict):
                logger.error("LLM response is not a dictionary")
                return {}

            # validate mappings
            validated = {}
            for value in values:
                if value in mappings:
                    mapped = mappings[value]

                    # validate canonical value if provided
                    if mapped and canonical_values and mapped not in canonical_values:
                        logger.warning(f"LLM mapped '{value}' to non-canonical '{mapped}'")
                        validated[value] = None
                    else:
                        validated[value] = mapped

            return validated

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {}

    def _generate_mock_response(self, column: str, values: List[str], canonical_values: List[str]) -> Dict[str, Any]:
        """Generate mock response for testing."""
        logger.info(f"Generating mock LLM response for {column}")

        mappings = {}

        for value in values:
            if column == 'Department':
                mappings[value] = self._mock_department(value)
            elif column == 'Account Name':
                mappings[value] = self._mock_account(value)
            else:
                mappings[value] = None

        return {
            'success': True,
            'mappings': mappings,
            'confidence': 0.9,
            'model': 'mock'
        }

    def _mock_department(self, value: str) -> Optional[str]:
        """Mock department mapping."""
        value_lower = value.lower()

        if 'tech' in value_lower or 'it' in value_lower:
            return 'IT'
        elif 'sale' in value_lower:
            return 'Sales'
        elif 'financ' in value_lower or 'account' in value_lower:
            return 'Finance'
        elif 'market' in value_lower:
            return 'Marketing'
        elif 'hr' in value_lower or 'human' in value_lower:
            return 'HR'
        elif 'legal' in value_lower or 'law' in value_lower:
            return 'Legal'
        elif 'engineer' in value_lower or 'dev' in value_lower:
            return 'Engineering'
        elif 'support' in value_lower or 'help' in value_lower:
            return 'Support'
        elif 'ops' in value_lower or 'operation' in value_lower:
            return 'Operations'
        elif 'admin' in value_lower or 'general' in value_lower:
            return 'Admin'
        else:
            return None

    def _mock_account(self, value: str) -> Optional[str]:
        """Mock account mapping."""
        value_lower = value.lower()

        if 'cash' in value_lower and 'petty' not in value_lower:
            return 'Cash'
        elif 'receivable' in value_lower or 'a/r' in value_lower:
            return 'Accounts Receivable'
        elif 'payable' in value_lower or 'a/p' in value_lower:
            return 'Accounts Payable'
        elif 'revenue' in value_lower or 'sales' in value_lower or 'income' in value_lower:
            return 'Sales Revenue'
        elif 'cogs' in value_lower or 'cost of' in value_lower:
            return 'Cost of Goods Sold'
        elif 'expense' in value_lower or 'cost' in value_lower:
            return 'Operating Expenses'
        elif 'equipment' in value_lower or 'asset' in value_lower:
            return 'Equipment'
        elif 'inventory' in value_lower or 'stock' in value_lower:
            return 'Inventory'
        elif 'retained' in value_lower or 'earnings' in value_lower:
            return 'Retained Earnings'
        else:
            return None


# global llm adapter instance
_llm_adapter: Optional[LLMClient] = None


async def get_llm_adapter(use_mock: bool = False) -> LLMClient:
    """
    Get LLM adapter instance.

    Args:
        use_mock: Force mock mode

    Returns:
        LLM adapter
    """
    global _llm_adapter

    if not _llm_adapter:
        if use_mock:
            os.environ['USE_MOCK_LLM'] = 'true'

        _llm_adapter = LLMClient()

    return _llm_adapter
