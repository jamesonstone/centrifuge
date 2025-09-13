"""
LLM client for Centrifuge.
Integrates with LiteLLM proxy for canonical value mapping.
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM interactions via LiteLLM proxy."""
    
    def __init__(self,
                 base_url: str = None,
                 api_key: str = None,
                 model: str = "gpt-4",
                 temperature: float = 0.1,
                 max_retries: int = 3):
        """
        Initialize LLM client.
        
        Args:
            base_url: LiteLLM proxy URL (default from env LITELLM_URL)
            api_key: OpenAI API key (default from env OPENAI_API_KEY)
            model: Model to use
            temperature: Temperature for responses
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url or os.getenv('LITELLM_URL', 'http://litellm:4000')
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        
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
        
        if not values:
            return {'success': True, 'mappings': {}}
        
        # check if we should use mock
        if os.getenv('USE_MOCK_LLM', 'false').lower() == 'true' or not self.api_key:
            return self._generate_mock_response(column, values, canonical_values)
        
        # prepare prompt
        system_prompt = self._get_system_prompt(column)
        user_prompt = self._get_user_prompt(column, values, canonical_values)
        
        # call llm with retry
        for attempt in range(self.max_retries):
            try:
                response = await self._call_llm(system_prompt, user_prompt)
                
                if response:
                    # parse and validate response
                    mappings = self._parse_llm_response(response, values, canonical_values)
                    
                    return {
                        'success': True,
                        'mappings': mappings,
                        'confidence': 0.85,
                        'model': self.model
                    }
                    
            except Exception as e:
                logger.error(f"LLM call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # exponential backoff
                    await asyncio.sleep(2 ** attempt)
        
        # all retries failed
        return {
            'success': False,
            'error': 'LLM processing failed after retries',
            'mappings': {}
        }
    
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Make actual LLM API call.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            LLM response text or None
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'temperature': self.temperature,
            'response_format': {'type': 'json_object'}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    logger.error(f"LLM API error: {response.status} - {error_text}")
                    
                    # check for specific errors
                    if response.status == 401:
                        raise ValueError("Invalid API key")
                    elif response.status == 429:
                        raise ValueError("Rate limit exceeded")
                    else:
                        raise ValueError(f"API error: {response.status}")
        
        return None
    
    def _get_system_prompt(self, column: str) -> str:
        """Get system prompt for column."""
        if column == 'Department':
            return """You are a data standardization expert specializing in department names.
Your task is to map variant department names to canonical values.

Canonical departments: Sales, Operations, Admin, IT, Finance, Marketing, HR, Legal, Engineering, Support

For each value, determine if it can be mapped to a canonical department.
If unclear or ambiguous, set mapped to null.

Respond with a JSON object mapping each input value to its canonical value or null."""
        
        elif column == 'Account Name':
            return """You are an accounting expert specializing in chart of accounts standardization.
Your task is to map variant account names to canonical values.

Canonical accounts: Cash, Accounts Receivable, Accounts Payable, Sales Revenue, 
Cost of Goods Sold, Operating Expenses, Equipment, Inventory, Retained Earnings

For each value, determine if it can be mapped to a canonical account.
If unclear or requires investigation, set mapped to null.

Respond with a JSON object mapping each input value to its canonical value or null."""
        
        else:
            return f"""You are a data standardization expert.
Map the provided values to canonical values for the column '{column}'.
Respond with a JSON object mapping each input value to its canonical value or null."""
    
    def _get_user_prompt(self, column: str, values: List[str], canonical_values: List[str]) -> str:
        """Get user prompt."""
        values_str = json.dumps(values, indent=2)
        canonical_str = json.dumps(canonical_values, indent=2) if canonical_values else "[]"
        
        return f"""Map these {column} values to canonical values.

Input values:
{values_str}

Canonical values:
{canonical_str}

Return a JSON object where keys are input values and values are canonical mappings or null.
Example: {{"Tech Support": "Support", "Unknown Dept": null}}"""
    
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