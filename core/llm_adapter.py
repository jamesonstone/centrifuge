"""LLM adapter for canonical value mapping using LiteLLM."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from core.models import PatchBatch, ErrorCategory
from core.planner import LLMBatch, ResidualItem
from core.config import settings

logger = structlog.get_logger()

# Try to import litellm, handle gracefully if not available
try:
    import litellm
    litellm.set_verbose = False  # Reduce logging noise
    LITELLM_AVAILABLE = True
except ImportError:
    logger.warning("LiteLLM not available, using mock adapter")
    LITELLM_AVAILABLE = False


class LLMResponse:
    """Structured response from LLM."""
    
    def __init__(self, raw_response: str, model: str, tokens_used: int = 0):
        self.raw_response = raw_response
        self.model = model
        self.tokens_used = tokens_used
        self.parsed_data: List[Dict[str, Any]] = []
        self.is_valid = False
        self.error_message: Optional[str] = None
        
    def parse_json(self) -> bool:
        """Parse JSON response from LLM."""
        try:
            # Clean up response if needed
            cleaned = self.raw_response.strip()
            
            # Remove markdown code blocks if present
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            # Parse JSON
            self.parsed_data = json.loads(cleaned)
            
            # Validate it's a list
            if not isinstance(self.parsed_data, list):
                self.error_message = "Response is not a JSON array"
                return False
            
            self.is_valid = True
            return True
            
        except json.JSONDecodeError as e:
            self.error_message = f"JSON parse error: {str(e)}"
            logger.error("Failed to parse LLM response", 
                        error=self.error_message,
                        response=self.raw_response[:200])
            return False
        except Exception as e:
            self.error_message = f"Unexpected error: {str(e)}"
            logger.error("Unexpected error parsing response", error=str(e))
            return False


class PromptTemplate:
    """Manages prompt templates."""
    
    def __init__(self, template_path: Path):
        """Load prompt template from YAML file."""
        with open(template_path, 'r') as f:
            self.template = yaml.safe_load(f)
        
        self.version = self.template['version']
        self.name = self.template['name']
        self.system_prompt = self.template['system_prompt']
        self.user_prompt_template = self.template['user_prompt_template']
        self.examples = self.template.get('examples', [])
        self.validation = self.template.get('validation', {})
        self.model_requirements = self.template.get('model_requirements', {})
    
    def format_user_prompt(self, values: List[str]) -> str:
        """Format user prompt with values."""
        values_json = json.dumps(values, indent=2)
        return self.user_prompt_template.replace('{values_json}', values_json)
    
    def validate_response(self, response_data: List[Dict[str, Any]], 
                         canonical_values: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate response against template requirements.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        required_fields = self.validation.get('required_fields', [])
        for item in response_data:
            for field in required_fields:
                if field not in item:
                    errors.append(f"Missing required field '{field}'")
            
            # Check confidence range
            if 'confidence' in item:
                conf = item['confidence']
                min_conf, max_conf = self.validation.get('confidence_range', [0.0, 1.0])
                if not (min_conf <= conf <= max_conf):
                    errors.append(f"Confidence {conf} outside range [{min_conf}, {max_conf}]")
            
            # Check canonical values
            if self.validation.get('mapped_values_must_be_canonical', False):
                mapped = item.get('mapped')
                if mapped is not None and mapped not in canonical_values:
                    errors.append(f"Mapped value '{mapped}' not in canonical list")
        
        return len(errors) == 0, errors


class LLMAdapter:
    """Adapter for LLM interactions using LiteLLM."""
    
    def __init__(self):
        """Initialize LLM adapter."""
        self.model = settings.llm_model_id
        self.temperature = settings.llm_temperature
        self.seed = settings.llm_seed
        self.base_url = settings.llm_base_url
        
        # Load prompt templates
        prompts_dir = Path(__file__).parent.parent / "prompts"
        self.prompts = {
            'Department': PromptTemplate(prompts_dir / "department.yaml"),
            'Account Name': PromptTemplate(prompts_dir / "account_name.yaml")
        }
        
        # Configure LiteLLM if available
        if LITELLM_AVAILABLE and not settings.mock_llm:
            litellm.api_base = self.base_url
            if settings.openai_api_key:
                import os
                os.environ['OPENAI_API_KEY'] = settings.openai_api_key
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def process_batch(self, batch: LLMBatch) -> List[ResidualItem]:
        """
        Process a batch of residual items through LLM.
        
        Args:
            batch: Batch of items to process
            
        Returns:
            List of ResidualItems with mapped values
        """
        start_time = time.time()
        
        # Get prompt template
        prompt_template = self.prompts.get(batch.column_name)
        if not prompt_template:
            logger.error(f"No prompt template for column {batch.column_name}")
            return batch.items
        
        # Get unique values
        unique_values = batch.get_unique_values()
        
        logger.info(f"Processing LLM batch",
                   column=batch.column_name,
                   values_count=len(unique_values),
                   model=self.model)
        
        # Call LLM
        if settings.mock_llm or not LITELLM_AVAILABLE:
            response = self._mock_llm_response(unique_values, batch.column_name)
        else:
            response = await self._call_llm(prompt_template, unique_values)
        
        # Parse response
        if not response.parse_json():
            logger.error("Failed to parse LLM response",
                        column=batch.column_name,
                        error=response.error_message)
            # Mark all items as failed
            for item in batch.items:
                item.confidence = 0.0
                item.mapped_value = None
            return batch.items
        
        # Validate response
        is_valid, errors = prompt_template.validate_response(
            response.parsed_data,
            batch.canonical_options
        )
        
        if not is_valid:
            logger.warning("LLM response validation failed",
                          column=batch.column_name,
                          errors=errors)
        
        # Apply mappings to items
        response_map = {
            item['value']: item 
            for item in response.parsed_data
        }
        
        for item in batch.items:
            if item.value in response_map:
                mapping = response_map[item.value]
                item.mapped_value = mapping.get('mapped')
                item.confidence = mapping.get('confidence', 0.0)
                
                # Log the mapping
                logger.debug("LLM mapping",
                           column=batch.column_name,
                           value=item.value,
                           mapped=item.mapped_value,
                           confidence=item.confidence,
                           reason=mapping.get('reason'))
            else:
                # No mapping found in response
                item.confidence = 0.0
                item.mapped_value = None
        
        # Calculate metrics
        duration_ms = int((time.time() - start_time) * 1000)
        successful_mappings = sum(
            1 for item in batch.items 
            if item.mapped_value and item.confidence >= settings.confidence_floor
        )
        
        logger.info("LLM batch processed",
                   column=batch.column_name,
                   duration_ms=duration_ms,
                   tokens_used=response.tokens_used,
                   successful_mappings=successful_mappings,
                   total_items=len(batch.items))
        
        return batch.items
    
    async def _call_llm(self, prompt_template: PromptTemplate, 
                       values: List[str]) -> LLMResponse:
        """
        Make actual LLM API call via LiteLLM.
        
        Args:
            prompt_template: Template for prompts
            values: Values to process
            
        Returns:
            LLMResponse object
        """
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": prompt_template.system_prompt},
                {"role": "user", "content": prompt_template.format_user_prompt(values)}
            ]
            
            # Add examples if available
            for example in prompt_template.examples[:2]:  # Use first 2 examples
                if 'input' in example and 'output' in example:
                    messages.append({
                        "role": "user",
                        "content": prompt_template.format_user_prompt(example['input']['values'])
                    })
                    messages.append({
                        "role": "assistant",
                        "content": example['output']
                    })
            
            # Call LLM via LiteLLM
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=prompt_template.model_requirements.get('max_tokens', 1000),
                response_format={"type": "json_object"} if self.model.startswith("gpt") else None
            )
            
            # Extract response
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            return LLMResponse(content, self.model, tokens)
            
        except Exception as e:
            logger.error("LLM API call failed", error=str(e))
            # Return empty response on error
            return LLMResponse("[]", self.model, 0)
    
    def _mock_llm_response(self, values: List[str], column: str) -> LLMResponse:
        """
        Generate mock LLM response for testing.
        
        Args:
            values: Values to process
            column: Column name
            
        Returns:
            Mock LLMResponse
        """
        logger.info("Using mock LLM response", column=column)
        
        # Generate deterministic mock mappings
        mock_mappings = []
        
        for value in values:
            if column == 'Department':
                # Mock department mappings
                mapping = self._mock_department_mapping(value)
            elif column == 'Account Name':
                # Mock account mappings
                mapping = self._mock_account_mapping(value)
            else:
                mapping = {
                    "value": value,
                    "mapped": None,
                    "confidence": 0.0,
                    "reason": "Unknown column type"
                }
            
            mock_mappings.append(mapping)
        
        response_json = json.dumps(mock_mappings)
        return LLMResponse(response_json, "mock-model", 0)
    
    def _mock_department_mapping(self, value: str) -> Dict[str, Any]:
        """Generate mock department mapping."""
        value_lower = value.lower()
        
        mappings = {
            'unknown': (None, 0.0, "Cannot determine department"),
            'tech': ('IT', 0.9, "Technology maps to IT"),
            'technology': ('IT', 0.95, "Technology department"),
            'treasury': ('Finance', 0.85, "Treasury is part of Finance"),
            'facilities': ('Operations', 0.8, "Facilities under Operations"),
            'maintenance': ('Operations', 0.85, "Maintenance under Operations"),
            'general': ('Admin', 0.7, "General maps to Admin"),
        }
        
        if value_lower in mappings:
            mapped, confidence, reason = mappings[value_lower]
            return {
                "value": value,
                "mapped": mapped,
                "confidence": confidence,
                "reason": reason
            }
        
        # Default mapping
        return {
            "value": value,
            "mapped": None,
            "confidence": 0.0,
            "reason": "No mock mapping available"
        }
    
    def _mock_account_mapping(self, value: str) -> Dict[str, Any]:
        """Generate mock account mapping."""
        value_lower = value.lower()
        
        mappings = {
            'bank fees': ('Operating Expenses', 0.9, "Bank fees are operating expenses"),
            'office supplies': ('Operating Expenses', 0.95, "Office supplies are operating expenses"),
            'petty cash': ('Cash', 0.85, "Petty cash is a cash account"),
            'interest income': ('Sales Revenue', 0.7, "Interest as revenue"),
            'suspense account': (None, 0.0, "Suspense needs investigation"),
            'travel & entertainment': ('Operating Expenses', 0.95, "T&E is operating expense"),
            'professional fees': ('Operating Expenses', 0.95, "Professional fees are operating"),
            'rent expense': ('Operating Expenses', 1.0, "Rent is operating expense"),
            'utilities expense': ('Operating Expenses', 1.0, "Utilities are operating"),
            'insurance expense': ('Operating Expenses', 0.95, "Insurance is operating"),
            'training expense': ('Operating Expenses', 0.9, "Training is operating"),
            'consulting fees': ('Operating Expenses', 0.9, "Consulting is operating"),
            'maintenance expense': ('Operating Expenses', 0.95, "Maintenance is operating"),
            'communications': ('Operating Expenses', 0.9, "Communications are operating"),
            'freight expense': ('Operating Expenses', 0.85, "Freight is operating"),
            'marketing expense': ('Operating Expenses', 0.95, "Marketing is operating"),
        }
        
        if value_lower in mappings:
            mapped, confidence, reason = mappings[value_lower]
            return {
                "value": value,
                "mapped": mapped,
                "confidence": confidence,
                "reason": reason
            }
        
        # Default mapping
        return {
            "value": value,
            "mapped": None,
            "confidence": 0.0,
            "reason": "No mock mapping available"
        }
    
    def get_prompt_version(self, column: str) -> str:
        """Get prompt version for a column."""
        if column in self.prompts:
            return self.prompts[column].version
        return "unknown"