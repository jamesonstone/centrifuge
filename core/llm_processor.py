"""
Enhanced LLM processor with detailed logging for pipeline visibility.
"""

import logging
from typing import Dict, List, Any, Optional
from core.llm_client import get_llm_adapter

logger = logging.getLogger(__name__)


class LLMProcessor:
    """Enhanced LLM processor with comprehensive logging."""
    
    def __init__(self, cache=None):
        """Initialize LLM processor."""
        self.cache = cache
        self.llm_adapter = None
        
    def _get_canonical_values(self, column: str) -> List[str]:
        """Get canonical values for a column."""
        if column == 'Department':
            return ['Sales', 'Operations', 'Admin', 'IT', 'Finance', 'Marketing', 'HR', 'Legal', 'Engineering', 'Support']
        elif column == 'Account Name':
            return ['Cash', 'Accounts Receivable', 'Accounts Payable', 'Sales Revenue', 
                   'Cost of Goods Sold', 'Operating Expenses', 'Equipment', 'Inventory', 'Retained Earnings']
        else:
            return []
    
    async def process_with_llm(self, 
                               residual_items: Dict[str, set],
                               data,
                               audit_trail: list,
                               dry_run: bool = False) -> Dict[str, Any]:
        """
        Process residual items with LLM.
        
        Args:
            residual_items: Dictionary of column -> set of unique values
            data: DataFrame with data to update
            audit_trail: List to append audit entries
            dry_run: If True, use mock LLM
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info("=" * 60)
        logger.info("LLM PROCESSING PHASE STARTING")
        logger.info("=" * 60)
        
        # Initialize LLM adapter if needed
        if not self.llm_adapter:
            self.llm_adapter = await get_llm_adapter(use_mock=dry_run)
            logger.info(f"LLM Adapter Configuration:")
            logger.info(f"  Model: {getattr(self.llm_adapter, 'model', 'unknown')}")
            logger.info(f"  Temperature: {getattr(self.llm_adapter, 'temperature', 'unknown')}")
            logger.info(f"  Seed: {getattr(self.llm_adapter, 'seed', 'unknown')}")
            logger.info(f"  Mock Mode: {dry_run}")
        
        # Create patches from residual items
        patches = []
        total_values = 0
        for column, values in residual_items.items():
            if values:
                patches.append({
                    'column': column,
                    'values': list(values),
                    'canonical_values': self._get_canonical_values(column)
                })
                total_values += len(values)
        
        logger.info(f"LLM Processing Overview:")
        logger.info(f"  Total patches to process: {len(patches)}")
        logger.info(f"  Total unique values: {total_values}")
        
        # Show breakdown by column
        for patch in patches:
            logger.info(f"  Column '{patch['column']}':")
            logger.info(f"    Unique values to canonicalize: {len(patch['values'])}")
            if len(patch['values']) <= 5:
                logger.info(f"    Values: {patch['values']}")
            else:
                logger.info(f"    Sample values: {patch['values'][:5]}...")
        
        logger.info("-" * 60)
        
        # Process each patch
        llm_fixed_count = 0
        llm_attempted = 0
        llm_errors = 0
        
        for i, patch in enumerate(patches, 1):
            logger.info(f"Processing Patch {i}/{len(patches)}")
            logger.info(f"  Column: '{patch['column']}'")
            logger.info(f"  Values to process: {len(patch['values'])}")
            
            try:
                # Prepare request
                request = {
                    'column': patch['column'],
                    'values': patch['values'],
                    'canonical_values': patch.get('canonical_values', [])
                }
                
                llm_attempted += len(patch['values'])
                
                # Call LLM
                logger.info(f"  Calling LLM...")
                response = await self.llm_adapter.process(request)
                
                # Log response status
                success = response.get('success', False)
                model_used = response.get('model', 'unknown')
                error_msg = response.get('error', '')
                
                logger.info(f"  LLM Response:")
                logger.info(f"    Success: {success}")
                logger.info(f"    Model: {model_used}")
                
                if success:
                    mappings = response.get('mappings', {})
                    confidence = response.get('confidence', 0)
                    
                    logger.info(f"    Mappings received: {len(mappings)}")
                    logger.info(f"    Confidence: {confidence}")
                    
                    # Show sample mappings
                    if mappings:
                        non_null_mappings = {k: v for k, v in mappings.items() if v is not None}
                        logger.info(f"    Non-null mappings: {len(non_null_mappings)}")
                        
                        if non_null_mappings:
                            sample = dict(list(non_null_mappings.items())[:3])
                            logger.info(f"    Sample mappings: {sample}")
                    
                    # Apply mappings
                    local_fixed = 0
                    for value, canonical in mappings.items():
                        if canonical:  # Only process non-null mappings
                            # Update DataFrame
                            mask = data[patch['column']] == value
                            rows_affected = mask.sum()
                            
                            if rows_affected > 0:
                                data.loc[mask, patch['column']] = canonical
                                local_fixed += rows_affected
                                llm_fixed_count += rows_affected
                                
                                # Add to audit trail
                                for idx in data[mask].index:
                                    audit_trail.append({
                                        'row_id': idx,
                                        'type': 'llm',
                                        'column': patch['column'],
                                        'before': value,
                                        'after': canonical,
                                        'confidence': confidence
                                    })
                                
                                # Cache the mapping
                                if self.cache and not dry_run:
                                    await self.cache.store_mapping(
                                        patch['column'],
                                        value,
                                        canonical,
                                        confidence=confidence
                                    )
                    
                    logger.info(f"  Applied {local_fixed} fixes to data")
                    
                else:
                    logger.warning(f"  LLM request failed: {error_msg}")
                    llm_errors += len(patch['values'])
                    
            except Exception as e:
                logger.error(f"  Exception processing patch: {e}")
                llm_errors += len(patch['values'])
        
        logger.info("-" * 60)
        logger.info("LLM PROCESSING PHASE COMPLETE")
        logger.info(f"  Total values attempted: {llm_attempted}")
        logger.info(f"  Total values fixed: {llm_fixed_count}")
        logger.info(f"  Total values errored: {llm_errors}")
        logger.info(f"  Success rate: {llm_fixed_count/llm_attempted*100:.1f}%" if llm_attempted > 0 else "N/A")
        logger.info("=" * 60)
        
        return {
            'llm_processed': len(patches),
            'llm_attempted': llm_attempted,
            'llm_fixed': llm_fixed_count,
            'llm_errors': llm_errors
        }