"""Residual planner for identifying issues requiring LLM assistance."""

import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import pandas as pd
import structlog
from core.models import (
    Schema, SourceType, Patch, AuditEvent, DiffEntry
)
from core.db_models import DatabaseManager
from core.config import settings

logger = structlog.get_logger()


class ResidualItem:
    """Represents a value that needs LLM processing."""
    
    def __init__(
        self,
        column_name: str,
        value: str,
        canonical_options: List[str],
        row_indices: List[int],
        row_uuids: List[UUID],
        row_numbers: List[int]
    ):
        self.column_name = column_name
        self.value = value
        self.canonical_options = canonical_options
        self.row_indices = row_indices  # DataFrame indices
        self.row_uuids = row_uuids
        self.row_numbers = row_numbers  # Original row numbers
        self.mapped_value: Optional[str] = None
        self.confidence: float = 0.0
        self.source: SourceType = SourceType.LLM
        
    def __repr__(self):
        return f"ResidualItem(column={self.column_name}, value='{self.value}', occurrences={len(self.row_indices)})"


class LLMBatch:
    """Batch of residual items for LLM processing."""
    
    def __init__(self, column_name: str, items: List[ResidualItem]):
        self.column_name = column_name
        self.items = items
        self.canonical_options: List[str] = []
        if items:
            self.canonical_options = items[0].canonical_options
    
    def get_unique_values(self) -> List[str]:
        """Get unique values in this batch."""
        return [item.value for item in self.items]
    
    def __len__(self):
        return len(self.items)
    
    def __repr__(self):
        return f"LLMBatch(column={self.column_name}, items={len(self.items)})"


class ResidualPlanner:
    """Plans and manages residual items requiring LLM assistance."""
    
    def __init__(self, schema: Schema, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize residual planner.
        
        Args:
            schema: Schema with canonical values
            db_manager: Database manager for cache lookups
        """
        self.schema = schema
        self.db = db_manager
        self.residual_items: Dict[str, List[ResidualItem]] = defaultdict(list)
        self.cache_hits: List[ResidualItem] = []
        self.cache_misses: List[ResidualItem] = []
        self.edit_caps: Dict[str, int] = {}
        
    async def identify_residuals(
        self,
        df: pd.DataFrame,
        llm_columns: List[str] = None
    ) -> Tuple[Dict[str, List[ResidualItem]], Dict[str, int]]:
        """
        Identify values requiring LLM processing.
        
        Args:
            df: DataFrame after rules processing
            llm_columns: Columns enabled for LLM processing
            
        Returns:
            Tuple of (residual_items_by_column, edit_counts)
        """
        if llm_columns is None:
            llm_columns = ['Department', 'Account Name']
        
        logger.info("Identifying residual items", columns=llm_columns)
        
        self.residual_items.clear()
        self.cache_hits.clear()
        self.cache_misses.clear()
        
        # Process each LLM-enabled column
        for column in llm_columns:
            if column not in df.columns:
                continue
                
            # Get canonical values for this column
            canonical_values = self._get_canonical_values(column)
            if not canonical_values:
                logger.warning(f"No canonical values for column {column}")
                continue
            
            # Group by unique values
            value_groups = defaultdict(list)
            
            for idx, row in df.iterrows():
                value = row[column]
                if pd.notna(value) and value not in canonical_values:
                    # This value needs processing
                    value_groups[value].append({
                        'index': idx,
                        'row_uuid': UUID(row['row_uuid']),
                        'row_number': row['row_number']
                    })
            
            # Create ResidualItem for each unique value
            for value, occurrences in value_groups.items():
                residual = ResidualItem(
                    column_name=column,
                    value=str(value),
                    canonical_options=canonical_values,
                    row_indices=[occ['index'] for occ in occurrences],
                    row_uuids=[occ['row_uuid'] for occ in occurrences],
                    row_numbers=[occ['row_number'] for occ in occurrences]
                )
                self.residual_items[column].append(residual)
        
        # Check cache for existing mappings
        if self.db:
            await self._check_cache()
        
        # Calculate edit counts
        edit_counts = self._calculate_edit_counts(df)
        
        # Log summary
        total_residuals = sum(len(items) for items in self.residual_items.values())
        total_rows_affected = sum(
            sum(len(item.row_indices) for item in items)
            for items in self.residual_items.values()
        )
        
        logger.info("Residual identification complete",
                   unique_values=total_residuals,
                   rows_affected=total_rows_affected,
                   cache_hits=len(self.cache_hits),
                   cache_misses=len(self.cache_misses))
        
        return self.residual_items, edit_counts
    
    def _get_canonical_values(self, column: str) -> List[str]:
        """Get canonical values for a column."""
        if column == 'Department':
            return self.schema.canonical_departments
        elif column == 'Account Name':
            return self.schema.canonical_accounts
        else:
            # Check if column definition has enum values
            col_def = self.schema.columns.get(column)
            if col_def and col_def.enum_values:
                return col_def.enum_values
        return []
    
    async def _check_cache(self):
        """Check cache for existing mappings."""
        for column, items in self.residual_items.items():
            # Get all unique values for this column
            unique_values = [item.value for item in items]
            
            # Batch lookup in cache
            cached_mappings = await self.db.get_mappings_batch(column, unique_values)
            
            # Apply cached mappings
            remaining_items = []
            for item in items:
                if item.value in cached_mappings:
                    # Cache hit!
                    item.mapped_value = cached_mappings[item.value]
                    item.confidence = 1.0  # Cached values are approved
                    item.source = SourceType.CACHE
                    self.cache_hits.append(item)
                    logger.debug("Cache hit",
                               column=column,
                               value=item.value,
                               mapped=item.mapped_value)
                else:
                    # Cache miss - needs LLM
                    self.cache_misses.append(item)
                    remaining_items.append(item)
            
            # Update residual items to only include cache misses
            self.residual_items[column] = remaining_items
    
    def _calculate_edit_counts(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate how many edits would be made per column."""
        edit_counts = {}
        
        for column, items in self.residual_items.items():
            # Count unique rows that would be edited
            unique_rows = set()
            for item in items:
                unique_rows.update(item.row_indices)
            
            edit_counts[column] = len(unique_rows)
            
            # Calculate percentage
            edit_percentage = (len(unique_rows) / len(df)) * 100 if len(df) > 0 else 0
            
            # Check against edit cap
            if edit_percentage > settings.edit_cap_pct:
                logger.warning(f"Edit cap exceeded for {column}",
                             edits=len(unique_rows),
                             percentage=edit_percentage,
                             cap=settings.edit_cap_pct)
                self.edit_caps[column] = len(unique_rows)
        
        return edit_counts
    
    def create_llm_batches(
        self,
        batch_size: int = None
    ) -> List[LLMBatch]:
        """
        Create batches of residual items for LLM processing.
        
        Args:
            batch_size: Maximum items per batch (default from settings)
            
        Returns:
            List of LLMBatch objects
        """
        if batch_size is None:
            batch_size = settings.batch_size
        
        batches = []
        
        for column, items in self.residual_items.items():
            if not items:
                continue
                
            # Check if column exceeds edit cap
            if column in self.edit_caps:
                logger.warning(f"Skipping {column} due to edit cap")
                continue
            
            # Group items into batches
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                batch = LLMBatch(column, batch_items)
                batches.append(batch)
                
                logger.debug(f"Created batch for {column}",
                           size=len(batch_items),
                           values=batch.get_unique_values()[:5])  # Log first 5
        
        logger.info("Created LLM batches",
                   total_batches=len(batches),
                   total_items=sum(len(b) for b in batches))
        
        return batches
    
    def apply_mappings(
        self,
        df: pd.DataFrame,
        run_id: UUID,
        include_cache_hits: bool = True
    ) -> Tuple[pd.DataFrame, List[Patch], List[AuditEvent], List[DiffEntry]]:
        """
        Apply resolved mappings to DataFrame.
        
        Args:
            df: DataFrame to update
            run_id: Run ID for audit
            include_cache_hits: Whether to apply cache hits
            
        Returns:
            Tuple of (updated_df, patches, audit_events, diff_entries)
        """
        patches = []
        audit_events = []
        diff_entries = []
        
        # Collect all items to apply
        items_to_apply = []
        
        if include_cache_hits:
            items_to_apply.extend(self.cache_hits)
        
        # Add resolved LLM items
        for items in self.residual_items.values():
            for item in items:
                if item.mapped_value and item.confidence >= settings.confidence_floor:
                    items_to_apply.append(item)
        
        # Apply mappings
        for item in items_to_apply:
            for idx, row_uuid, row_number in zip(
                item.row_indices,
                item.row_uuids,
                item.row_numbers
            ):
                # Update DataFrame
                old_value = df.at[idx, item.column_name]
                df.at[idx, item.column_name] = item.mapped_value
                
                # Create patch
                patch = Patch(
                    row_uuid=row_uuid,
                    row_number=row_number,
                    column_name=item.column_name,
                    before_value=str(old_value),
                    after_value=item.mapped_value,
                    confidence=item.confidence,
                    source=item.source,
                    rule_id='canonical_mapping' if item.source == SourceType.CACHE else None,
                    contract_id='llm_mapping' if item.source == SourceType.LLM else None,
                    reason=f'Mapped to canonical value via {item.source.value}'
                )
                patches.append(patch)
                
                # Create audit event
                audit = AuditEvent(
                    run_id=run_id,
                    row_uuid=row_uuid,
                    column_name=item.column_name,
                    before_value=str(old_value),
                    after_value=item.mapped_value,
                    source=item.source,
                    rule_id='canonical_mapping' if item.source == SourceType.CACHE else None,
                    contract_id='llm_mapping' if item.source == SourceType.LLM else None,
                    reason=f'Mapped to canonical value',
                    confidence=item.confidence
                )
                audit_events.append(audit)
                
                # Create diff entry
                diff = DiffEntry(
                    row_number=row_number,
                    row_uuid=row_uuid,
                    column_name=item.column_name,
                    before_value=str(old_value),
                    after_value=item.mapped_value,
                    source=item.source,
                    reason=f'Canonical mapping ({item.source.value})'
                )
                diff_entries.append(diff)
        
        logger.info("Applied mappings",
                   total_patches=len(patches),
                   cache_source=sum(1 for p in patches if p.source == SourceType.CACHE),
                   llm_source=sum(1 for p in patches if p.source == SourceType.LLM))
        
        return df, patches, audit_events, diff_entries
    
    async def save_to_cache(
        self,
        model_id: str,
        prompt_version: str
    ) -> int:
        """
        Save successful LLM mappings to cache.
        
        Args:
            model_id: Model used for mappings
            prompt_version: Prompt version used
            
        Returns:
            Number of mappings saved
        """
        if not self.db:
            return 0
        
        saved_count = 0
        
        for items in self.residual_items.values():
            for item in items:
                if (item.mapped_value and 
                    item.source == SourceType.LLM and
                    item.confidence >= settings.confidence_floor):
                    
                    # Save to cache
                    mapping_id = await self.db.create_canonical_mapping(
                        column_name=item.column_name,
                        variant_value=item.value,
                        canonical_value=item.mapped_value,
                        model_id=model_id,
                        prompt_version=prompt_version,
                        source=SourceType.LLM,
                        confidence=item.confidence
                    )
                    
                    if mapping_id:
                        saved_count += 1
                        logger.debug("Saved to cache",
                                   column=item.column_name,
                                   value=item.value,
                                   mapped=item.mapped_value,
                                   confidence=item.confidence)
        
        logger.info("Saved mappings to cache", count=saved_count)
        return saved_count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of planning results."""
        return {
            'total_unique_values': sum(len(items) for items in self.residual_items.values()),
            'total_rows_affected': sum(
                sum(len(item.row_indices) for item in items)
                for items in self.residual_items.values()
            ),
            'cache_hits': len(self.cache_hits),
            'cache_misses': len(self.cache_misses),
            'cache_hit_ratio': (
                len(self.cache_hits) / (len(self.cache_hits) + len(self.cache_misses))
                if (len(self.cache_hits) + len(self.cache_misses)) > 0
                else 0
            ),
            'columns_with_residuals': list(self.residual_items.keys()),
            'edit_cap_violations': list(self.edit_caps.keys()),
            'residuals_by_column': {
                col: {
                    'unique_values': len(items),
                    'rows_affected': sum(len(item.row_indices) for item in items),
                    'examples': [item.value for item in items[:5]]  # First 5 examples
                }
                for col, items in self.residual_items.items()
            }
        }