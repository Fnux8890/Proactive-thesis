"""Era-aware context provider for MOEA optimization.

This module provides temporal context based on multi-level era detection
to enable era-specific optimization strategies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class EraContextProvider:
    """Provides era-based context for optimization."""
    
    def __init__(self, db_url: str):
        """Initialize with database connection.
        
        Args:
            db_url: Database connection string
        """
        self.engine = create_engine(db_url)
        self._era_cache = {}
        self._load_era_data()
    
    def _load_era_data(self):
        """Load era data from all three levels."""
        for level in ['a', 'b', 'c']:
            query = f"""
            SELECT 
                era_id,
                signal_name,
                level,
                stage,
                start_time,
                end_time,
                rows as era_duration_rows,
                EXTRACT(EPOCH FROM (end_time - start_time))/3600 as era_duration_hours
            FROM era_labels_level_{level}
            WHERE rows > 50  -- Minimum viable era size
            ORDER BY start_time
            """
            
            try:
                df = pd.read_sql(text(query), self.engine)
                self._era_cache[f'level_{level}'] = df
                logger.info(f"Loaded {len(df)} eras from level {level.upper()}")
            except Exception as e:
                logger.warning(f"Could not load era data for level {level}: {e}")
    
    def get_context(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Get era context for a specific timestamp.
        
        Args:
            timestamp: Time point for context (defaults to current time)
            
        Returns:
            Dictionary with era context information
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        context = {
            'timestamp': timestamp,
            'era_levels': {}
        }
        
        # Find active era at each level
        for level in ['a', 'b', 'c']:
            level_key = f'level_{level}'
            if level_key not in self._era_cache:
                continue
                
            df = self._era_cache[level_key]
            
            # Find era containing this timestamp
            active_era = df[
                (df['start_time'] <= timestamp) & 
                (df['end_time'] >= timestamp)
            ]
            
            if not active_era.empty:
                era = active_era.iloc[0]
                context['era_levels'][level] = {
                    'era_id': era['era_id'],
                    'signal_name': era['signal_name'],
                    'stage': era['stage'],
                    'duration_hours': era['era_duration_hours'],
                    'duration_rows': era['era_duration_rows'],
                    'progress': self._calculate_era_progress(
                        timestamp, era['start_time'], era['end_time']
                    )
                }
        
        # Add hierarchical relationships
        context['era_hierarchy'] = self._get_era_hierarchy(context['era_levels'])
        
        # Add era-based recommendations
        context['era_recommendations'] = self._get_era_recommendations(context['era_levels'])
        
        return context
    
    def _calculate_era_progress(self, current: datetime, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """Calculate how far through an era we are (0.0 to 1.0)."""
        total_duration = (end - start).total_seconds()
        if total_duration == 0:
            return 0.5
        
        elapsed = (current - start).total_seconds()
        return min(1.0, max(0.0, elapsed / total_duration))
    
    def _get_era_hierarchy(self, era_levels: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze hierarchical relationships between era levels."""
        hierarchy = {}
        
        # Check if we're in a stable period (all levels agree)
        if all(level in era_levels for level in ['a', 'b', 'c']):
            signals = [era_levels[level]['signal_name'] for level in ['a', 'b', 'c']]
            hierarchy['stability'] = 'high' if len(set(signals)) == 1 else 'mixed'
        
        # Analyze time scales
        if 'a' in era_levels and 'c' in era_levels:
            hierarchy['time_scale_ratio'] = (
                era_levels['a']['duration_hours'] / 
                (era_levels['c']['duration_hours'] + 1e-6)
            )
        
        return hierarchy
    
    def _get_era_recommendations(self, era_levels: Dict[str, Dict]) -> Dict[str, Any]:
        """Get optimization recommendations based on era characteristics."""
        recommendations = {}
        
        # Level A (macro) recommendations
        if 'a' in era_levels:
            era_a = era_levels['a']
            if era_a['duration_hours'] > 168:  # Longer than a week
                recommendations['strategy'] = 'long_term_stable'
                recommendations['update_frequency'] = 'daily'
            else:
                recommendations['strategy'] = 'medium_term_adaptive'
                recommendations['update_frequency'] = 'twice_daily'
        
        # Level C (micro) recommendations  
        if 'c' in era_levels:
            era_c = era_levels['c']
            if era_c['duration_hours'] < 24:  # Less than a day
                recommendations['responsiveness'] = 'high'
                recommendations['control_precision'] = 'fine'
            else:
                recommendations['responsiveness'] = 'moderate'
                recommendations['control_precision'] = 'standard'
        
        # Signal-based recommendations
        for level, era in era_levels.items():
            signal = era['signal_name']
            if 'temp' in signal.lower():
                recommendations['temperature_sensitivity'] = 'high'
            elif 'light' in signal.lower() or 'dli' in signal.lower():
                recommendations['light_optimization'] = 'critical'
            elif 'co2' in signal.lower():
                recommendations['co2_enrichment'] = 'beneficial'
        
        return recommendations


class MultiEraOptimizationStrategy:
    """Strategy for using multi-era information in optimization."""
    
    def __init__(self, era_context_provider: EraContextProvider):
        self.context_provider = era_context_provider
    
    def adjust_bounds_for_era(
        self, 
        base_bounds: Dict[str, tuple], 
        era_context: Dict[str, Any]
    ) -> Dict[str, tuple]:
        """Adjust decision variable bounds based on era context.
        
        Args:
            base_bounds: Original bounds for decision variables
            era_context: Era context from provider
            
        Returns:
            Adjusted bounds
        """
        adjusted_bounds = base_bounds.copy()
        recommendations = era_context.get('era_recommendations', {})
        
        # Tighten bounds for fine control periods
        if recommendations.get('control_precision') == 'fine':
            for var in ['temperature_setpoint', 'humidity_setpoint']:
                if var in adjusted_bounds:
                    low, high = adjusted_bounds[var]
                    center = (low + high) / 2
                    range_reduction = 0.8  # Reduce range by 20%
                    new_range = (high - low) * range_reduction
                    adjusted_bounds[var] = (
                        center - new_range / 2,
                        center + new_range / 2
                    )
        
        # Adjust for light-critical periods
        if recommendations.get('light_optimization') == 'critical':
            if 'light_intensity' in adjusted_bounds:
                # Increase minimum light intensity
                low, high = adjusted_bounds['light_intensity']
                adjusted_bounds['light_intensity'] = (low * 1.2, high)
        
        return adjusted_bounds
    
    def get_objective_weights_for_era(
        self,
        era_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get objective weights based on era context.
        
        Args:
            era_context: Era context from provider
            
        Returns:
            Objective weight adjustments
        """
        weights = {
            'energy_consumption': 1.0,
            'plant_growth': 1.0,
            'water_usage': 1.0,
            'crop_quality': 1.0
        }
        
        hierarchy = era_context.get('era_hierarchy', {})
        
        # Long-term stable periods: emphasize efficiency
        if hierarchy.get('stability') == 'high':
            weights['energy_consumption'] *= 1.2
            weights['water_usage'] *= 1.1
        
        # Rapid change periods: emphasize growth and quality
        elif hierarchy.get('stability') == 'mixed':
            weights['plant_growth'] *= 1.3
            weights['crop_quality'] *= 1.2
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}