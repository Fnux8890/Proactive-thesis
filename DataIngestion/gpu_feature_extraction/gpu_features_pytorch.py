#!/usr/bin/env python3
"""
GPU-accelerated feature extraction using PyTorch
Implements the features that were previously in CUDA kernels
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtendedStatistics:
    """GPU-computed extended statistics"""
    mean: float
    std: float
    min: float
    max: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    skewness: float
    kurtosis: float
    mad: float
    iqr: float
    entropy: float
    count: int


@dataclass
class WeatherCouplingFeatures:
    """Features capturing greenhouse-weather interactions"""
    temp_differential_mean: float
    temp_differential_std: float
    solar_efficiency: float
    weather_response_lag: float
    correlation_strength: float
    thermal_mass_indicator: float
    ventilation_effectiveness: float


@dataclass
class EnergyFeatures:
    """Energy consumption and cost features"""
    cost_weighted_consumption: float
    peak_offpeak_ratio: float
    hours_until_cheap: float
    energy_efficiency_score: float
    cost_per_degree_hour: float
    optimal_load_shift_hours: float


@dataclass
class GrowthFeatures:
    """Plant growth-related features"""
    growing_degree_days: float
    daily_light_integral: float
    photoperiod_hours: float
    temperature_optimality: float
    light_sufficiency: float
    stress_degree_days: float
    flowering_signal: float
    expected_growth_rate: float


class GPUFeatureExtractor:
    """GPU-accelerated feature extraction using PyTorch"""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize GPU feature extractor
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initialized GPU feature extractor on device: {self.device}")
    
    def compute_extended_statistics(self, data: np.ndarray) -> ExtendedStatistics:
        """Compute extended statistics on GPU
        
        Args:
            data: Input data array
            
        Returns:
            ExtendedStatistics object with computed values
        """
        if len(data) == 0:
            return ExtendedStatistics(
                mean=0, std=0, min=0, max=0, p5=0, p25=0, p50=0, p75=0, p95=0,
                skewness=0, kurtosis=0, mad=0, iqr=0, entropy=0, count=0
            )
        
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(data).float().to(self.device)
        
        # Basic statistics
        mean = tensor.mean().item()
        std = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        count = len(tensor)
        
        # Percentiles
        percentiles = torch.quantile(tensor, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]).to(self.device))
        p5, p25, p50, p75, p95 = percentiles.cpu().numpy()
        
        # Skewness and kurtosis
        centered = tensor - mean
        m2 = (centered ** 2).mean()
        m3 = (centered ** 3).mean()
        m4 = (centered ** 4).mean()
        
        skewness = m3 / (m2 ** 1.5) if m2 > 0 else 0
        kurtosis = (m4 / (m2 ** 2) - 3) if m2 > 0 else 0
        
        # Median absolute deviation
        median = p50
        mad = torch.median(torch.abs(tensor - median)).item()
        
        # Interquartile range
        iqr = p75 - p25
        
        # Shannon entropy (discretized)
        hist = torch.histc(tensor, bins=50)
        hist = hist[hist > 0]  # Remove zero bins
        probs = hist / hist.sum()
        entropy = -torch.sum(probs * torch.log2(probs)).item() if len(probs) > 0 else 0
        
        return ExtendedStatistics(
            mean=mean, std=std, min=min_val, max=max_val,
            p5=p5, p25=p25, p50=p50, p75=p75, p95=p95,
            skewness=skewness, kurtosis=kurtosis,
            mad=mad, iqr=iqr, entropy=entropy, count=count
        )
    
    def compute_weather_coupling(
        self,
        internal_temp: np.ndarray,
        external_temp: np.ndarray,
        solar_radiation: np.ndarray
    ) -> WeatherCouplingFeatures:
        """Compute weather coupling features on GPU"""
        
        # Convert to tensors
        int_temp = torch.from_numpy(internal_temp).float().to(self.device)
        ext_temp = torch.from_numpy(external_temp).float().to(self.device)
        solar_rad = torch.from_numpy(solar_radiation).float().to(self.device)
        
        # Temperature differential
        temp_diff = int_temp - ext_temp
        temp_diff_mean = temp_diff.mean().item()
        temp_diff_std = temp_diff.std().item()
        
        # Solar efficiency (temperature rise per unit radiation)
        if solar_rad.sum() > 0:
            solar_mask = solar_rad > 0
            solar_efficiency = (temp_diff[solar_mask] / solar_rad[solar_mask]).mean().item()
        else:
            solar_efficiency = 0.0
        
        # Cross-correlation for lag detection
        if len(int_temp) > 10:
            # Normalize signals
            int_norm = (int_temp - int_temp.mean()) / (int_temp.std() + 1e-8)
            ext_norm = (ext_temp - ext_temp.mean()) / (ext_temp.std() + 1e-8)
            
            # Compute cross-correlation using FFT
            correlation = torch.nn.functional.conv1d(
                int_norm.unsqueeze(0).unsqueeze(0),
                ext_norm.flip(0).unsqueeze(0).unsqueeze(0),
                padding=len(ext_norm)-1
            )
            
            # Find lag with maximum correlation
            max_corr_idx = correlation.argmax()
            weather_response_lag = (max_corr_idx - len(ext_norm) + 1).item()
            correlation_strength = correlation.max().item()
        else:
            weather_response_lag = 0.0
            correlation_strength = 0.0
        
        # Thermal mass indicator (slower response = higher thermal mass)
        thermal_mass_indicator = temp_diff_std / (ext_temp.std() + 1e-8)
        
        # Ventilation effectiveness (how quickly internal matches external)
        vent_effectiveness = 1.0 / (1.0 + temp_diff_std)
        
        return WeatherCouplingFeatures(
            temp_differential_mean=temp_diff_mean,
            temp_differential_std=temp_diff_std,
            solar_efficiency=solar_efficiency,
            weather_response_lag=weather_response_lag,
            correlation_strength=correlation_strength,
            thermal_mass_indicator=thermal_mass_indicator.item(),
            ventilation_effectiveness=vent_effectiveness.item()
        )
    
    def compute_energy_features(
        self,
        lamp_power: np.ndarray,
        heating_power: np.ndarray,
        energy_prices: np.ndarray
    ) -> EnergyFeatures:
        """Compute energy-related features on GPU"""
        
        # Convert to tensors
        lamp = torch.from_numpy(lamp_power).float().to(self.device)
        heating = torch.from_numpy(heating_power).float().to(self.device)
        prices = torch.from_numpy(energy_prices).float().to(self.device)
        
        # Total power consumption
        total_power = lamp + heating
        
        # Cost-weighted consumption
        cost_weighted = (total_power * prices).sum().item()
        
        # Peak vs off-peak ratio
        price_threshold = prices.quantile(0.75).item()
        peak_mask = prices > price_threshold
        
        if peak_mask.sum() > 0 and (~peak_mask).sum() > 0:
            peak_consumption = total_power[peak_mask].mean().item()
            offpeak_consumption = total_power[~peak_mask].mean().item()
            peak_offpeak_ratio = peak_consumption / (offpeak_consumption + 1e-8)
        else:
            peak_offpeak_ratio = 1.0
        
        # Hours until cheap electricity
        current_price = prices[-1] if len(prices) > 0 else 0
        cheap_threshold = prices.quantile(0.25).item()
        
        if current_price > cheap_threshold:
            # Find next cheap period
            future_cheap = torch.where(prices < cheap_threshold)[0]
            if len(future_cheap) > 0:
                hours_until_cheap = future_cheap[0].item()
            else:
                hours_until_cheap = float('inf')
        else:
            hours_until_cheap = 0.0
        
        # Energy efficiency score
        if total_power.sum() > 0:
            efficiency_score = 1.0 / (cost_weighted / total_power.sum().item())
        else:
            efficiency_score = 0.0
        
        # Cost per degree-hour (simplified)
        cost_per_degree_hour = cost_weighted / (24 * 20)  # Assume 20°C average
        
        # Optimal load shift potential
        sorted_prices, _ = prices.sort()
        cheapest_hours = len(prices) // 4  # Bottom 25%
        potential_savings = (prices.mean() - sorted_prices[:cheapest_hours].mean()).item()
        optimal_load_shift = potential_savings * total_power.mean().item()
        
        return EnergyFeatures(
            cost_weighted_consumption=cost_weighted,
            peak_offpeak_ratio=peak_offpeak_ratio,
            hours_until_cheap=hours_until_cheap,
            energy_efficiency_score=efficiency_score,
            cost_per_degree_hour=cost_per_degree_hour,
            optimal_load_shift_hours=optimal_load_shift
        )
    
    def compute_growth_features(
        self,
        temperature: np.ndarray,
        light_intensity: np.ndarray,
        photoperiod: np.ndarray,
        base_temp: float = 10.0,
        optimal_temp_day: float = 22.0,
        optimal_temp_night: float = 18.0,
        light_requirement: float = 150.0
    ) -> GrowthFeatures:
        """Compute plant growth features on GPU"""
        
        # Convert to tensors
        temp = torch.from_numpy(temperature).float().to(self.device)
        light = torch.from_numpy(light_intensity).float().to(self.device)
        photo = torch.from_numpy(photoperiod).float().to(self.device)
        
        # Growing degree days
        gdd = torch.clamp(temp - base_temp, min=0).sum().item() / 24.0
        
        # Daily light integral (mol/m²/day)
        dli = light.sum().item() * 3600 / 1e6  # Convert µmol/s to mol
        
        # Photoperiod hours
        photoperiod_hours = photo.sum().item()
        
        # Temperature optimality
        day_mask = light > 0
        night_mask = ~day_mask
        
        if day_mask.sum() > 0:
            day_temp_opt = 1 - torch.abs(temp[day_mask] - optimal_temp_day) / 10
            temp_optimality_day = torch.clamp(day_temp_opt, 0, 1).mean().item()
        else:
            temp_optimality_day = 0.0
            
        if night_mask.sum() > 0:
            night_temp_opt = 1 - torch.abs(temp[night_mask] - optimal_temp_night) / 10
            temp_optimality_night = torch.clamp(night_temp_opt, 0, 1).mean().item()
        else:
            temp_optimality_night = 0.0
            
        temperature_optimality = (temp_optimality_day + temp_optimality_night) / 2
        
        # Light sufficiency
        light_sufficiency = torch.clamp(light / light_requirement, 0, 1).mean().item()
        
        # Stress degree days (too hot or too cold)
        stress_hot = torch.clamp(temp - 30, min=0).sum().item() / 24.0
        stress_cold = torch.clamp(5 - temp, min=0).sum().item() / 24.0
        stress_degree_days = stress_hot + stress_cold
        
        # Flowering signal (simplified - based on photoperiod)
        if photoperiod_hours < 10:  # Short day plant
            flowering_signal = 1.0
        elif photoperiod_hours > 14:  # Long day plant
            flowering_signal = 0.0
        else:
            flowering_signal = 0.5
        
        # Expected growth rate (simplified model)
        expected_growth_rate = (
            temperature_optimality * 
            light_sufficiency * 
            (1 - stress_degree_days / 10)
        )
        
        return GrowthFeatures(
            growing_degree_days=gdd,
            daily_light_integral=dli,
            photoperiod_hours=photoperiod_hours,
            temperature_optimality=temperature_optimality,
            light_sufficiency=light_sufficiency,
            stress_degree_days=stress_degree_days,
            flowering_signal=flowering_signal,
            expected_growth_rate=expected_growth_rate
        )
    
    def extract_multiresolution_features(
        self,
        data: np.ndarray,
        resolutions: List[int] = [1, 6, 24, 168]  # 1h, 6h, 1d, 1w
    ) -> Dict[str, ExtendedStatistics]:
        """Extract features at multiple temporal resolutions"""
        
        features = {}
        tensor = torch.from_numpy(data).float().to(self.device)
        
        for res in resolutions:
            if len(tensor) >= res:
                # Downsample by averaging
                downsampled = tensor.unfold(0, res, res).mean(dim=1)
                stats = self.compute_extended_statistics(downsampled.cpu().numpy())
                features[f"res_{res}h"] = stats
            else:
                # Not enough data for this resolution
                features[f"res_{res}h"] = ExtendedStatistics(
                    mean=0, std=0, min=0, max=0, p5=0, p25=0, p50=0, p75=0, p95=0,
                    skewness=0, kurtosis=0, mad=0, iqr=0, entropy=0, count=0
                )
        
        return features


def main():
    """Example usage of GPU feature extractor"""
    
    # Initialize extractor
    extractor = GPUFeatureExtractor()
    
    # Generate example data
    n_samples = 1000
    temperature = np.random.normal(22, 3, n_samples)
    external_temp = np.random.normal(15, 5, n_samples)
    solar_radiation = np.random.uniform(0, 1000, n_samples)
    
    # Compute features
    stats = extractor.compute_extended_statistics(temperature)
    logger.info(f"Extended statistics: {stats}")
    
    weather_coupling = extractor.compute_weather_coupling(
        temperature, external_temp, solar_radiation
    )
    logger.info(f"Weather coupling features: {weather_coupling}")
    
    # Multi-resolution features
    multi_res = extractor.extract_multiresolution_features(temperature)
    for res, features in multi_res.items():
        logger.info(f"{res} statistics: mean={features.mean:.2f}, std={features.std:.2f}")


if __name__ == "__main__":
    main()