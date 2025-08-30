#preprocessing module for ECG signals
import numpy as np
import scipy.signal as signal
from scipy import ndimage
import matplotlib.pyplot as plt

class ECGPreprocessor:
    def __init__(self, sampling_rate=250):
        """
        Initialize ECG preprocessor
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 250 Hz)
        """
        self.sampling_rate = sampling_rate
        
    def remove_baseline_wander(self, ecg_data, method='polynomial'):
        """
        Remove baseline wander using different methods
        
        Args:
            ecg_data: Raw ECG signal
            method: 'polynomial', 'median', or 'wavelet'
        
        Returns:
            Baseline-corrected ECG signal
        """
        if method == 'polynomial':
            # Polynomial fitting method
            x = np.arange(len(ecg_data))
            # Fit 3rd order polynomial
            coeffs = np.polyfit(x, ecg_data, 3)
            baseline = np.polyval(coeffs, x)
            return ecg_data - baseline
            
        elif method == 'median':
            # Median filtering method
            window_size = int(0.2 * self.sampling_rate)  # 200ms window
            if window_size % 2 == 0:
                window_size += 1
            baseline = signal.medfilt(ecg_data, window_size)
            return ecg_data - baseline
            
        elif method == 'wavelet':
            # Wavelet-based baseline removal
            import pywt
            # Decompose signal
            coeffs = pywt.wavedec(ecg_data, 'db4', level=8)
            # Remove low-frequency components (baseline)
            coeffs[0] = np.zeros_like(coeffs[0])
            # Reconstruct signal
            return pywt.waverec(coeffs, 'db4')
    
    def remove_powerline_noise(self, ecg_data, powerline_freq=50):
        """
        Remove powerline interference (50/60 Hz)
        
        Args:
            ecg_data: ECG signal
            powerline_freq: Powerline frequency (50 Hz for Europe, 60 Hz for US)
        
        Returns:
            Powerline noise removed signal
        """
        # Design notch filter
        nyquist = self.sampling_rate / 2
        low = (powerline_freq - 2) / nyquist
        high = (powerline_freq + 2) / nyquist
        
        # Butterworth notch filter
        b, a = signal.butter(4, [low, high], btype='bandstop')
        return signal.filtfilt(b, a, ecg_data)
    
    def remove_high_frequency_noise(self, ecg_data, cutoff_freq=40):
        """
        Remove high-frequency noise using low-pass filter
        
        Args:
            ecg_data: ECG signal
            cutoff_freq: Cutoff frequency in Hz
        
        Returns:
            High-frequency noise removed signal
        """
        nyquist = self.sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Butterworth low-pass filter
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, ecg_data)
    
    def remove_motion_artifacts(self, ecg_data, window_size=0.5):
        """
        Remove motion artifacts using adaptive thresholding
        
        Args:
            ecg_data: ECG signal
            window_size: Window size in seconds
        
        Returns:
            Motion artifact removed signal
        """
        window_samples = int(window_size * self.sampling_rate)
        
        # Calculate moving standard deviation
        moving_std = ndimage.uniform_filter1d(ecg_data**2, window_samples)
        moving_std = np.sqrt(moving_std)
        
        # Calculate adaptive threshold
        threshold = np.mean(moving_std) + 2 * np.std(moving_std)
        
        # Identify motion artifacts
        artifact_mask = moving_std > threshold
        
        # Replace artifacts with interpolated values
        clean_signal = ecg_data.copy()
        if np.any(artifact_mask):
            # Find artifact boundaries
            artifact_starts = np.where(np.diff(artifact_mask.astype(int)) == 1)[0]
            artifact_ends = np.where(np.diff(artifact_mask.astype(int)) == -1)[0]
            
            # Handle edge cases
            if len(artifact_starts) > len(artifact_ends):
                artifact_ends = np.append(artifact_ends, len(ecg_data))
            if len(artifact_ends) > len(artifact_starts):
                artifact_starts = np.insert(artifact_starts, 0, 0)
            
            # Interpolate each artifact segment
            for start, end in zip(artifact_starts, artifact_ends):
                if start > 0 and end < len(ecg_data):
                    # Linear interpolation
                    clean_signal[start:end] = np.linspace(
                        clean_signal[start-1], clean_signal[end], end-start
                    )
        
        return clean_signal
    
    def normalize_signal(self, ecg_data, method='zscore'):
        """
        Normalize ECG signal
        
        Args:
            ecg_data: ECG signal
            method: 'zscore', 'minmax', or 'robust'
        
        Returns:
            Normalized signal
        """
        if method == 'zscore':
            # Z-score normalization
            return (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)
            
        elif method == 'minmax':
            # Min-max normalization
            return (ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))
            
        elif method == 'robust':
            # Robust normalization using median and MAD
            median = np.median(ecg_data)
            mad = np.median(np.abs(ecg_data - median))
            return (ecg_data - median) / (1.4826 * mad)  # 1.4826 is constant for normal distribution
    
    def complete_preprocessing(self, ecg_data, visualize=False):
        """
        Complete preprocessing pipeline
        
        Args:
            ecg_data: Raw ECG signal
            visualize: Whether to show preprocessing steps
        
        Returns:
            Preprocessed ECG signal
        """
        original_signal = ecg_data.copy()
        
        if visualize:
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            axes[0, 0].plot(original_signal)
            axes[0, 0].set_title('Original Signal')
            axes[0, 0].grid(True)
        
        # Step 1: Remove baseline wander
        print("1. Removing baseline wander...")
        ecg_data = self.remove_baseline_wander(ecg_data, method='polynomial')
        
        if visualize:
            axes[0, 1].plot(ecg_data)
            axes[0, 1].set_title('After Baseline Removal')
            axes[0, 1].grid(True)
        
        # Step 2: Remove powerline noise
        print("2. Removing powerline noise...")
        ecg_data = self.remove_powerline_noise(ecg_data, powerline_freq=50)
        
        if visualize:
            axes[1, 0].plot(ecg_data)
            axes[1, 0].set_title('After Powerline Removal')
            axes[1, 0].grid(True)
        
        # Step 3: Remove high-frequency noise
        print("3. Removing high-frequency noise...")
        ecg_data = self.remove_high_frequency_noise(ecg_data, cutoff_freq=40)
        
        if visualize:
            axes[1, 1].plot(ecg_data)
            axes[1, 1].set_title('After High-Freq Removal')
            axes[1, 1].grid(True)
        
        # Step 4: Remove motion artifacts
        print("4. Removing motion artifacts...")
        ecg_data = self.remove_motion_artifacts(ecg_data, window_size=0.5)
        
        if visualize:
            axes[2, 0].plot(ecg_data)
            axes[2, 0].set_title('After Motion Artifact Removal')
            axes[2, 0].grid(True)
        
        # Step 5: Normalize signal
        print("5. Normalizing signal...")
        ecg_data = self.normalize_signal(ecg_data, method='zscore')
        
        if visualize:
            axes[2, 1].plot(ecg_data)
            axes[2, 1].set_title('After Normalization')
            axes[2, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig('ecg_preprocessing_steps.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Preprocessing completed!")
        return ecg_data
    
    def calculate_snr(self, original_signal, processed_signal):
        """
        Calculate Signal-to-Noise Ratio improvement
        
        Args:
            original_signal: Original noisy signal
            processed_signal: Processed signal
        
        Returns:
            SNR improvement in dB
        """
        # Estimate noise as difference between original and processed
        noise = original_signal - processed_signal
        
        # Calculate SNR
        signal_power = np.mean(processed_signal**2)
        noise_power = np.mean(noise**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return snr
        else:
            return float('inf')

def demo_preprocessing():
    """Demo function to show preprocessing effects"""
    print("ECG Preprocessing Demo")
    print("=" * 40)
    
    # Create synthetic ECG with noise
    t = np.linspace(0, 10, 2500)
    
    # Clean ECG signal (simulated)
    clean_ecg = np.zeros(2500)
    for i in range(10):
        peak_time = i + 0.5
        peak_idx = int(peak_time * 250)
        if peak_idx < 2500:
            clean_ecg[peak_idx] = 1.0
            if peak_idx + 1 < 2500:
                clean_ecg[peak_idx + 1] = 0.5
            if peak_idx - 1 >= 0:
                clean_ecg[peak_idx - 1] = 0.5
    
    # Add different types of noise
    noisy_ecg = clean_ecg.copy()
    
    # Baseline wander
    baseline_wander = 0.3 * np.sin(2 * np.pi * 0.1 * t)
    noisy_ecg += baseline_wander
    
    # Powerline noise (50 Hz)
    powerline_noise = 0.2 * np.sin(2 * np.pi * 50 * t)
    noisy_ecg += powerline_noise
    
    # High-frequency noise
    high_freq_noise = 0.1 * np.random.randn(2500)
    noisy_ecg += high_freq_noise
    
    # Motion artifacts (spikes)
    motion_spikes = np.random.choice([0, 0.5], 2500, p=[0.95, 0.05])
    noisy_ecg += motion_spikes
    
    print("Created synthetic ECG with multiple noise types")
    print(f"Original signal range: [{np.min(clean_ecg):.3f}, {np.max(clean_ecg):.3f}]")
    print(f"Noisy signal range: [{np.min(noisy_ecg):.3f}, {np.max(noisy_ecg):.3f}]")
    
    # Apply preprocessing
    preprocessor = ECGPreprocessor(sampling_rate=250)
    processed_ecg = preprocessor.complete_preprocessing(noisy_ecg, visualize=True)
    
    # Calculate SNR improvement
    snr_improvement = preprocessor.calculate_snr(noisy_ecg, processed_ecg)
    print(f"\n📊 SNR Improvement: {snr_improvement:.2f} dB")
    
    # Compare with clean signal
    correlation = np.corrcoef(clean_ecg, processed_ecg)[0, 1]
    print(f"📊 Correlation with clean signal: {correlation:.3f}")
    
    return processed_ecg

if __name__ == "__main__":
    demo_preprocessing()



